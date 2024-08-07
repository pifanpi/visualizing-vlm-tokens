import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm


def all_to_all_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # This can overflow in fp16
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    a_norm = (a32**2).sum(1).view(-1, 1)
    b_norm = (b32**2).sum(1).view(1, -1)
    dist = a_norm + b_norm - 2.0 * torch.mm(a, b.transpose(0, 1))
    rms = torch.sqrt(torch.clamp(dist, min=0.0))
    return rms


def all_to_all_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    a_normalized = F.normalize(a32, p=2, dim=1)
    b_normalized = F.normalize(b32, p=2, dim=1)
    return torch.mm(a_normalized, b_normalized.t())


def regularized_least_squares(A: torch.Tensor, B: torch.Tensor, lambda_reg: float = 1.0) -> torch.Tensor:
    # A: basis vectors (32000 x 4096)
    # B: query vectors (500 x 4096)
    # lambda_reg: regularization parameter
    # Returns: coefficients (500 x 32000)
    m, n = A.shape
    ATA = torch.mm(A, A.T)  # This uses many GB of memory
    reg_term = lambda_reg * torch.eye(m, device=A.device)
    X = torch.linalg.solve(ATA + reg_term, torch.mm(A, B.T))
    return X.T


def sparse_reconstruction(A: torch.Tensor, B: torch.Tensor, num_iterations: int = 400, learning_rate: float = 0.03) -> torch.Tensor:
    """Explicitly solves for the coefficients of the basis vectors in the least squares sense,
    using an L1 regularization on the coefficients to promote sparsity.

    I have not seen this work well.
    """
    l1_lambda = 0.03  # L1 regularization strength
    # A: basis vectors (32000 x 4096)
    # B: query vectors (500 x 4096)
    m = A.shape[0]
    n = B.shape[0]
    device = A.device
    X = nn.Parameter(0.1 * torch.rand(n, m, device=device))
    optimizer = optim.AdamW([X], lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

    print(f"Initial L2 norm of B: {torch.norm(B, dim=1, p=2).mean():.4f}.  (If the err doesn't go below this, we're not learning anything.)")

    progress = tqdm(range(num_iterations))
    for _ in progress:
        optimizer.zero_grad()
        reconstruction = torch.matmul(X, A)
        err = reconstruction - B
        # Loss per token
        err_loss = torch.norm(err, dim=1, p=2)
        # Loss per basis vector
        l1_loss = torch.norm(X, dim=1, p=1)
        loss = err_loss + l1_lambda * l1_loss
        scalar_loss = loss.mean()
        progress.set_description(f"err={err_loss.mean():.4f} l1={l1_loss.mean():.4f}")
        scalar_loss.backward()
        optimizer.step()
        scheduler.step()

    return X.detach()


def batch_omp(queries: torch.Tensor, vocab: torch.Tensor, coef_cnt: int, pos_only: bool = False) -> torch.Tensor:
    """Uses orthogonal matching pursuit to find the best subset of basis vectors to reconstruct the query vectors.
    This is great!
    """
    # vocab: basis vectors e.g. (32046 x 4096)
    # queries: query vectors e.g. (576 x 4096)
    # coef_cnt: number of non-zero coefficients to use
    device = vocab.device
    m, d = vocab.shape  # m = 32046, d = 4096
    n, _ = queries.shape  # n = 576

    # X keeps track of the coefficients for each token, for each basis vector
    X = torch.zeros(m, n, device=device)
    # residual is the remaining error to be corrected
    residual = queries.clone()
    # selected_indices is the index of the basis vectors that are selected for each token
    selected_indices = torch.zeros(coef_cnt, n, dtype=torch.long, device=device)
    
    # Greedily add the most useful basis vector one at a time.
    for i in range(coef_cnt):
        corr = torch.mm(vocab, residual.t())  # correlations
        
        # Select the basis vector that best matches the residual
        _, indices = torch.max(torch.abs(corr), dim=0)
        selected_indices[i] = indices

        # For each token, run least-squares to find the best combination of
        # selected basis vectors, and update the residual
        for j in range(n):
            if pos_only:
                # Only adjust the most recently found token
                idx = selected_indices[i:i+1, j]
                target = residual[j]
            else:
                idx = selected_indices[:i+1, j]
                target = queries[j]
            vocab_selected = vocab[idx]
            sol = torch.linalg.lstsq(vocab_selected.T, target).solution
            sol = sol.detach()
            X[idx, j] = sol
            # TODO: This is much faster if we only used the selected basis vectors
            # instead of the whole vocab
            # residual[j] = queries[j] - torch.matmul(vocab_selected.t(), sol)
            residual[j] = queries[j] - torch.matmul(vocab.t(), X[:,j])
    
    return X.T


class SimilarityEngine:
    """Does the math to find a similarity matrix between two sets of vectors.
    One is the "vocabulary" which is the embeddings for the word-pieces.  Maybe <32k,4096>
    The other is the "tokens" which is the embedding for each image patch Maybe <524,4096>.
    The result is a similarity matrix of <524,32k> giving a score for every combination,
    of which you'll want to keep just the top k.

    We can think of this as a problem of simply finding the closest word-piece for each image patch.
    So we just need to use some similarity metric like cosine similarity, dot-product, or euclidean distance.

    You can also think of this as an under-specified decomposition problem.  The vocabulary can
    be viewed as an over-specified basis set, and we're trying to find the linear combination of
    basis elements that best approximate each token.
    """

    def __init__(self, vocab: torch.Tensor, allow_bad_algorithms:bool = False):
        # Convert to fp32 because many algorithms need it
        self.vocab = vocab.to(torch.float32)  # lots of things work better in fp32
        self.vocab_pinv = None
        self.allow_bad_algorithms = allow_bad_algorithms

    def similarity_matrix(self, tokens: torch.Tensor, method:str, cnt:int = 8) -> torch.Tensor:
        """Given a set of tokens <tokens, dim> compute the similarity matrix to the vocabulary.
        The result is a <tokens, vocab> matrix of similarity scores.
        :args cnt: is the number of basis vectors to use. (only used by omp)
        :args method: is a string that describes what method of similarity to use
          - "l2" is euclidean distance returned as -dist/sqrt(num_dims)
          - "cosine" is cosine similarity  (actually doesn't work very well)
          - "dot" is dot product similarity
          - "omp" uses orthogonal matching pursuit to find a subset of basis vectors

        The bad methods which are left in for curiosity sake, but not recommended:
          - "pinv" uses the pseudo-inverse of the vocab matrix to reconsruct each token vector
          - "sparse" runs an Adam optimization to find the best subset of basis vectors, L1-penalized for sparseness
            This is slow, and never works very well, but might if you tune it well.
          - "rls" uses regularized least squares to find the best subset of basis vectors
            This needs a lot of memory, and isn't even great.
        """
        tokens32 = tokens.to(torch.float32)
        dim = self.vocab.shape[1]
        if method == "l2":
            dist = all_to_all_l2_distance(tokens32, self.vocab)
            # large distance is less similar, so invert
            # Normalize by square root of the number of dimensions
            return -dist / torch.sqrt(torch.tensor(dim))
        elif method == "cosine":
            return all_to_all_cosine_similarity(tokens32, self.vocab)
        elif method == "dot":
            return torch.matmul(tokens32, self.vocab.T)
        elif method == "omp":
            return batch_omp(tokens32, self.vocab, cnt)
        elif method == "ompp":
            return batch_omp(tokens32, self.vocab, cnt, pos_only=True)
        
        if not self.allow_bad_algorithms:
            raise ValueError(f"Similarity method {method} is either unknown or too sucky to allow")
        if method == "pinv":
            if self.vocab_pinv is None:
                self.vocab_pinv = torch.pinverse(self.vocab)  # slow and doesn't work well
            return torch.matmul(tokens32, self.vocab_pinv)
        elif method == "sparse":
            # This one is quite slow, and need tuning to work well.
            return sparse_reconstruction(self.vocab, tokens32)
        elif method == "rls":
            # Uses lots of memory, and isn't even great.
            return regularized_least_squares(self.vocab, tokens32)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

