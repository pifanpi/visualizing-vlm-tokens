import torch
import torch.nn.functional as F


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


def batch_regularized_least_squares(A: torch.Tensor, B: torch.Tensor, lambda_reg: float = 1.0) -> torch.Tensor:
    # A: basis vectors (32000 x 4096)
    # B: query vectors (4096 x 500)
    # lambda_reg: regularization parameter
    # Returns: coefficients (32000 x 500)
    m, n = A.shape
    ATA = torch.mm(A.t(), A)
    reg_term = lambda_reg * torch.eye(n, device=A.device)
    X = torch.linalg.solve(ATA + reg_term, torch.mm(A.t(), B))
    return X


def batch_l1_minimization(A: torch.Tensor, B: torch.Tensor, num_iterations: int = 1000, learning_rate: float = 0.01) -> torch.Tensor:
    # A: basis vectors (32000 x 4096)
    # B: query vectors (4096 x 500)
    X = nn.Parameter(torch.zeros(32000, 500))  # Initialize coefficients
    optimizer = optim.Adam([X], lr=learning_rate)
    loss_fn = nn.L1Loss()

    for _ in range(num_iterations):
        optimizer.zero_grad()
        reconstruction = A @ X
        loss = loss_fn(reconstruction, B) + 0.1 * torch.norm(X, 1)  # L1 regularization
        loss.backward()
        optimizer.step()

    return X.detach()

def batch_omp(queries: torch.Tensor, vocab: torch.Tensor, coef_cnt: int) -> torch.Tensor:
    """Uses orthogonal matching pursuit to find the best subset of basis vectors to reconstruct the query vectors.
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
            idx = selected_indices[:i+1, j]
            vocab_selected = vocab[idx]
            sol = torch.linalg.lstsq(vocab_selected.T, queries[j]).solution
            X[idx, j] = sol
            residual[j] = queries[j] - torch.matmul(vocab_selected.t(), sol)
    
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

    def __init__(self, vocab: torch.Tensor):
        # Convert to fp32 because many algorithms need it
        self.vocab = vocab.to(torch.float32)  # lots of things work better in fp32
        self.vocab_pinv = torch.pinverse(self.vocab)

    def similarity_matrix(self, tokens: torch.Tensor, method:str, cnt:int = 8) -> torch.Tensor:
        """Given a set of tokens <tokens, dim> compute the similarity matrix to the vocabulary.
        The result is a <tokens, vocab> matrix of similarity scores.
        :args cnt: is the number of basis vectors to use. (only used by omp)
        :args method: is a string that describes what method of similarity to use
          - "l2" is euclidean distance returned as -dist/sqrt(num_dims)
          - "cosine" is cosine similarity
          - "dot" is dot product similarity
          - "pinv" uses the pseudo-inverse of the vocab matrix to reconsruct each token vector
          - "omp" uses orthogonal matching pursuit to find a subset of basis vectors
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
        elif method == "pinv":
            return torch.matmul(tokens32, self.vocab_pinv)
        elif method == "omp":
            return batch_omp(tokens32, self.vocab, cnt)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

