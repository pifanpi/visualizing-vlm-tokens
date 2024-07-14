#!/usr/bin/env python
from io import BytesIO
import json
from functools import lru_cache

from PIL import Image
from PIL import ImageDraw, ImageFont
from imgcat import imgcat
from transformers import BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
import requests
import torch
import torch.nn.functional as F

import vecmath

class BailOutWithData(RuntimeError):
    """A glorious hack!  This lets you bail out in the middle of somebody else's code 
    and just return the one thing you care about, and not worry about unwinding a deep call trace.
    """
    def __init__(self, data):
        self.data = data

class HackedLlavaNextReturnsImageTokens(LlavaNextForConditionalGeneration):
    """A totally broken version of LlavaNextForConditionalGeneration.  If you try to do anything
    with it, it will raise a BailOutWithData exception that gives you the raw image tokens
    projected into LLM space.
    """
    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        raise BailOutWithData(image_features[0])

class XiaoDict:
    """Small chinese dictionary
    """

    def __init__(self, filename: str = "cedict-top.json"):
        with open(filename, "r", encoding="utf-8") as f:
            self.dict = json.load(f)

    def augment(self, word: str) -> str:
        """Augments a word by adding a definition if it's in the dictionary.
        e.g. "天" -> "天 (sky)"
        """
        if word in self.dict:
            defn = self.dict[word]
            return f"{word} ({defn})"
        else:
            return word

XIAODICT = XiaoDict()


class WordList:
    """Represents all the discovered word-pieces associated with a token.
    Provides methods to render them.
    """

    def __init__(self):
        self.words = []
        self.strengths = []

    def append(self, word: str, strength: float = 1.0):
        """Add another word-piece to the list.
        """
        augmented = XIAODICT.augment(word)
        self.words.append(augmented)
        self.strengths.append(strength)

    def as_html(self) -> str:
        """Renders the words as HTML for a plotly tooltip
        """
        html = "<br>".join(f"{w} ({s:.3f})" for w, s in zip(self.words, self.strengths))
        return html

    def first(self) -> str:
        return self.words[0]

    def __str__(self) -> str:
        return "\n".join(self.words)


class ImagePatchWordTokenizer:
    """Uses a llava-style VLM to process an image into language-tokens, but projects them
    back onto the tokens so you get text instead of vectors.
    Can render a list of lists of words, or render them on the image - directly or interactively
    with plotly.
    """

    def __init__(self, model_name:str = "llava-hf/llava-v1.6-vicuna-7b-hf", use_4bit:bool = True, similarity:str="dot"):
        """
        :arg model_name: is the name of the model to use.  
            Options known to work are:
                llava-hf/llava-v1.6-mistral-7b-hf
                llava-hf/llava-v1.6-vicuna-7b-hf
        :arg similarity: is the similarity metric to use.  Can be "dot", "cosine", or "l2".
        """
        self.model_str = model_name
        self.use_4bit = use_4bit
        self.similarity = similarity
        self.processor = None 
        self.model = None
        self.img = None

    def _init_model(self):
        """Lazily load and initialize the model - this takes at least several seconds, and maybe a lot
        longer if it's not downloaded.  Useful in a notebook to do this lazily.
        """
        if self.processor is None:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_str, torch_dtype=torch.float16)
            self.processor.tokenizer.padding_side = "left"
        if self.model is None:
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                quantization_config = None
            self.model = HackedLlavaNextReturnsImageTokens.from_pretrained(
                self.model_str, cache_dir="", quantization_config=quantization_config
            )

    def _extract_image_features(self) -> torch.Tensor:
        self._init_model()
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"  # doesn't really matter
        inputs = self.processor(prompt, self.img, return_tensors="pt").to("cuda:0")
        try:
            self.model.generate(**inputs, max_new_tokens=10)
            raise RuntimeError("Should never get here")
        except BailOutWithData as e:
            data = e.data
            assert isinstance(data, torch.Tensor), f"Expected a tensor, got {type(data)}"
            return data

    def _closest_wordpieces(self, output_vectors: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Finds the k closest word-pieces to the output vectors.  Returns the indices and strengths.
        """
        # Now we have a bunch of predicted tokens, in vector (embedding) space.
        # To convert them to token-Ids, we dot them against the embedding matrix.
        #embedding_matrix = self.model.get_output_embeddings().weight
        # ^^ returns a strangely empty object which seems like it's all zeros.
        embedding_matrix = self.model.language_model.model.embed_tokens.weight  # <vocab, dim>
        # embedding_matrix is shape <vocab, dim>
        # output_vectors is shape <tok, dim>

        if self.similarity == "dot":
            similarity_matrix = torch.matmul(output_vectors, embedding_matrix.T)  # <tok, vocab>
            invert = 1
        elif self.similarity == "cosine":
            similarity_matrix = vecmath.all_to_all_cosine_similarity(output_vectors, embedding_matrix)
            invert = 1
        elif self.similarity == "l2":
            similarity_matrix = vecmath.all_to_all_l2_distance(output_vectors, embedding_matrix)
            invert = -1  # these are distances, so we want to sort for small values
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")
        k_closest = torch.topk(similarity_matrix * invert, k=k, dim=-1).indices  # <tok, k>
        strengths = torch.gather(similarity_matrix, 1, k_closest)  # <tok, k>
        return k_closest, strengths

    def _vectors_to_words(self, output_vectors: torch.Tensor, num_words: int = 1) -> list[list[WordList]]:
        """Takes the output_vectors and converts them to a list of lists of strings.
        The list of lists represents the geometry of the image tokens.
        Each string is the num_words closest word-pieces in the vocabulary.
        """
        self._init_model()
        assert output_vectors.ndim == 2, f"Expect <tok, dim> shaped tensor, but got {output_vectors.shape}"
        numtok = output_vectors.shape[0]
        edge = int(numtok ** 0.5)
        assert edge ** 2 == numtok, f"Expected a perfect square-lengthed number of tokens (e.g. 576=24^2), but got {numtok}"

        k_closest, strengths = self._closest_wordpieces(output_vectors, num_words)

        # Now go through it as a square matrix and convert to a list of lists of strings.
        out = []
        for i in range(edge):
            line = []
            for j in range(edge):
                idx = i*edge + j
                entry = WordList()
                for k in range(num_words):
                    wordpiece = self.processor.tokenizer.decode(k_closest[idx][k])
                    entry.append(wordpiece, strengths[idx][k])
                line.append(entry)
            out.append(line)
        return out

    def _opposing_color_near(self, img: Image.Image, x: int, y: int, boxsize: int) -> tuple[int, int, int]:
        """Samples the color near the given pixel, and returns the color that is opposite it.
        """
        box = (x, y, x + boxsize, y + boxsize)
        box = (
            max(0, box[0]), max(0, box[1]),
            min(img.size[0], box[2]), min(img.size[1], box[3])
        )
        region = img.crop(box)
        # TODO: use numpy to make this faster, but it doesn't really matter
        pixels = list(region.getdata())
        avg_color_rgb = tuple(sum(col) // len(pixels) for col in zip(*pixels))
        avg_brightness = sum(avg_color_rgb) / 3
        if avg_brightness < 128:
            return (255, 255, 255)
        else:
            return (0, 0, 0)

    def _standardize_img(self, img: Image.Image, size: int = 1000) -> Image.Image:
        """Standardizes the image to a standard resolution, and ensures it's RGB.
        """
        scale = size / max(img.size)
        img = img.copy()
        img = img.convert("RGB")
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        return img

    def set_image(self, img: Image.Image):
        self.img = self._standardize_img(img)

    def render_words_on_image(self, words: list[list[str]], size:int=1000) -> Image.Image:
        """Renders the words on the image, in the center of each cell.
        Returns a new image.
        """
        img = self._standardize_img(self.img, size)
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(img)
        grid_size = img.size[0] // len(words), img.size[1] // len(words[0])
        for i, row in enumerate(words):
            for j, wordlist in enumerate(row):
                text = str(wordlist)
                # Calculate the position to draw the text
                x = j * grid_size[0]
                y = i * grid_size[1]
                # Calculate the position to draw the text in the middle of the cell
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x + (grid_size[0] - text_width) / 2
                text_y = y + (grid_size[1] - text_height) / 2
                # Draw the text
                text_color = self._opposing_color_near(img, x, y, grid_size[0] // 2)
                draw.text((text_x, text_y), text, font=font, fill=text_color)
        return img

    def process_img(self, img: Image.Image, num_words: int = 1) -> list[list[str]]:
        """Takes an image, and returns a visualization of the image tokens.
        """
        self.set_image(img)
        tokens = self._extract_image_features()
        words = self._vectors_to_words(tokens[0], num_words)
        return words

    def draw_with_plotly(self, words: list[list[str]], size: int = 1500, iframe:bool = True) -> "go.Figure":
        """Renders the image, and overlays a grid with words in it, in iPython notebook using plotly
        :param words: the word list returned by process_img
        :param size: the size of the image to render
        :param iframe: if True, will use an iframe to render plotly, which works around a common jupyter problem
        :param show_plot: if True, will show the plot in a popup window
        :returns: a plotly figure
        """
        import plotly.express as px
        import pandas as pd
        import plotly.graph_objects as go
        if iframe:
            # A common workaround for jupyter problems
            import plotly.io as pio
            pio.renderers.default='iframe'

        # First just draw the image with plotly
        img = self._standardize_img(self.img, size=size)

        # Plotly likes pandas dataframes, so let's load the words into a dataframe
        data = []
        width_unit = img.width / len(words[0])
        height_unit = img.height / len(words)
        for i, row in enumerate(words):
            for j, word in enumerate(row):
                data.append({
                    # Add 0.5 to put it in the center of each cell
                    "x": (j + 0.5) * width_unit,
                    "y": (i + 0.5) * height_unit,
                    "firstword": word.first(),
                    "words": word.as_html(),
                    "font": dict(color="black", size=16)
                })
        df = pd.DataFrame(data)

        fig = px.imshow(img, width=img.width, height=img.height, binary_format="jpg")
        fig.add_trace(go.Scatter(
            x=df["x"], 
            y=df["y"], 
            text=df["firstword"], 
            hovertext=df["words"],
            mode="text", 
            textposition="bottom center",
            hoverinfo="text", # prevents pixel coordinates from showing up in hover
        ))
        return fig


def process_image_cli(img_url: str, num_words: int = 1, size: int = 1000, save_img: str = None, save_words: str = None):
    if img_url.startswith("http"):
        print(f"Fetching img from {img_url}")
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
    else:
        print(f"Trying to load img from local file {img_url}")
        img = Image.open(img_url)
    imgcat(img)
    print(f"Initializing model...")
    ipwt = ImagePatchWordTokenizer()
    print("Processing image...")
    words = ipwt.process_img(img, num_words)
    if save_words:
        words_serializable = [[str(w) for w in row] for row in words]
        with open(save_words, "w") as f:
            json.dump(words_serializable, f)
        print(f"Saved words to {save_words}")
    # render the words
    rendered = ipwt.render_words_on_image(words, size)
    imgcat(rendered, pixels_per_line=16)
    if save_img is not None:
        rendered.save(save_img, quality=95)
        print(f"Saved image to {save_img}")


def preload_models(model_str: str="llava-hf/llava-v1.6-vicuna-7b-hf"):
    """Preloads the models for the server.
    """
    # Note this doesn't seem to work properly for some reason.
    # The models download and cache appropriately, but
    # when an IPWT is created it downloads them again itself.
    print(f"Preloading model {model_str}")
    HackedLlavaNextReturnsImageTokens.from_pretrained(model_str)
    print(f"Preloading processor for {model_str}")
    LlavaNextProcessor.from_pretrained(model_str, torch_dtype=torch.float16)
    print("Done preloading models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image and visualize image tokens.")
    parser.add_argument("img_url", type=str, nargs='?', help="URL or local path of the image to process")
    parser.add_argument("--num-words", type=int, default=1, help="Number of words to display per token")
    parser.add_argument("--size", type=int, default=1000, help="Size to standardize the image to")
    parser.add_argument("--save-img", type=str, default=None, help="Path to save the image to")
    parser.add_argument("--save-words", type=str, default=None, help="Path to save the words to")
    parser.add_argument("--preload", action="store_true", help="Preload the models and exit")
    args = parser.parse_args()

    if args.preload:
        preload_models()
        print("Models preloaded successfully.")
        exit(0)
    elif args.img_url:
        process_image_cli(args.img_url, args.num_words, args.size, args.save_img, args.save_words)
    else:
        parser.print_help()
