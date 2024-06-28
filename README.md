# What is your VLM saying about your Image?

Visual Language Models (VLM's) use a pretrained LLM for their core smarts, but take images as inputs.  There are many variations on how to do this, but nowadays they've settled into a standard pattern which is fairly straightforward, depicted below.  Images are first prepared and sent through a neural network that is pre-trained for image analysis - typically a ViT like CLIP.  ViTs break the image down into patches that are maybe 14x14 pixels, and each patch gets converted to its own vector at the output.  The VLM than translates these image vectors into the same embedding space as word tokens, and sends them into the LLM for analysis.

# Try it yourself

## Run it in Google Colab

Warning - this will take a **solid 5 minutes** to load before you can use it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pifanpi/visualizing-vlm-tokens/blob/main/run-in-colab.ipynb)

## Run on your own GPU machine

Either from the command line:

```
./install-requirements.sh
./imgtokens.py "https://www.mcgawgraphics.com/cdn/shop/products/O360PF_1024x1024.jpg?v=1662584163"
```

Or you can see more than one word per token, and change the output size like:

```
./imgtokens.py "https://i0.wp.com/champagnecoloredglasses.com/wp-content/uploads/2017/06/IMGP3124.jpg?resize=2000%2C1335&ssl=1" --num-words=2 --size=1500
```

Or running the [notebook](Visualizing%20VLM%20Tokens.ipynb).
