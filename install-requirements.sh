#!/bin/bash
# This is the simplest possible install script.  It's deliberately not specific
# which makes it brittle in a sense - in the sense that changing versions might rot the code
# But it's going to run the fastest on a system like colab where a bunch of "good enough"
# versions are already installed.
pip install \
    torch \
    jupyter \
    notebook \
    transformers \
    datasets \
    hf-transfer \
    accelerate \
    imgcat \
    matplotlib \
    typer \
    sentencepiece \
    protobuf \
    plotly \
    imgcat \
    bitsandbytes
