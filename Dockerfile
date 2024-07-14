FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python-is-python3 

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the model binaries into the container
# TODO: Figure out why preload_models() doesn't cache properly
# COPY imgtokens.py .
# COPY cedict-top.json .
# RUN python -u imgtokens.py --preload
# NOTE: on second thought, given how slow Docker registries are, and how often 
# we'd do this during development, and how fast S3 downloads are in prod, 
# we probably just don't bother preloading the models

# Copy everything else
COPY . .

# Set up the server
EXPOSE 8000
CMD ["python", "server.py"]
