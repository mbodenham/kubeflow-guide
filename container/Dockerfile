FROM nvcr.io/nvidia/pytorch:26.03-py3

# Install transformers and other dependencies
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir accelerate

COPY main.py /workspace/main.py
WORKDIR /workspace