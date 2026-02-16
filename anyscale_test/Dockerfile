# Dockerfile for testing Ray Data shuffle memory estimation feature
# This disables Anyscale Runtime and sets up local Ray development

FROM anyscale/ray:2.44.1-py311

# Disable Anyscale Runtime to use custom Ray installation
ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1

# Fix protobuf version conflict between nightly wheel and base image
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install git and other dependencies
USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
USER ray

# Clone the fork with the shuffle memory estimation feature
WORKDIR /home/ray
RUN git clone --depth 1 --branch feature/data-joins https://github.com/robertnishihara/ray.git ray-dev

# Install the nightly wheel
RUN pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl"

# Copy only our modified files over the installed Ray package
# This avoids using setup-dev.py which would break the dashboard
RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
    cp /home/ray/ray-dev/python/ray/data/_internal/shuffle_memory_estimate.py $SITE_PACKAGES/ray/data/_internal/ && \
    cp /home/ray/ray-dev/python/ray/data/_internal/shuffle_memory_estimator.py $SITE_PACKAGES/ray/data/_internal/ && \
    cp /home/ray/ray-dev/python/ray/data/_internal/execution/operators/hash_shuffle.py $SITE_PACKAGES/ray/data/_internal/execution/operators/ && \
    cp /home/ray/ray-dev/python/ray/data/dataset.py $SITE_PACKAGES/ray/data/

WORKDIR /home/ray
