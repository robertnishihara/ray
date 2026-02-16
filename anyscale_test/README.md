# Testing Shuffle Memory Estimation on Anyscale

This directory contains files to test the shuffle memory estimation feature on Anyscale.

## Files

- `Dockerfile` - Container definition that:
  - Disables Anyscale Runtime (`ANYSCALE_DISABLE_OPTIMIZED_RAY=1`)
  - Clones the `feature/data-joins` branch from `robertnishihara/ray`
  - Installs Ray nightly wheel
  - Sets up local development with `setup-dev.py`

- `test_shuffle_estimate.py` - Test script demonstrating the feature
- `anyscale_job.yaml` - Anyscale job configuration

## Running on Anyscale

### Option 1: Using Anyscale CLI

```bash
cd anyscale_test

# Submit the job (builds image from Dockerfile)
anyscale job submit --config-file anyscale_job.yaml
```

### Option 2: Using Anyscale Workspace

1. Create a new workspace with a base Ray image
2. In the workspace terminal:

```bash
# Disable Anyscale Runtime
export ANYSCALE_DISABLE_OPTIMIZED_RAY=1

# Clone the branch
git clone --depth 1 --branch feature/data-joins https://github.com/robertnishihara/ray.git ~/ray-dev

# Install nightly wheel
pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl"

# Set up local development
cd ~/ray-dev && python python/ray/setup-dev.py -y

# Run the test
cd ~ && python test_shuffle_estimate.py
```

### Option 3: Direct Job Submit

```bash
# From this directory
anyscale job submit \
  --containerfile Dockerfile \
  --working-dir . \
  --entrypoint "python test_shuffle_estimate.py" \
  --env-var ANYSCALE_DISABLE_OPTIMIZED_RAY=1
```

## Expected Output

The test script will:
1. Test `repartition(estimate_only=True)` with a skewed dataset
2. Test `join(estimate_only=True)` with two datasets
3. Test skew detection with highly skewed data

Each test prints a detailed summary including:
- Partition size distribution
- Skew metrics and warnings
- Per-aggregator memory breakdown
- Recommended `aggregator_ray_remote_args`

## GitHub Branch

The changes are on: https://github.com/robertnishihara/ray/tree/feature/data-joins

Files added/modified:
- `python/ray/data/_internal/shuffle_memory_estimate.py` (new)
- `python/ray/data/_internal/shuffle_memory_estimator.py` (new)
- `python/ray/data/dataset.py` (modified - added `estimate_only` param)
- `python/ray/data/tests/test_shuffle_memory_estimate.py` (new)
