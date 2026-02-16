#!/usr/bin/env python
"""Submit all shuffle estimation test jobs to Anyscale.

This script submits jobs 1-6 with appropriate cluster configurations.
Jobs 1-3 can run in parallel (smaller clusters).
Jobs 4-6 require larger clusters and more resources.

Usage:
    python submit_all_jobs.py [--skip-large] [--jobs 1,2,3]

Options:
    --skip-large    Skip jobs 4, 5, 6 (large scale tests)
    --jobs          Comma-separated list of job numbers to run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Job configurations
JOBS = {
    1: {
        "name": "shuffle-est-scale-datatypes",
        "script": "job1_scale_and_datatypes.py",
        "compute_config": {
            "head_node": {"instance_type": "m5.2xlarge"},
            "worker_nodes": [
                {"instance_type": "m5.2xlarge", "min_nodes": 7, "max_nodes": 7}
            ],
        },
        "description": "Scale (1GB, 10GB) and data type tests",
        "estimated_runtime_min": 30,
    },
    2: {
        "name": "shuffle-est-skew-cardinality",
        "script": "job2_skew_and_cardinality.py",
        "compute_config": {
            "head_node": {"instance_type": "m5.2xlarge"},
            "worker_nodes": [
                {"instance_type": "m5.2xlarge", "min_nodes": 7, "max_nodes": 7}
            ],
        },
        "description": "Skew distribution and key cardinality tests",
        "estimated_runtime_min": 30,
    },
    3: {
        "name": "shuffle-est-joins",
        "script": "job3_joins.py",
        "compute_config": {
            "head_node": {"instance_type": "m5.2xlarge"},
            "worker_nodes": [
                {"instance_type": "m5.2xlarge", "min_nodes": 15, "max_nodes": 15}
            ],
        },
        "description": "Join operation tests",
        "estimated_runtime_min": 45,
    },
    4: {
        "name": "shuffle-est-large-scale",
        "script": "job4_large_scale.py",
        "compute_config": {
            "head_node": {"instance_type": "m5.4xlarge"},
            "worker_nodes": [
                {"instance_type": "m5.4xlarge", "min_nodes": 15, "max_nodes": 15}
            ],
        },
        "description": "Large scale tests (100GB)",
        "estimated_runtime_min": 60,
    },
    5: {
        "name": "shuffle-est-memory-pressure",
        "script": "job5_memory_pressure.py",
        "compute_config": {
            "head_node": {"instance_type": "m5.2xlarge"},
            "worker_nodes": [
                {"instance_type": "m5.2xlarge", "min_nodes": 7, "max_nodes": 7}
            ],
        },
        "description": "Memory pressure and edge case tests",
        "estimated_runtime_min": 30,
    },
    6: {
        "name": "shuffle-est-very-large",
        "script": "job6_very_large_scale.py",
        "compute_config": {
            "head_node": {"instance_type": "m5.8xlarge"},
            "worker_nodes": [
                {"instance_type": "m5.8xlarge", "min_nodes": 31, "max_nodes": 31}
            ],
        },
        "description": "Very large scale tests (500GB)",
        "estimated_runtime_min": 90,
    },
}

# Dockerfile with shuffle estimation feature
DOCKERFILE = """
FROM anyscale/ray:2.44.1-py311

ENV ANYSCALE_DISABLE_OPTIMIZED_RAY=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
USER ray

WORKDIR /home/ray
RUN git clone --depth 1 --branch feature/data-joins https://github.com/robertnishihara/ray.git ray-dev

RUN pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl"

RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \\
    cp /home/ray/ray-dev/python/ray/data/_internal/shuffle_memory_estimate.py $SITE_PACKAGES/ray/data/_internal/ && \\
    cp /home/ray/ray-dev/python/ray/data/_internal/shuffle_memory_estimator.py $SITE_PACKAGES/ray/data/_internal/ && \\
    cp /home/ray/ray-dev/python/ray/data/_internal/execution/operators/hash_shuffle.py $SITE_PACKAGES/ray/data/_internal/execution/operators/ && \\
    cp /home/ray/ray-dev/python/ray/data/dataset.py $SITE_PACKAGES/ray/data/ && \\
    cp /home/ray/ray-dev/python/ray/data/context.py $SITE_PACKAGES/ray/data/

WORKDIR /home/ray
"""


def get_script_path(script_name: str) -> Path:
    """Get the full path to a job script."""
    return Path(__file__).parent / script_name


def print_job_summary():
    """Print summary of all jobs."""
    print("\nJob Summary:")
    print("-" * 80)
    total_time = 0
    for job_num, config in JOBS.items():
        nodes = config["compute_config"]["worker_nodes"][0]["min_nodes"] + 1
        instance = config["compute_config"]["worker_nodes"][0]["instance_type"]
        runtime = config["estimated_runtime_min"]
        total_time += runtime
        print(f"  Job {job_num}: {config['name']}")
        print(f"          {config['description']}")
        print(f"          Cluster: {nodes}x {instance}, ~{runtime} min")
    print("-" * 80)
    print(
        f"Total estimated runtime (sequential): {total_time} min ({total_time/60:.1f} hours)"
    )
    print(
        f"If jobs 1,2,3,5 run in parallel: ~{max(JOBS[1]['estimated_runtime_min'], JOBS[2]['estimated_runtime_min'], JOBS[3]['estimated_runtime_min'], JOBS[5]['estimated_runtime_min']) + JOBS[4]['estimated_runtime_min'] + JOBS[6]['estimated_runtime_min']} min"
    )
    print()


def submit_job(job_num: int, dry_run: bool = False) -> bool:
    """Submit a single job to Anyscale."""
    if job_num not in JOBS:
        print(f"Error: Unknown job number {job_num}")
        return False

    config = JOBS[job_num]
    script_path = get_script_path(config["script"])

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False

    print(f"\nSubmitting Job {job_num}: {config['name']}")
    print(f"  Script: {script_path}")
    print(f"  Description: {config['description']}")

    # Build anyscale job submit command
    # Note: This uses the Anyscale CLI - adjust as needed for your setup
    cmd = [
        "anyscale",
        "job",
        "submit",
        "--name",
        config["name"],
        "--image-uri",
        "anyscale/ray:2.44.1-py311",  # Base image
        "--compute-config",
        json.dumps(config["compute_config"]),
        "--entrypoint",
        f"python {config['script']}",
        "--working-dir",
        str(script_path.parent),
    ]

    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  SUCCESS: Job submitted")
            print(f"  {result.stdout}")
            return True
        else:
            print(f"  FAILED: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error submitting job: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit shuffle estimation test jobs to Anyscale"
    )
    parser.add_argument("--skip-large", action="store_true", help="Skip jobs 4, 5, 6")
    parser.add_argument(
        "--jobs", type=str, help="Comma-separated list of job numbers (e.g., 1,2,3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print job summary and exit"
    )
    args = parser.parse_args()

    print_job_summary()

    if args.summary:
        return 0

    # Determine which jobs to run
    if args.jobs:
        job_nums = [int(j.strip()) for j in args.jobs.split(",")]
    elif args.skip_large:
        job_nums = [1, 2, 3]
    else:
        job_nums = list(JOBS.keys())

    print(f"Jobs to submit: {job_nums}")

    if args.dry_run:
        print("\n*** DRY RUN - No jobs will be submitted ***\n")

    # Submit jobs
    success_count = 0
    for job_num in job_nums:
        if submit_job(job_num, dry_run=args.dry_run):
            success_count += 1

    print(f"\nSubmitted {success_count}/{len(job_nums)} jobs")

    return 0 if success_count == len(job_nums) else 1


if __name__ == "__main__":
    sys.exit(main())
