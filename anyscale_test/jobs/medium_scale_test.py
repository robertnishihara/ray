#!/usr/bin/env python
"""Medium scale test for shuffle memory estimation.

Runs a 10GB shuffle with strict validation enabled.
If estimation is inaccurate, the job will fail.

Cluster recommendation: 8x m5.2xlarge (8 vCPU, 32GB each)
Expected runtime: ~10-15 minutes
"""

import time

import ray


def main():
    print("Initializing Ray...")
    ray.init()

    print(f"Ray version: {ray.__version__}")
    resources = ray.cluster_resources()
    print("Cluster resources:")
    print(f"  CPUs: {resources.get('CPU', 0)}")
    print(f"  Memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
    print(
        f"  Object store: {resources.get('object_store_memory', 0) / (1024**3):.1f} GB"
    )

    # Enable strict validation - will fail if estimation is wrong
    from ray.data.context import DataContext

    ctx = DataContext.get_current()
    ctx.strict_shuffle_estimation = True
    print("\nStrict shuffle estimation: ENABLED")

    # Test parameters
    # ~100 bytes per row -> 100M rows for 10GB
    NUM_ROWS = 100_000_000
    NUM_PARTITIONS = 200

    print(f"\n{'='*60}")
    print("Test: 10GB shuffle with uniform distribution")
    print(f"  Rows: {NUM_ROWS:,}")
    print(f"  Partitions: {NUM_PARTITIONS}")
    print(f"{'='*60}")

    try:
        # Create dataset with ~100 bytes per row
        print("\nCreating dataset...")
        ds = ray.data.range(NUM_ROWS).map(
            lambda row: {
                "id": row["id"] % 10000,  # 10K distinct keys
                "value": row["id"],
                "payload": f"data_{row['id'] % 100000:06d}",  # ~30 byte string
                "extra": row["id"] * 2,
            }
        )

        # Get size estimate
        size_bytes = ds.size_bytes()
        if size_bytes:
            print(f"Estimated data size: {size_bytes / (1024**3):.2f} GB")

        # Run shuffle with timing
        print("\nRunning shuffle (with memory estimation)...")
        start_time = time.time()

        result = ds.repartition(
            num_blocks=NUM_PARTITIONS,
            shuffle=True,
            keys=["id"],
        ).materialize()

        total_time = time.time() - start_time

        # Verify
        result_count = result.count()
        assert (
            result_count == NUM_ROWS
        ), f"Row count mismatch: {result_count} vs {NUM_ROWS}"

        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"  Execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Throughput: {(size_bytes or 0) / (1024**3) / total_time:.2f} GB/s")
        print(f"  Row count verified: {result_count:,}")
        print(f"{'='*60}")

        # If we get here, estimation was accurate (strict mode didn't fail)
        print("\nMemory estimation validation PASSED")
        print("  - Partition size estimates matched actual sizes")
        print("  - Heap memory estimates were within threshold")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FAILED: {e}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        ray.shutdown()
        return 1

    ray.shutdown()
    return 0


if __name__ == "__main__":
    exit(main())
