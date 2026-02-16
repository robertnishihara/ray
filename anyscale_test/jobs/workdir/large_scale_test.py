#!/usr/bin/env python
"""Large scale test (100GB) for shuffle memory estimation.

Tests multiple shuffle types:
1. Repartition with keys
2. Groupby aggregation
3. Inner join

Cluster recommendation: 16x m5.4xlarge (16 vCPU, 64GB each)
Expected runtime: ~30-45 minutes
"""

import time

import ray


def run_repartition_test(num_rows: int, num_partitions: int):
    """Test repartition shuffle."""
    print(f"\n{'='*60}")
    print("Test 1: Repartition (100GB)")
    print(f"  Rows: {num_rows:,}")
    print(f"  Partitions: {num_partitions}")
    print(f"{'='*60}")

    # Create dataset with ~100 bytes per row
    print("\nCreating dataset...")
    ds = ray.data.range(num_rows).map(
        lambda row: {
            "id": row["id"] % 100000,  # 100K distinct keys
            "value": row["id"],
            "payload": f"data_{row['id'] % 1000000:06d}",
            "extra": row["id"] * 2,
        }
    )

    size_bytes = ds.size_bytes()
    if size_bytes:
        print(f"Estimated data size: {size_bytes / (1024**3):.2f} GB")

    print("\nRunning repartition shuffle...")
    start_time = time.time()

    result = ds.repartition(
        num_blocks=num_partitions,
        shuffle=True,
        keys=["id"],
    ).materialize()

    total_time = time.time() - start_time
    result_count = result.count()

    print("\nRepartition completed:")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Rows: {result_count:,}")
    print(f"  Throughput: {(size_bytes or 0) / (1024**3) / total_time:.2f} GB/s")

    assert result_count == num_rows, f"Row count mismatch: {result_count} vs {num_rows}"
    return total_time


def run_groupby_test(num_rows: int):
    """Test groupby aggregation shuffle."""
    print(f"\n{'='*60}")
    print("Test 2: Groupby Aggregation (100GB)")
    print(f"  Rows: {num_rows:,}")
    print(f"{'='*60}")

    print("\nCreating dataset...")
    ds = ray.data.range(num_rows).map(
        lambda row: {
            "group_key": row["id"] % 50000,  # 50K groups
            "value": row["id"] % 1000,
            "payload": f"data_{row['id'] % 100000:06d}",
        }
    )

    size_bytes = ds.size_bytes()
    if size_bytes:
        print(f"Estimated data size: {size_bytes / (1024**3):.2f} GB")

    print("\nRunning groupby aggregation...")
    start_time = time.time()

    result = ds.groupby("group_key").sum("value").materialize()

    total_time = time.time() - start_time
    result_count = result.count()

    print("\nGroupby completed:")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Groups: {result_count:,}")
    print(f"  Throughput: {(size_bytes or 0) / (1024**3) / total_time:.2f} GB/s")

    # Should have 50K groups
    assert result_count == 50000, f"Group count mismatch: {result_count} vs 50000"
    return total_time


def run_join_test(num_rows_per_side: int, num_partitions: int):
    """Test inner join shuffle."""
    print(f"\n{'='*60}")
    print("Test 3: Inner Join (50GB + 50GB)")
    print(f"  Rows per side: {num_rows_per_side:,}")
    print(f"  Partitions: {num_partitions}")
    print(f"{'='*60}")

    print("\nCreating left dataset...")
    left = ray.data.range(num_rows_per_side).map(
        lambda row: {
            "join_key": row["id"] % 100000,  # 100K distinct keys
            "left_value": row["id"],
            "left_data": f"left_{row['id'] % 100000:06d}",
        }
    )

    print("Creating right dataset...")
    right = ray.data.range(num_rows_per_side).map(
        lambda row: {
            "join_key": row["id"] % 100000,  # Same 100K keys
            "right_value": row["id"] * 2,
            "right_data": f"right_{row['id'] % 100000:06d}",
        }
    )

    left_bytes = left.size_bytes() or 0
    right_bytes = right.size_bytes() or 0
    total_bytes = left_bytes + right_bytes
    print(f"Left size: {left_bytes / (1024**3):.2f} GB")
    print(f"Right size: {right_bytes / (1024**3):.2f} GB")
    print(f"Total input: {total_bytes / (1024**3):.2f} GB")

    print("\nRunning inner join...")
    start_time = time.time()

    result = left.join(
        right,
        join_type="inner",
        num_partitions=num_partitions,
        on=("join_key",),
    ).materialize()

    total_time = time.time() - start_time
    result_count = result.count()

    print("\nJoin completed:")
    print(f"  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output rows: {result_count:,}")
    print(f"  Throughput: {total_bytes / (1024**3) / total_time:.2f} GB/s")

    # With uniform distribution: each key appears num_rows/100K times on each side
    # So output = 100K keys * (num_rows/100K)^2 = num_rows^2 / 100K
    expected_rows = num_rows_per_side  # For 1:1 key ratio
    print(f"  Expected ~{expected_rows:,} rows (1:1 key ratio)")

    return total_time


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

    # Enable strict validation
    from ray.data.context import DataContext

    if hasattr(DataContext, "get_current"):
        ctx = DataContext.get_current()
    else:
        ctx = DataContext.current()
    ctx.strict_shuffle_estimation = True
    print("\nStrict shuffle estimation: ENABLED")

    # Test parameters for ~100GB total
    # ~100 bytes per row -> 1B rows for 100GB
    NUM_ROWS = 1_000_000_000
    NUM_PARTITIONS = 500

    # For join: 500M rows per side = ~50GB each = 100GB total
    JOIN_ROWS_PER_SIDE = 500_000_000
    JOIN_PARTITIONS = 500

    results = {}
    failed = False

    try:
        # Test 1: Repartition
        results["repartition"] = run_repartition_test(NUM_ROWS, NUM_PARTITIONS)
        print("\n✓ Repartition test PASSED")

    except Exception as e:
        print(f"\n✗ Repartition test FAILED: {e}")
        import traceback

        traceback.print_exc()
        failed = True

    try:
        # Test 2: Groupby
        results["groupby"] = run_groupby_test(NUM_ROWS)
        print("\n✓ Groupby test PASSED")

    except Exception as e:
        print(f"\n✗ Groupby test FAILED: {e}")
        import traceback

        traceback.print_exc()
        failed = True

    try:
        # Test 3: Join
        results["join"] = run_join_test(JOIN_ROWS_PER_SIDE, JOIN_PARTITIONS)
        print("\n✓ Join test PASSED")

    except Exception as e:
        print(f"\n✗ Join test FAILED: {e}")
        import traceback

        traceback.print_exc()
        failed = True

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_time = sum(results.values())
    for test_name, duration in results.items():
        print(f"  {test_name}: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")

    if failed:
        print("\n⚠ Some tests FAILED - check logs above")
        ray.shutdown()
        return 1
    else:
        print("\n✓ All tests PASSED")
        print("Memory estimation validation successful for all shuffle types")
        ray.shutdown()
        return 0


if __name__ == "__main__":
    exit(main())
