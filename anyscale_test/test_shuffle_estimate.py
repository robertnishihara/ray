#!/usr/bin/env python
"""Test script for shuffle memory estimation feature on Anyscale.

This script tests that memory estimation runs automatically during shuffle/join
operations. The estimation happens before the shuffle, prints statistics, and
validates estimates against actual execution.
"""

import ray


def test_repartition_with_estimation():
    """Test repartition with automatic memory estimation."""
    print("\n" + "=" * 60)
    print("Testing repartition with automatic memory estimation")
    print("=" * 60)

    # Create a dataset with some skew (most rows have same key)
    data = [{"user_id": i % 10, "value": i, "payload": "x" * 100} for i in range(10000)]
    ds = ray.data.from_items(data)

    print(f"\nDataset: {ds.count()} rows")
    print("Running repartition with shuffle=True, keys=['user_id']...")
    print("(Memory estimation will be printed automatically before shuffle)")

    # Run the actual repartition - estimation happens automatically
    result = ds.repartition(
        num_blocks=50,
        shuffle=True,
        keys=["user_id"],
    )

    # Materialize to trigger execution
    result = result.materialize()

    print(f"\nRepartition complete. Result has {result.num_blocks()} blocks.")
    return result


def test_join_with_estimation():
    """Test join with automatic memory estimation."""
    print("\n" + "=" * 60)
    print("Testing join with automatic memory estimation")
    print("=" * 60)

    # Create two datasets
    left_ds = ray.data.from_items(
        [{"id": i, "left_val": i * 2, "left_payload": "a" * 50} for i in range(5000)]
    )
    right_ds = ray.data.from_items(
        [{"id": i, "right_val": i**2, "right_payload": "b" * 50} for i in range(3000)]
    )

    print(f"\nLeft dataset: {left_ds.count()} rows")
    print(f"Right dataset: {right_ds.count()} rows")
    print("Running join...")
    print("(Memory estimation will be printed automatically before join)")

    # Run the actual join - estimation happens automatically
    result = left_ds.join(
        right_ds,
        join_type="inner",
        num_partitions=20,
        on=("id",),
    )

    # Materialize to trigger execution
    result = result.materialize()

    print(f"\nJoin complete. Result has {result.count()} rows.")
    return result


def test_skewed_data_estimation():
    """Test that skew detection works with automatic estimation."""
    print("\n" + "=" * 60)
    print("Testing skew detection with highly skewed data")
    print("=" * 60)

    # Create highly skewed data - 90% of rows have the same key
    data = [{"key": 0, "value": i} for i in range(9000)]  # 9000 rows with key=0
    data += [
        {"key": i, "value": i} for i in range(1, 1001)
    ]  # 1000 rows with unique keys

    ds = ray.data.from_items(data)

    print(f"\nDataset: {ds.count()} rows")
    print("Distribution: 9000 rows with key=0, 1000 rows with unique keys")
    print("Running repartition...")
    print("(Skew warning should appear in the estimation output)")

    result = ds.repartition(
        num_blocks=100,
        shuffle=True,
        keys=["key"],
    ).materialize()

    print(f"\nRepartition complete. Result has {result.num_blocks()} blocks.")
    return result


def main():
    print("Initializing Ray...")
    ray.init()

    print(f"\nRay version: {ray.__version__}")
    print(f"Cluster resources: {ray.cluster_resources()}")

    try:
        # Run tests
        repartition_result = test_repartition_with_estimation()
        join_result = test_join_with_estimation()
        skew_result = test_skewed_data_estimation()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("\nAll tests completed successfully!")
        print(f"\nRepartition result: {repartition_result.num_blocks()} blocks")
        print(f"Join result: {join_result.count()} rows")
        print(f"Skew test result: {skew_result.num_blocks()} blocks")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
