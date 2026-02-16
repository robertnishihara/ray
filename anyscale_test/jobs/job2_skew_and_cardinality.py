#!/usr/bin/env python
"""Job 2: Skew tests and Cardinality tests.

Cluster: 8 nodes, m5.4xlarge (16 vCPU, 64GB each)
Expected runtime: ~30 minutes
"""

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

import ray


@dataclass
class TestResult:
    test_name: str
    data_size_gb: float
    num_rows: int
    num_partitions: int
    distinct_keys: int
    skew_description: str
    estimation_time_sec: float
    execution_time_sec: float
    max_partition_error_pct: float
    avg_partition_error_pct: float
    status: str
    error_message: Optional[str] = None


def run_shuffle_test(
    test_name: str,
    data_items: List[dict],
    num_partitions: int,
    key_columns: List[str],
    skew_description: str = "",
) -> TestResult:
    """Run a shuffle test with pre-generated data."""
    from ray.data.context import DataContext

    ctx = DataContext.get_current()
    ctx.strict_shuffle_estimation = True

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Rows: {len(data_items):,}, Partitions: {num_partitions}")
    print(f"Skew: {skew_description}")
    print(f"{'='*60}")

    try:
        # Create dataset
        ds = ray.data.from_items(data_items)

        # Get distinct key count
        distinct_keys = len(
            {tuple(item[k] for k in key_columns) for item in data_items}
        )

        # Get size estimate
        size_bytes = ds.size_bytes()
        data_size_gb = size_bytes / (1024**3) if size_bytes else 0

        # Run repartition with timing
        start_time = time.time()
        _result = ds.repartition(
            num_blocks=num_partitions,
            shuffle=True,
            keys=key_columns,
        ).materialize()
        total_time = time.time() - start_time

        # Verify counts
        result_count = _result.count()
        assert result_count == len(data_items), "Row count mismatch"

        print(f"SUCCESS: {test_name}")
        print(f"  Data size: {data_size_gb:.2f} GB")
        print(f"  Distinct keys: {distinct_keys:,}")
        print(f"  Execution time: {total_time:.1f}s")

        return TestResult(
            test_name=test_name,
            data_size_gb=data_size_gb,
            num_rows=len(data_items),
            num_partitions=num_partitions,
            distinct_keys=distinct_keys,
            skew_description=skew_description,
            estimation_time_sec=0,
            execution_time_sec=total_time,
            max_partition_error_pct=0,
            avg_partition_error_pct=0,
            status="PASSED",
        )

    except Exception as e:
        print(f"FAILED: {test_name}")
        print(f"  Error: {e}")
        return TestResult(
            test_name=test_name,
            data_size_gb=0,
            num_rows=len(data_items),
            num_partitions=num_partitions,
            distinct_keys=0,
            skew_description=skew_description,
            estimation_time_sec=0,
            execution_time_sec=0,
            max_partition_error_pct=100,
            avg_partition_error_pct=100,
            status="FAILED",
            error_message=str(e),
        )


def run_skew_tests() -> List[TestResult]:
    """Run skew distribution tests."""
    results = []

    # Zipf distribution (80/20 rule)
    print("\nGenerating Zipf distribution data...")
    zipf_data = []
    np.random.seed(42)
    # Generate keys following Zipf distribution
    num_keys = 1000
    for i in range(5_000_000):
        # Zipf: P(k) ~ 1/k^s, using s=1.5
        key = int(np.random.zipf(1.5)) % num_keys
        zipf_data.append({"id": key, "value": i, "payload": f"data_{i:08d}"})

    results.append(
        run_shuffle_test(
            test_name="skew_zipf_distribution",
            data_items=zipf_data,
            num_partitions=100,
            key_columns=["id"],
            skew_description="Zipf distribution (s=1.5), 1000 keys",
        )
    )

    # Hot key (single key has 50%+ of data)
    print("\nGenerating hot key data...")
    hot_key_data = []
    for i in range(5_000_000):
        if i < 2_500_000:
            key = 0  # 50% to key 0
        else:
            key = (i % 999) + 1  # Rest distributed across 999 keys
        hot_key_data.append({"id": key, "value": i, "payload": f"data_{i:08d}"})

    results.append(
        run_shuffle_test(
            test_name="skew_hot_key_50pct",
            data_items=hot_key_data,
            num_partitions=100,
            key_columns=["id"],
            skew_description="50% of rows have key=0",
        )
    )

    # Extreme hot key (90% to one key)
    print("\nGenerating extreme hot key data...")
    extreme_hot_data = []
    for i in range(5_000_000):
        if i < 4_500_000:
            key = 0  # 90% to key 0
        else:
            key = (i % 99) + 1
        extreme_hot_data.append({"id": key, "value": i, "payload": f"data_{i:08d}"})

    results.append(
        run_shuffle_test(
            test_name="skew_hot_key_90pct",
            data_items=extreme_hot_data,
            num_partitions=100,
            key_columns=["id"],
            skew_description="90% of rows have key=0",
        )
    )

    # Long tail (many keys with 1 row, few with millions)
    print("\nGenerating long tail data...")
    long_tail_data = []
    # 10 keys get 400K rows each (4M total)
    for key in range(10):
        for i in range(400_000):
            long_tail_data.append(
                {
                    "id": key,
                    "value": key * 400_000 + i,
                    "payload": f"data_{key}_{i:06d}",
                }
            )
    # 100K keys get 10 rows each (1M total)
    for key in range(10, 100_010):
        for i in range(10):
            long_tail_data.append(
                {
                    "id": key,
                    "value": 4_000_000 + (key - 10) * 10 + i,
                    "payload": f"data_{key}_{i}",
                }
            )

    random.shuffle(long_tail_data)  # Shuffle to simulate realistic arrival

    results.append(
        run_shuffle_test(
            test_name="skew_long_tail",
            data_items=long_tail_data,
            num_partitions=200,
            key_columns=["id"],
            skew_description="10 keys with 400K rows, 100K keys with 10 rows",
        )
    )

    # Time-based skew (simulates event logs)
    print("\nGenerating time-based skew data...")
    time_skew_data = []
    # Recent "hour" has 80% of data
    for i in range(4_000_000):
        hour = 23  # Most recent hour
        time_skew_data.append(
            {
                "hour": hour,
                "user_id": i % 10000,
                "event": f"event_{i:08d}",
            }
        )
    # Previous 23 hours have 20% of data
    for i in range(1_000_000):
        hour = i % 23  # Hours 0-22
        time_skew_data.append(
            {
                "hour": hour,
                "user_id": i % 10000,
                "event": f"event_{4_000_000 + i:08d}",
            }
        )

    random.shuffle(time_skew_data)

    results.append(
        run_shuffle_test(
            test_name="skew_time_based",
            data_items=time_skew_data,
            num_partitions=100,
            key_columns=["hour"],
            skew_description="80% in hour 23, 20% in hours 0-22",
        )
    )

    return results


def run_cardinality_tests() -> List[TestResult]:
    """Run key cardinality tests."""
    results = []
    from ray.data.context import DataContext

    ctx = DataContext.get_current()
    ctx.strict_shuffle_estimation = True

    # Low cardinality (100 keys, 1000 partitions -> many empty)
    print("\nRunning low cardinality test...")
    results.append(
        run_shuffle_test(
            test_name="cardinality_low_100_keys",
            data_items=[
                {"id": i % 100, "value": i, "data": f"row_{i:08d}"}
                for i in range(5_000_000)
            ],
            num_partitions=1000,
            key_columns=["id"],
            skew_description="100 distinct keys -> ~900 empty partitions",
        )
    )

    # Medium cardinality (10K keys)
    print("\nRunning medium cardinality test...")
    ds = ray.data.range(10_000_000).map(
        lambda row: {
            "id": row["id"] % 10000,
            "value": row["id"],
            "data": f"row_{row['id']:08d}",
        }
    )

    start_time = time.time()
    _result = ds.repartition(
        num_blocks=1000,
        shuffle=True,
        keys=["id"],
    ).materialize()
    total_time = time.time() - start_time

    results.append(
        TestResult(
            test_name="cardinality_medium_10k_keys",
            data_size_gb=ds.size_bytes() / (1024**3) if ds.size_bytes() else 0,
            num_rows=10_000_000,
            num_partitions=1000,
            distinct_keys=10000,
            skew_description="10K distinct keys, 1000 partitions",
            estimation_time_sec=0,
            execution_time_sec=total_time,
            max_partition_error_pct=0,
            avg_partition_error_pct=0,
            status="PASSED",
        )
    )

    # High cardinality (1M keys)
    print("\nRunning high cardinality test...")
    ds = ray.data.range(20_000_000).map(
        lambda row: {
            "id": row["id"] % 1_000_000,
            "value": row["id"],
            "data": f"row_{row['id']:08d}",
        }
    )

    start_time = time.time()
    _result = ds.repartition(
        num_blocks=1000,
        shuffle=True,
        keys=["id"],
    ).materialize()
    total_time = time.time() - start_time

    results.append(
        TestResult(
            test_name="cardinality_high_1m_keys",
            data_size_gb=ds.size_bytes() / (1024**3) if ds.size_bytes() else 0,
            num_rows=20_000_000,
            num_partitions=1000,
            distinct_keys=1_000_000,
            skew_description="1M distinct keys >> 1000 partitions",
            estimation_time_sec=0,
            execution_time_sec=total_time,
            max_partition_error_pct=0,
            avg_partition_error_pct=0,
            status="PASSED",
        )
    )

    # Very high cardinality (unique keys)
    print("\nRunning unique keys test...")
    ds = ray.data.range(10_000_000).map(
        lambda row: {
            "id": row["id"],  # Every key is unique
            "value": row["id"] * 2,
            "data": f"row_{row['id']:08d}",
        }
    )

    start_time = time.time()
    _result = ds.repartition(
        num_blocks=500,
        shuffle=True,
        keys=["id"],
    ).materialize()
    total_time = time.time() - start_time

    results.append(
        TestResult(
            test_name="cardinality_unique_keys",
            data_size_gb=ds.size_bytes() / (1024**3) if ds.size_bytes() else 0,
            num_rows=10_000_000,
            num_partitions=500,
            distinct_keys=10_000_000,
            skew_description="Every row has unique key",
            estimation_time_sec=0,
            execution_time_sec=total_time,
            max_partition_error_pct=0,
            avg_partition_error_pct=0,
            status="PASSED",
        )
    )

    # Compound keys (2 columns)
    print("\nRunning compound keys test...")
    ds = ray.data.range(10_000_000).map(
        lambda row: {
            "key1": row["id"] % 100,
            "key2": row["id"] % 1000,
            "value": row["id"],
        }
    )

    start_time = time.time()
    _result = ds.repartition(
        num_blocks=500,
        shuffle=True,
        keys=["key1", "key2"],
    ).materialize()
    total_time = time.time() - start_time

    results.append(
        TestResult(
            test_name="cardinality_compound_keys",
            data_size_gb=ds.size_bytes() / (1024**3) if ds.size_bytes() else 0,
            num_rows=10_000_000,
            num_partitions=500,
            distinct_keys=100 * 1000,  # Approximate
            skew_description="Compound key (key1, key2)",
            estimation_time_sec=0,
            execution_time_sec=total_time,
            max_partition_error_pct=0,
            avg_partition_error_pct=0,
            status="PASSED",
        )
    )

    return results


def main():
    print("Initializing Ray...")
    ray.init()

    print(f"Ray version: {ray.__version__}")
    print(f"Cluster resources: {ray.cluster_resources()}")

    all_results = []

    try:
        # Run skew tests
        print("\n" + "=" * 80)
        print("SKEW TESTS")
        print("=" * 80)
        all_results.extend(run_skew_tests())

        # Run cardinality tests
        print("\n" + "=" * 80)
        print("CARDINALITY TESTS")
        print("=" * 80)
        all_results.extend(run_cardinality_tests())

    finally:
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in all_results if r.status == "PASSED")
        failed = sum(1 for r in all_results if r.status == "FAILED")

        print(f"\nTotal: {len(all_results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for r in all_results:
                if r.status == "FAILED":
                    print(f"  - {r.test_name}: {r.error_message}")

        # Save results to JSON
        results_dict = [asdict(r) for r in all_results]
        with open("/tmp/job2_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        print("\nResults saved to /tmp/job2_results.json")

        ray.shutdown()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
