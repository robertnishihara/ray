#!/usr/bin/env python
"""Job 3: Join tests.

Cluster: 16x m5.2xlarge (8 vCPU, 32GB each = 512GB total, ~150GB object store)
Expected runtime: ~45 minutes
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import ray


@dataclass
class TestResult:
    test_name: str
    left_size_gb: float
    right_size_gb: float
    left_rows: int
    right_rows: int
    result_rows: int
    num_partitions: int
    join_type: str
    skew_description: str
    execution_time_sec: float
    status: str
    error_message: Optional[str] = None


def run_join_test(
    test_name: str,
    left_ds,
    right_ds,
    num_partitions: int,
    join_type: str,
    left_key: str,
    right_key: str,
    skew_description: str = "",
) -> TestResult:
    """Run a join test and capture metrics."""
    from ray.data.context import DataContext

    ctx = DataContext.get_current()
    ctx.strict_shuffle_estimation = True

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Join type: {join_type}, Partitions: {num_partitions}")
    print(f"Skew: {skew_description}")
    print(f"{'='*60}")

    try:
        # Get sizes
        left_bytes = left_ds.size_bytes() or 0
        right_bytes = right_ds.size_bytes() or 0
        left_count = left_ds.count()
        right_count = right_ds.count()

        print(f"Left: {left_count:,} rows, {left_bytes / (1024**3):.2f} GB")
        print(f"Right: {right_count:,} rows, {right_bytes / (1024**3):.2f} GB")

        # Run join with timing
        start_time = time.time()
        if left_key == right_key:
            result = left_ds.join(
                right_ds,
                join_type=join_type,
                num_partitions=num_partitions,
                on=(left_key,),
            ).materialize()
        else:
            result = left_ds.join(
                right_ds,
                join_type=join_type,
                num_partitions=num_partitions,
                on=(left_key,),
                right_on=(right_key,),
            ).materialize()
        total_time = time.time() - start_time

        result_count = result.count()

        print(f"SUCCESS: {test_name}")
        print(f"  Result rows: {result_count:,}")
        print(f"  Execution time: {total_time:.1f}s")

        return TestResult(
            test_name=test_name,
            left_size_gb=left_bytes / (1024**3),
            right_size_gb=right_bytes / (1024**3),
            left_rows=left_count,
            right_rows=right_count,
            result_rows=result_count,
            num_partitions=num_partitions,
            join_type=join_type,
            skew_description=skew_description,
            execution_time_sec=total_time,
            status="PASSED",
        )

    except Exception as e:
        print(f"FAILED: {test_name}")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return TestResult(
            test_name=test_name,
            left_size_gb=0,
            right_size_gb=0,
            left_rows=0,
            right_rows=0,
            result_rows=0,
            num_partitions=num_partitions,
            join_type=join_type,
            skew_description=skew_description,
            execution_time_sec=0,
            status="FAILED",
            error_message=str(e),
        )


def run_join_tests() -> List[TestResult]:
    """Run all join tests."""
    results = []

    # Test 1: Balanced inner join (10GB x 10GB)
    print("\nCreating balanced join datasets...")
    left = ray.data.range(50_000_000).map(
        lambda row: {
            "id": row["id"],
            "left_val": row["id"] * 2,
            "left_data": f"left_{row['id']:08d}",
        }
    )
    right = ray.data.range(50_000_000).map(
        lambda row: {
            "id": row["id"],
            "right_val": row["id"] * 3,
            "right_data": f"right_{row['id']:08d}",
        }
    )

    results.append(
        run_join_test(
            test_name="join_balanced_inner",
            left_ds=left,
            right_ds=right,
            num_partitions=200,
            join_type="inner",
            left_key="id",
            right_key="id",
            skew_description="50M rows each, 1:1 join",
        )
    )

    # Test 2: Build-side heavy (1GB left, 10GB right)
    print("\nCreating build-side heavy datasets...")
    left_small = ray.data.range(5_000_000).map(
        lambda row: {
            "id": row["id"],
            "left_val": row["id"] * 2,
        }
    )
    right_large = ray.data.range(50_000_000).map(
        lambda row: {
            "id": row["id"] % 5_000_000,  # Match with left keys
            "right_val": row["id"] * 3,
            "right_data": f"right_{row['id']:08d}",
        }
    )

    results.append(
        run_join_test(
            test_name="join_build_side_heavy",
            left_ds=left_small,
            right_ds=right_large,
            num_partitions=200,
            join_type="inner",
            left_key="id",
            right_key="id",
            skew_description="5M left, 50M right (10:1 ratio)",
        )
    )

    # Test 3: Probe-side heavy (10GB left, 1GB right)
    print("\nCreating probe-side heavy datasets...")
    left_large = ray.data.range(50_000_000).map(
        lambda row: {
            "id": row["id"] % 5_000_000,
            "left_val": row["id"] * 2,
            "left_data": f"left_{row['id']:08d}",
        }
    )
    right_small = ray.data.range(5_000_000).map(
        lambda row: {
            "id": row["id"],
            "right_val": row["id"] * 3,
        }
    )

    results.append(
        run_join_test(
            test_name="join_probe_side_heavy",
            left_ds=left_large,
            right_ds=right_small,
            num_partitions=200,
            join_type="inner",
            left_key="id",
            right_key="id",
            skew_description="50M left, 5M right (10:1 ratio)",
        )
    )

    # Test 4: Skewed join (hot keys on both sides)
    print("\nCreating skewed join datasets...")
    # 50% of rows have key=0 on both sides
    left_skewed_items = []
    for i in range(10_000_000):
        if i < 5_000_000:
            key = 0
        else:
            key = i % 100000 + 1
        left_skewed_items.append(
            {
                "id": key,
                "left_val": i,
            }
        )
    left_skewed = ray.data.from_items(left_skewed_items)

    right_skewed_items = []
    for i in range(10_000_000):
        if i < 5_000_000:
            key = 0
        else:
            key = i % 100000 + 1
        right_skewed_items.append(
            {
                "id": key,
                "right_val": i,
            }
        )
    right_skewed = ray.data.from_items(right_skewed_items)

    results.append(
        run_join_test(
            test_name="join_skewed_hot_keys",
            left_ds=left_skewed,
            right_ds=right_skewed,
            num_partitions=100,
            join_type="inner",
            left_key="id",
            right_key="id",
            skew_description="50% of rows have key=0 on both sides",
        )
    )

    # Test 5: Left outer join with mismatched keys
    print("\nCreating outer join datasets...")
    left_outer = ray.data.range(20_000_000).map(
        lambda row: {
            "id": row["id"],
            "left_val": row["id"] * 2,
        }
    )
    # Right has only 50% of the keys
    right_outer = ray.data.range(10_000_000).map(
        lambda row: {
            "id": row["id"] * 2,  # Only even keys
            "right_val": row["id"] * 3,
        }
    )

    results.append(
        run_join_test(
            test_name="join_left_outer_mismatch",
            left_ds=left_outer,
            right_ds=right_outer,
            num_partitions=100,
            join_type="left",
            left_key="id",
            right_key="id",
            skew_description="Left 20M rows, Right 10M rows (50% key overlap)",
        )
    )

    # Test 6: Different key names
    print("\nCreating different key names datasets...")
    left_diff_key = ray.data.range(10_000_000).map(
        lambda row: {
            "user_id": row["id"],
            "left_val": row["id"] * 2,
        }
    )
    right_diff_key = ray.data.range(10_000_000).map(
        lambda row: {
            "customer_id": row["id"],
            "right_val": row["id"] * 3,
        }
    )

    results.append(
        run_join_test(
            test_name="join_different_key_names",
            left_ds=left_diff_key,
            right_ds=right_diff_key,
            num_partitions=100,
            join_type="inner",
            left_key="user_id",
            right_key="customer_id",
            skew_description="Different key column names",
        )
    )

    # Test 7: Many-to-many join (cartesian-like per key)
    print("\nCreating many-to-many join datasets...")
    # Each key appears 100 times on left and 50 times on right
    left_many = ray.data.range(10_000_000).map(
        lambda row: {
            "id": row["id"] // 100,  # 100K distinct keys, 100 rows each
            "left_seq": row["id"] % 100,
            "left_val": row["id"],
        }
    )
    right_many = ray.data.range(5_000_000).map(
        lambda row: {
            "id": row["id"] // 50,  # 100K distinct keys, 50 rows each
            "right_seq": row["id"] % 50,
            "right_val": row["id"],
        }
    )

    results.append(
        run_join_test(
            test_name="join_many_to_many",
            left_ds=left_many,
            right_ds=right_many,
            num_partitions=200,
            join_type="inner",
            left_key="id",
            right_key="id",
            skew_description="100 rows/key left x 50 rows/key right = 5000 output per key",
        )
    )

    # Test 8: String keys
    print("\nCreating string key datasets...")
    left_str_key = ray.data.range(10_000_000).map(
        lambda row: {
            "id": f"user_{row['id'] % 100000:06d}",
            "left_val": row["id"],
        }
    )
    right_str_key = ray.data.range(10_000_000).map(
        lambda row: {
            "id": f"user_{row['id'] % 100000:06d}",
            "right_val": row["id"],
        }
    )

    results.append(
        run_join_test(
            test_name="join_string_keys",
            left_ds=left_str_key,
            right_ds=right_str_key,
            num_partitions=200,
            join_type="inner",
            left_key="id",
            right_key="id",
            skew_description="String keys (user_XXXXXX)",
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
        # Run join tests
        print("\n" + "=" * 80)
        print("JOIN TESTS")
        print("=" * 80)
        all_results.extend(run_join_tests())

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
        with open("/tmp/job3_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        print("\nResults saved to /tmp/job3_results.json")

        ray.shutdown()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
