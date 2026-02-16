#!/usr/bin/env python
"""Job 4: Large scale tests (100GB).

Cluster: 16x m5.4xlarge (16 vCPU, 64GB each = 1TB total, ~300GB object store)
Expected runtime: ~60 minutes
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import ray


@dataclass
class TestResult:
    test_name: str
    data_size_gb: float
    num_rows: int
    num_partitions: int
    execution_time_sec: float
    throughput_gb_per_sec: float
    status: str
    error_message: Optional[str] = None


def run_large_scale_test(
    test_name: str,
    num_rows: int,
    num_partitions: int,
    row_generator,
    key_columns: List[str],
) -> TestResult:
    """Run a large scale shuffle test."""
    from ray.data.context import DataContext

    ctx = DataContext.get_current()
    ctx.strict_shuffle_estimation = True

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Rows: {num_rows:,}, Partitions: {num_partitions}")
    print(f"{'='*60}")

    try:
        # Create dataset
        ds = ray.data.range(num_rows).map(row_generator)

        # Get size estimate
        size_bytes = ds.size_bytes()
        data_size_gb = size_bytes / (1024**3) if size_bytes else 0
        print(f"Estimated data size: {data_size_gb:.2f} GB")

        # Run repartition with timing
        start_time = time.time()
        result = ds.repartition(
            num_blocks=num_partitions,
            shuffle=True,
            keys=key_columns,
        ).materialize()
        total_time = time.time() - start_time

        # Verify counts
        result_count = result.count()
        assert (
            result_count == num_rows
        ), f"Row count mismatch: {result_count} vs {num_rows}"

        throughput = data_size_gb / total_time if total_time > 0 else 0

        print(f"SUCCESS: {test_name}")
        print(f"  Data size: {data_size_gb:.2f} GB")
        print(f"  Execution time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.2f} GB/s")

        return TestResult(
            test_name=test_name,
            data_size_gb=data_size_gb,
            num_rows=num_rows,
            num_partitions=num_partitions,
            execution_time_sec=total_time,
            throughput_gb_per_sec=throughput,
            status="PASSED",
        )

    except Exception as e:
        print(f"FAILED: {test_name}")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return TestResult(
            test_name=test_name,
            data_size_gb=0,
            num_rows=num_rows,
            num_partitions=num_partitions,
            execution_time_sec=0,
            throughput_gb_per_sec=0,
            status="FAILED",
            error_message=str(e),
        )


def main():
    print("Initializing Ray...")
    ray.init()

    print(f"Ray version: {ray.__version__}")
    resources = ray.cluster_resources()
    print(f"Cluster resources: {resources}")
    print(f"Total memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
    print(f"Object store: {resources.get('object_store_memory', 0) / (1024**3):.1f} GB")

    results = []

    try:
        # Test 1: 100GB uniform distribution
        # ~100 bytes per row -> 1B rows for 100GB
        results.append(
            run_large_scale_test(
                test_name="large_100gb_uniform",
                num_rows=1_000_000_000,
                num_partitions=1000,
                row_generator=lambda row: {
                    "id": row["id"] % 100000,
                    "value": row["id"],
                    "payload": f"data_{row['id'] % 1000000:06d}",
                },
                key_columns=["id"],
            )
        )

        # Test 2: 100GB with high partition count
        results.append(
            run_large_scale_test(
                test_name="large_100gb_2000_partitions",
                num_rows=1_000_000_000,
                num_partitions=2000,
                row_generator=lambda row: {
                    "id": row["id"] % 500000,
                    "value": row["id"],
                    "payload": f"data_{row['id'] % 1000000:06d}",
                },
                key_columns=["id"],
            )
        )

        # Test 3: 100GB with skew
        # Use modulo to create skew - lower keys get more rows
        results.append(
            run_large_scale_test(
                test_name="large_100gb_skewed",
                num_rows=1_000_000_000,
                num_partitions=1000,
                row_generator=lambda row: {
                    # Skew: keys 0-99 get 10x more rows than keys 100-999
                    "id": row["id"] % 100
                    if row["id"] % 10 < 9
                    else 100 + row["id"] % 900,
                    "value": row["id"],
                    "payload": f"data_{row['id'] % 1000000:06d}",
                },
                key_columns=["id"],
            )
        )

        # Test 4: 100GB with strings
        results.append(
            run_large_scale_test(
                test_name="large_100gb_strings",
                num_rows=500_000_000,  # Larger rows due to strings
                num_partitions=1000,
                row_generator=lambda row: {
                    "id": row["id"] % 100000,
                    "name": f"user_{row['id']:012d}",
                    "email": f"user_{row['id']}@example.com",
                    "data": f"payload_{row['id'] % 10000:04d}",
                },
                key_columns=["id"],
            )
        )

        # Test 5: 100GB join
        print("\n" + "=" * 60)
        print("Running large scale join test")
        print("=" * 60)

        from ray.data.context import DataContext

        ctx = DataContext.get_current()
        ctx.strict_shuffle_estimation = True

        left = ray.data.range(500_000_000).map(
            lambda row: {
                "id": row["id"],
                "left_val": row["id"] * 2,
            }
        )
        right = ray.data.range(500_000_000).map(
            lambda row: {
                "id": row["id"],
                "right_val": row["id"] * 3,
            }
        )

        start_time = time.time()
        _result = left.join(
            right,
            join_type="inner",
            num_partitions=1000,
            on=("id",),
        ).materialize()
        total_time = time.time() - start_time

        left_bytes = left.size_bytes() or 0
        right_bytes = right.size_bytes() or 0
        total_gb = (left_bytes + right_bytes) / (1024**3)

        results.append(
            TestResult(
                test_name="large_100gb_join",
                data_size_gb=total_gb,
                num_rows=500_000_000,
                num_partitions=1000,
                execution_time_sec=total_time,
                throughput_gb_per_sec=total_gb / total_time if total_time > 0 else 0,
                status="PASSED",
            )
        )
        print(f"SUCCESS: large_100gb_join in {total_time:.1f}s")

    except Exception as e:
        print(f"Error in large scale tests: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in results if r.status == "PASSED")
        failed = sum(1 for r in results if r.status == "FAILED")

        print(f"\nTotal: {len(results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        print("\nPerformance summary:")
        for r in results:
            if r.status == "PASSED":
                print(
                    f"  {r.test_name}: {r.data_size_gb:.1f}GB in {r.execution_time_sec:.1f}s ({r.throughput_gb_per_sec:.2f} GB/s)"
                )

        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if r.status == "FAILED":
                    print(f"  - {r.test_name}: {r.error_message}")

        # Save results to JSON
        results_dict = [asdict(r) for r in results]
        with open("/tmp/job4_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        print("\nResults saved to /tmp/job4_results.json")

        ray.shutdown()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
