#!/usr/bin/env python
"""Job 6: Very large scale tests (500GB).

Cluster: 32x m5.8xlarge (32 vCPU, 128GB each = 4TB total, ~1.2TB object store)
Expected runtime: ~90 minutes

NOTE: This job requires significant cluster resources and should only be
run if the smaller scale tests pass successfully.
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


def run_very_large_test(
    test_name: str,
    num_rows: int,
    num_partitions: int,
    row_generator,
    key_columns: List[str],
) -> TestResult:
    """Run a very large scale shuffle test."""
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
        print(f"  Execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
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
    print(f"Total CPUs: {resources.get('CPU', 0)}")
    print(f"Total memory: {resources.get('memory', 0) / (1024**3):.1f} GB")
    print(f"Object store: {resources.get('object_store_memory', 0) / (1024**3):.1f} GB")

    # Verify cluster has enough resources
    object_store_gb = resources.get("object_store_memory", 0) / (1024**3)
    if object_store_gb < 500:
        print(
            f"\nWARNING: Object store ({object_store_gb:.0f} GB) may be insufficient for 500GB tests"
        )
        print("Recommended: >= 1TB object store")

    results = []

    try:
        # Test 1: 500GB uniform distribution
        # ~100 bytes per row -> 5B rows for 500GB
        results.append(
            run_very_large_test(
                test_name="very_large_500gb_uniform",
                num_rows=5_000_000_000,
                num_partitions=2000,
                row_generator=lambda row: {
                    "id": row["id"] % 500000,
                    "value": row["id"],
                    "payload": f"data_{row['id'] % 1000000:06d}",
                },
                key_columns=["id"],
            )
        )

        # Test 2: 500GB with high partition count
        results.append(
            run_very_large_test(
                test_name="very_large_500gb_5000_partitions",
                num_rows=5_000_000_000,
                num_partitions=5000,
                row_generator=lambda row: {
                    "id": row["id"] % 1000000,
                    "value": row["id"],
                    "payload": f"data_{row['id'] % 1000000:06d}",
                },
                key_columns=["id"],
            )
        )

        # Test 3: 500GB skewed
        results.append(
            run_very_large_test(
                test_name="very_large_500gb_skewed",
                num_rows=5_000_000_000,
                num_partitions=2000,
                row_generator=lambda row: {
                    # Zipf-like: lower keys get more rows
                    "id": row["id"] % 100
                    if row["id"] % 10 < 7
                    else 100 + row["id"] % 10000,
                    "value": row["id"],
                    "payload": f"data_{row['id'] % 1000000:06d}",
                },
                key_columns=["id"],
            )
        )

        # Test 4: 500GB join (250GB + 250GB)
        print("\n" + "=" * 60)
        print("Running very large scale join test")
        print("=" * 60)

        from ray.data.context import DataContext

        ctx = DataContext.get_current()
        ctx.strict_shuffle_estimation = True

        left = ray.data.range(2_500_000_000).map(
            lambda row: {
                "id": row["id"],
                "left_val": row["id"] * 2,
            }
        )
        right = ray.data.range(2_500_000_000).map(
            lambda row: {
                "id": row["id"],
                "right_val": row["id"] * 3,
            }
        )

        print("Left dataset: 2.5B rows")
        print("Right dataset: 2.5B rows")

        start_time = time.time()
        _result = left.join(
            right,
            join_type="inner",
            num_partitions=2000,
            on=("id",),
        ).materialize()
        total_time = time.time() - start_time

        left_bytes = left.size_bytes() or 0
        right_bytes = right.size_bytes() or 0
        total_gb = (left_bytes + right_bytes) / (1024**3)

        results.append(
            TestResult(
                test_name="very_large_500gb_join",
                data_size_gb=total_gb,
                num_rows=2_500_000_000,
                num_partitions=2000,
                execution_time_sec=total_time,
                throughput_gb_per_sec=total_gb / total_time if total_time > 0 else 0,
                status="PASSED",
            )
        )
        print(
            f"SUCCESS: very_large_500gb_join in {total_time:.1f}s ({total_time/60:.1f} min)"
        )

    except Exception as e:
        print(f"Error in very large scale tests: {e}")
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
                print(f"  {r.test_name}:")
                print(f"    Size: {r.data_size_gb:.1f} GB")
                print(
                    f"    Time: {r.execution_time_sec:.1f}s ({r.execution_time_sec/60:.1f} min)"
                )
                print(f"    Throughput: {r.throughput_gb_per_sec:.2f} GB/s")

        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if r.status == "FAILED":
                    print(f"  - {r.test_name}: {r.error_message}")

        # Save results to JSON
        results_dict = [asdict(r) for r in results]
        with open("/tmp/job6_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        print("\nResults saved to /tmp/job6_results.json")

        ray.shutdown()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
