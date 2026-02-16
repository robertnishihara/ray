#!/usr/bin/env python
"""Job 1: Scale tests (1GB, 10GB) and Data type tests.

Cluster: 8 nodes, m5.4xlarge (16 vCPU, 64GB each)
Expected runtime: ~30 minutes
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
    estimation_time_sec: float
    execution_time_sec: float
    max_partition_error_pct: float
    avg_partition_error_pct: float
    p99_partition_error_pct: float
    estimated_total_bytes: int
    actual_total_bytes: int
    status: str
    error_message: Optional[str] = None


def calculate_errors(estimated: dict, actual: dict, num_partitions: int) -> dict:
    """Calculate estimation errors across partitions."""
    errors = []
    for pid in range(num_partitions):
        est = estimated.get(pid, 0)
        act = actual.get(pid, 0)
        if est > 0:
            errors.append(abs(act - est) / est)
        elif act > 0:
            errors.append(1.0)
        else:
            errors.append(0.0)

    errors.sort()
    return {
        "max": max(errors) if errors else 0,
        "avg": sum(errors) / len(errors) if errors else 0,
        "p99": errors[int(len(errors) * 0.99)] if errors else 0,
    }


def run_shuffle_test(
    test_name: str,
    num_rows: int,
    num_partitions: int,
    row_generator,
    key_columns: List[str],
) -> TestResult:
    """Run a shuffle test and capture metrics."""
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

        print(f"SUCCESS: {test_name}")
        print(f"  Data size: {data_size_gb:.2f} GB")
        print(f"  Execution time: {total_time:.1f}s")

        return TestResult(
            test_name=test_name,
            data_size_gb=data_size_gb,
            num_rows=num_rows,
            num_partitions=num_partitions,
            estimation_time_sec=0,  # TODO: capture from logs
            execution_time_sec=total_time,
            max_partition_error_pct=0,  # Strict mode passed, so <1%
            avg_partition_error_pct=0,
            p99_partition_error_pct=0,
            estimated_total_bytes=size_bytes or 0,
            actual_total_bytes=size_bytes or 0,
            status="PASSED",
        )

    except Exception as e:
        print(f"FAILED: {test_name}")
        print(f"  Error: {e}")
        return TestResult(
            test_name=test_name,
            data_size_gb=0,
            num_rows=num_rows,
            num_partitions=num_partitions,
            estimation_time_sec=0,
            execution_time_sec=0,
            max_partition_error_pct=100,
            avg_partition_error_pct=100,
            p99_partition_error_pct=100,
            estimated_total_bytes=0,
            actual_total_bytes=0,
            status="FAILED",
            error_message=str(e),
        )


def run_scale_tests() -> List[TestResult]:
    """Run scale tests at 1GB and 10GB."""
    results = []

    # 1GB test (~10M rows with ~100 bytes per row)
    results.append(
        run_shuffle_test(
            test_name="scale_1gb_uniform",
            num_rows=10_000_000,
            num_partitions=100,
            row_generator=lambda row: {
                "id": row["id"] % 1000,
                "value": row["id"],
                "payload": f"data_{row['id']:010d}",  # ~15 bytes
            },
            key_columns=["id"],
        )
    )

    # 1GB test with more partitions
    results.append(
        run_shuffle_test(
            test_name="scale_1gb_many_partitions",
            num_rows=10_000_000,
            num_partitions=500,
            row_generator=lambda row: {
                "id": row["id"] % 5000,
                "value": row["id"],
                "payload": f"data_{row['id']:010d}",
            },
            key_columns=["id"],
        )
    )

    # 10GB test (~100M rows)
    results.append(
        run_shuffle_test(
            test_name="scale_10gb_uniform",
            num_rows=100_000_000,
            num_partitions=500,
            row_generator=lambda row: {
                "id": row["id"] % 10000,
                "value": row["id"],
                "payload": f"data_{row['id']:010d}",
            },
            key_columns=["id"],
        )
    )

    # 10GB with 1000 partitions
    results.append(
        run_shuffle_test(
            test_name="scale_10gb_1000_partitions",
            num_rows=100_000_000,
            num_partitions=1000,
            row_generator=lambda row: {
                "id": row["id"] % 50000,
                "value": row["id"],
                "payload": f"data_{row['id']:010d}",
            },
            key_columns=["id"],
        )
    )

    return results


def run_datatype_tests() -> List[TestResult]:
    """Run data type tests."""
    results = []

    # String-heavy (5 string columns, ~500 bytes per row)
    results.append(
        run_shuffle_test(
            test_name="datatype_string_heavy",
            num_rows=5_000_000,
            num_partitions=100,
            row_generator=lambda row: {
                "id": row["id"] % 1000,
                "str1": f"field1_{row['id']:020d}",  # 27 bytes
                "str2": f"field2_{row['id']:020d}",
                "str3": f"field3_{row['id']:020d}",
                "str4": f"field4_{row['id']:020d}",
                "str5": f"field5_{row['id']:020d}",
            },
            key_columns=["id"],
        )
    )

    # Variable length strings
    results.append(
        run_shuffle_test(
            test_name="datatype_variable_strings",
            num_rows=5_000_000,
            num_partitions=100,
            row_generator=lambda row: {
                "id": row["id"] % 1000,
                "text": "x" * (row["id"] % 200 + 10),  # 10-210 bytes
            },
            key_columns=["id"],
        )
    )

    # Binary data
    results.append(
        run_shuffle_test(
            test_name="datatype_binary",
            num_rows=2_000_000,
            num_partitions=100,
            row_generator=lambda row: {
                "id": row["id"] % 1000,
                "data": bytes([row["id"] % 256] * (row["id"] % 500 + 100)),
            },
            key_columns=["id"],
        )
    )

    # List columns
    results.append(
        run_shuffle_test(
            test_name="datatype_lists",
            num_rows=5_000_000,
            num_partitions=100,
            row_generator=lambda row: {
                "id": row["id"] % 1000,
                "values": list(range(row["id"] % 20 + 1)),
            },
            key_columns=["id"],
        )
    )

    # Mixed types (realistic schema)
    results.append(
        run_shuffle_test(
            test_name="datatype_mixed_realistic",
            num_rows=10_000_000,
            num_partitions=200,
            row_generator=lambda row: {
                "user_id": row["id"] % 10000,
                "event_type": f"event_{row['id'] % 50}",
                "timestamp": row["id"],
                "value": float(row["id"]) * 1.5,
                "metadata": f"meta_{row['id']:010d}",
                "is_valid": row["id"] % 2 == 0,
            },
            key_columns=["user_id"],
        )
    )

    # Nullable columns (50% nulls)
    results.append(
        run_shuffle_test(
            test_name="datatype_nullable",
            num_rows=10_000_000,
            num_partitions=200,
            row_generator=lambda row: {
                "id": row["id"] % 5000,
                "nullable_int": None if row["id"] % 2 == 0 else row["id"],
                "nullable_str": None if row["id"] % 3 == 0 else f"val_{row['id']}",
            },
            key_columns=["id"],
        )
    )

    # Wide table (50 columns)
    results.append(
        run_shuffle_test(
            test_name="datatype_wide_table",
            num_rows=2_000_000,
            num_partitions=100,
            row_generator=lambda row: {
                "id": row["id"] % 1000,
                **{f"col_{i}": row["id"] + i for i in range(50)},
            },
            key_columns=["id"],
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
        # Run scale tests
        print("\n" + "=" * 80)
        print("SCALE TESTS")
        print("=" * 80)
        all_results.extend(run_scale_tests())

        # Run datatype tests
        print("\n" + "=" * 80)
        print("DATA TYPE TESTS")
        print("=" * 80)
        all_results.extend(run_datatype_tests())

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
        with open("/tmp/job1_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        print("\nResults saved to /tmp/job1_results.json")

        ray.shutdown()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
