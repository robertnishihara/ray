"""Tests for shuffle memory estimation accuracy.

These tests validate that the memory estimation phase produces accurate
predictions of partition sizes. They use strict_shuffle_estimation mode
which raises an exception if any partition size differs from actual by >1%.

The tests cover edge cases that stress the estimation accuracy:
- Different data types (integers, strings, binary, nested)
- Different partition distributions (uniform, skewed, empty partitions)
- Different key cardinalities (single key, low, high, unique)
- Different block structures (single block, many blocks)
"""

import numpy as np
import pytest

import ray
from ray.data.context import DataContext
from ray.tests.conftest import *  # noqa


@pytest.fixture
def strict_estimation_context():
    """Enable strict shuffle estimation for the duration of the test."""
    ctx = DataContext.get_current()
    original = ctx.strict_shuffle_estimation
    ctx.strict_shuffle_estimation = True
    yield ctx
    ctx.strict_shuffle_estimation = original


class TestIntegerData:
    """Test estimation accuracy with integer data (fixed-width columns)."""

    def test_uniform_distribution(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with uniformly distributed integer keys."""
        # Create data where keys are evenly distributed
        ds = ray.data.range(1000).map(
            lambda row: {"id": row["id"] % 10, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 1000
        assert result.num_blocks() == 10

    def test_single_key(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test where all rows have the same key (extreme skew)."""
        ds = ray.data.range(500).map(lambda row: {"id": 0, "value": row["id"]})

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        # All rows should end up in one partition
        assert result.count() == 500

    def test_unique_keys(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test where every row has a unique key."""
        ds = ray.data.range(500).map(
            lambda row: {"id": row["id"], "value": row["id"] * 2}
        )

        result = ds.repartition(
            num_blocks=50,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_multiple_integer_columns(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with multiple integer columns of different sizes."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 20,
                "int8_col": row["id"] % 128,
                "int64_col": row["id"] * 1000000,
                "float64_col": float(row["id"]) * 3.14159,
            }
        )

        result = ds.repartition(
            num_blocks=20,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_compound_key(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with compound keys (multiple key columns)."""
        ds = ray.data.range(500).map(
            lambda row: {
                "key1": row["id"] % 5,
                "key2": row["id"] % 10,
                "value": row["id"],
            }
        )

        result = ds.repartition(
            num_blocks=25,
            shuffle=True,
            keys=["key1", "key2"],
        ).materialize()

        assert result.count() == 500


class TestStringData:
    """Test estimation accuracy with string data (variable-width columns)."""

    def test_fixed_length_strings(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with fixed-length strings."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 20,
                "name": f"user_{row['id']:05d}",  # Fixed 10 chars
            }
        )

        result = ds.repartition(
            num_blocks=20,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_variable_length_strings(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with variable-length strings where length correlates with key."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                # String length varies by row
                "payload": "x" * (row["id"] % 100 + 1),
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_string_keys(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with string keys instead of integer keys."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": f"key_{row['id'] % 20}",
                "value": row["id"],
            }
        )

        result = ds.repartition(
            num_blocks=20,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_empty_strings(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with empty strings mixed with non-empty."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "text": "" if row["id"] % 3 == 0 else f"content_{row['id']}",
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500


class TestNullValues:
    """Test estimation accuracy with null values."""

    def test_nullable_integers(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with nullable integer columns."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "value": None if row["id"] % 5 == 0 else row["id"],
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_nullable_strings(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with nullable string columns."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "name": None if row["id"] % 4 == 0 else f"user_{row['id']}",
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_high_null_ratio(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with mostly null values."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                # 90% null
                "value": None if row["id"] % 10 != 0 else row["id"],
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500


class TestSkewedDistributions:
    """Test estimation accuracy with skewed data distributions."""

    def test_moderate_skew(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with moderately skewed distribution."""
        # Create data where some keys have more rows
        data = []
        for i in range(500):
            # Keys 0-4 get more rows than keys 5-9
            key = i % 5 if i < 300 else 5 + (i % 5)
            data.append({"id": key, "value": i})

        ds = ray.data.from_items(data)

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_extreme_skew(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with extreme skew (90% to one key)."""
        data = []
        # 450 rows with key=0
        for i in range(450):
            data.append({"id": 0, "value": i})
        # 50 rows distributed across keys 1-9
        for i in range(50):
            data.append({"id": 1 + (i % 9), "value": 450 + i})

        ds = ray.data.from_items(data)

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_power_law_distribution(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with power-law (Zipf) distribution."""
        # Zipf distribution: key k appears proportional to 1/k
        data = []
        for i in range(500):
            # Approximate Zipf by using logarithm
            key = int(np.log2(i + 1)) % 10
            data.append({"id": key, "value": i})

        ds = ray.data.from_items(data)

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500


class TestEmptyPartitions:
    """Test estimation with empty partitions."""

    def test_more_partitions_than_keys(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test when num_partitions > number of distinct keys."""
        # Only 5 distinct keys but 20 partitions
        ds = ray.data.range(500).map(
            lambda row: {"id": row["id"] % 5, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=20,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_sparse_keys(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with sparse key distribution that creates empty partitions."""
        # Keys are spread out: 0, 100, 200, etc.
        ds = ray.data.range(50).map(
            lambda row: {"id": row["id"] * 100, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 50


class TestBlockStructure:
    """Test estimation with different input block structures."""

    def test_single_input_block(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with a single input block."""
        ds = ray.data.from_items(
            [{"id": i % 10, "value": i} for i in range(200)],
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 200

    def test_many_small_blocks(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with many small input blocks."""
        # Create many small datasets and union them
        ds = ray.data.range(200, override_num_blocks=20).map(
            lambda row: {"id": row["id"] % 10, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 200

    def test_uneven_block_sizes(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with unevenly sized input blocks."""
        # Create data with different amounts per block
        small = ray.data.from_items([{"id": i % 10, "value": i} for i in range(50)])
        large = ray.data.from_items(
            [{"id": i % 10, "value": i + 50} for i in range(150)]
        )
        ds = small.union(large)

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 200


class TestKeyCardinality:
    """Test estimation with different key cardinalities."""

    def test_low_cardinality(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with very low cardinality (< 10 distinct keys)."""
        ds = ray.data.range(500).map(
            lambda row: {"id": row["id"] % 3, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_medium_cardinality(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with medium cardinality (~100 distinct keys)."""
        ds = ray.data.range(500).map(
            lambda row: {"id": row["id"] % 100, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=50,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_high_cardinality_below_threshold(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with cardinality just below the 1000 threshold."""
        ds = ray.data.range(2000).map(
            lambda row: {"id": row["id"] % 900, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=100,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 2000

    def test_high_cardinality_above_threshold(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with cardinality above the 1000 threshold."""
        ds = ray.data.range(5000).map(
            lambda row: {"id": row["id"] % 2000, "value": row["id"]}
        )

        result = ds.repartition(
            num_blocks=100,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 5000


class TestBinaryData:
    """Test estimation with binary data."""

    def test_fixed_size_binary(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with fixed-size binary data."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "data": bytes([row["id"] % 256] * 100),  # 100 bytes each
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_variable_size_binary(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with variable-size binary data."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                # Size varies from 1 to 100 bytes
                "data": bytes([row["id"] % 256] * (row["id"] % 100 + 1)),
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500


class TestListData:
    """Test estimation with list/array columns."""

    def test_fixed_length_lists(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with fixed-length list columns."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "values": [
                    row["id"],
                    row["id"] + 1,
                    row["id"] + 2,
                ],  # Always 3 elements
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_variable_length_lists(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with variable-length list columns."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                # List length varies from 1 to 10
                "values": list(range(row["id"] % 10 + 1)),
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_empty_lists(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with empty list values."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "values": [] if row["id"] % 3 == 0 else [row["id"]],
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500


class TestMixedTypes:
    """Test estimation with mixed column types."""

    def test_mixed_columns(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with a mix of different column types."""
        ds = ray.data.range(500).map(
            lambda row: {
                "id": row["id"] % 10,
                "int_col": row["id"],
                "float_col": float(row["id"]) * 1.5,
                "str_col": f"value_{row['id']}",
                "bool_col": row["id"] % 2 == 0,
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 500

    def test_wide_table(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with many columns (wide table)."""
        ds = ray.data.range(200).map(
            lambda row: {
                "id": row["id"] % 10,
                **{f"col_{i}": row["id"] + i for i in range(20)},
            }
        )

        result = ds.repartition(
            num_blocks=10,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 200


class TestSmallData:
    """Test estimation with very small datasets."""

    def test_single_row(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test with a single row."""
        ds = ray.data.from_items([{"id": 1, "value": 100}])

        result = ds.repartition(
            num_blocks=4,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 1

    def test_few_rows(self, ray_start_regular_shared_2_cpus, strict_estimation_context):
        """Test with just a few rows."""
        ds = ray.data.from_items([{"id": i % 3, "value": i} for i in range(10)])

        result = ds.repartition(
            num_blocks=5,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 10

    def test_rows_less_than_partitions(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test when number of rows < number of partitions."""
        ds = ray.data.from_items([{"id": i, "value": i} for i in range(5)])

        result = ds.repartition(
            num_blocks=20,
            shuffle=True,
            keys=["id"],
        ).materialize()

        assert result.count() == 5


class TestJoinEstimation:
    """Test estimation accuracy for join operations."""

    def test_basic_inner_join(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test basic inner join estimation."""
        left = ray.data.range(200).map(
            lambda row: {"id": row["id"], "left_val": row["id"] * 2}
        )
        right = ray.data.range(200).map(
            lambda row: {"id": row["id"], "right_val": row["id"] * 3}
        )

        result = left.join(
            right,
            join_type="inner",
            num_partitions=10,
            on=("id",),
        ).materialize()

        assert result.count() == 200

    def test_join_with_skew(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test join estimation with skewed data."""
        # Left has skew
        left_data = [{"id": 0, "left_val": i} for i in range(150)]
        left_data += [{"id": i, "left_val": i} for i in range(1, 51)]
        left = ray.data.from_items(left_data)

        # Right is uniform
        right = ray.data.from_items([{"id": i, "right_val": i * 2} for i in range(50)])

        result = left.join(
            right,
            join_type="inner",
            num_partitions=10,
            on=("id",),
        ).materialize()

        # 150 rows with id=0 join with 1 row from right (id=0)
        # 49 rows from left (id=1-49) join with 49 rows from right
        assert result.count() == 150 + 49

    def test_join_size_mismatch(
        self, ray_start_regular_shared_2_cpus, strict_estimation_context
    ):
        """Test join where left and right have very different sizes."""
        left = ray.data.range(500).map(
            lambda row: {"id": row["id"] % 50, "left_val": row["id"]}
        )
        right = ray.data.range(100).map(
            lambda row: {"id": row["id"] % 50, "right_val": row["id"]}
        )

        result = left.join(
            right,
            join_type="inner",
            num_partitions=20,
            on=("id",),
        ).materialize()

        # Each of 50 keys in left has 10 rows (500/50)
        # Each of 50 keys in right has 2 rows (100/50)
        # Inner join: 10 * 2 = 20 rows per key * 50 keys = 1000 rows
        assert result.count() == 1000


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
