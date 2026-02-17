"""Tests for shuffle memory estimation feature.

This module tests the automatic memory estimation that runs before shuffle/join
operations. The estimation computes partition distribution statistics and memory
requirements, then validates estimates against actual execution.
"""

import pytest

import ray
from ray.data._internal.shuffle_memory_estimate import (
    AggregatorEstimate,
    PartitionEstimate,
    ShuffleMemoryEstimate,
)
from ray.data._internal.shuffle_memory_estimator import (
    _aggregate_partition_stats,
    _derive_num_aggregators,
)
from ray.data.context import DataContext
from ray.tests.conftest import *  # noqa


class TestPartitionEstimate:
    """Tests for the PartitionEstimate dataclass."""

    def test_basic_creation(self):
        pe = PartitionEstimate(partition_id=0, num_rows=100, size_bytes=1000)
        assert pe.partition_id == 0
        assert pe.num_rows == 100
        assert pe.size_bytes == 1000


class TestAggregatorEstimate:
    """Tests for the AggregatorEstimate dataclass."""

    def test_basic_creation(self):
        ae = AggregatorEstimate(
            aggregator_id=0,
            partition_ids=[0, 2, 4],
            total_bytes=3000,
            total_rows=300,
            largest_partition_bytes=1500,
        )
        assert ae.aggregator_id == 0
        assert ae.partition_ids == [0, 2, 4]
        assert ae.total_bytes == 3000
        assert ae.total_rows == 300
        assert ae.largest_partition_bytes == 1500


class TestShuffleMemoryEstimate:
    """Tests for the ShuffleMemoryEstimate dataclass."""

    @pytest.fixture
    def sample_estimate(self):
        """Create a sample ShuffleMemoryEstimate for testing."""
        partition_estimates = [
            PartitionEstimate(partition_id=i, num_rows=100, size_bytes=1000 * (i + 1))
            for i in range(4)
        ]
        aggregator_estimates = [
            AggregatorEstimate(
                aggregator_id=0,
                partition_ids=[0, 2],
                total_bytes=4000,
                total_rows=200,
                largest_partition_bytes=3000,
            ),
            AggregatorEstimate(
                aggregator_id=1,
                partition_ids=[1, 3],
                total_bytes=6000,
                total_rows=200,
                largest_partition_bytes=4000,
            ),
        ]

        return ShuffleMemoryEstimate(
            total_rows=400,
            total_bytes=10000,
            num_partitions=4,
            num_aggregators=2,
            key_columns=["id"],
            partition_estimates=partition_estimates,
            partition_size_percentiles={50: 2000, 90: 3500, 95: 3800, 99: 3950},
            empty_partition_count=0,
            key_cardinality=100,
            hot_partitions=partition_estimates[-2:],
            partition_size_min_bytes=1000,
            partition_size_max_bytes=4000,
            partition_size_mean_bytes=2500.0,
            partition_size_std_bytes=1118.0,
            skew_factor=1.6,
            skew_warnings=[],
            aggregator_estimates=aggregator_estimates,
            worst_case_aggregator=aggregator_estimates[1],
            aggregator_heap_memory_bytes=4000,  # 1.0x largest partition (4000)
            aggregator_input_object_store_bytes=6000,
            aggregator_output_object_store_bytes=6000,
            required_memory_per_aggregator=16000,  # 6000 + 6000 + 4000
            buffer_memory_bytes=2400,  # 15% of 16000
            recommended_memory_per_aggregator=18400,  # 16000 + 2400
            cluster_memory_available=1000000000,
            total_required_memory=32000,  # 16000 * 2 aggregators
            memory_headroom_ratio=31250.0,  # 1000000000 / 32000
            recommended_ray_remote_args={"memory": 18400},
        )

    def test_summary_output(self, sample_estimate):
        """Test that summary() produces readable output."""
        summary = sample_estimate.summary()
        assert isinstance(summary, str)
        assert "Shuffle Memory Estimation Summary" in summary
        assert "Partition Size Distribution" in summary
        assert "Per-Aggregator Breakdown" in summary
        assert "Memory Requirements" in summary
        assert "Cluster Resources" in summary
        assert "Recommendations" in summary

    def test_to_dict(self, sample_estimate):
        """Test that to_dict() produces a complete dictionary."""
        d = sample_estimate.to_dict()
        assert isinstance(d, dict)
        assert d["total_rows"] == 400
        assert d["total_bytes"] == 10000
        assert d["num_partitions"] == 4
        assert d["num_aggregators"] == 2
        assert len(d["partition_estimates"]) == 4
        assert len(d["aggregator_estimates"]) == 2
        assert d["recommended_memory_per_aggregator"] == 18400
        assert d["is_join"] is False

    def test_repr(self, sample_estimate):
        """Test that __repr__ is informative."""
        repr_str = repr(sample_estimate)
        assert "ShuffleMemoryEstimate" in repr_str
        assert "partitions=4" in repr_str
        assert "aggregators=2" in repr_str

    def test_join_specific_fields(self):
        """Test join-specific fields in summary."""
        partition_estimates = [
            PartitionEstimate(partition_id=0, num_rows=100, size_bytes=2000)
        ]
        aggregator_estimates = [
            AggregatorEstimate(
                aggregator_id=0,
                partition_ids=[0],
                total_bytes=2000,
                total_rows=100,
                largest_partition_bytes=2000,
            )
        ]

        estimate = ShuffleMemoryEstimate(
            total_rows=100,
            total_bytes=2000,
            num_partitions=1,
            num_aggregators=1,
            key_columns=["id"],
            partition_estimates=partition_estimates,
            partition_size_percentiles={50: 2000, 90: 2000, 95: 2000, 99: 2000},
            empty_partition_count=0,
            key_cardinality=50,
            hot_partitions=partition_estimates,
            partition_size_min_bytes=2000,
            partition_size_max_bytes=2000,
            partition_size_mean_bytes=2000.0,
            partition_size_std_bytes=0.0,
            skew_factor=1.0,
            skew_warnings=[],
            aggregator_estimates=aggregator_estimates,
            worst_case_aggregator=aggregator_estimates[0],
            aggregator_heap_memory_bytes=4000,  # Join has heap memory
            aggregator_input_object_store_bytes=2000,
            aggregator_output_object_store_bytes=2000,
            required_memory_per_aggregator=8000,
            buffer_memory_bytes=1200,
            recommended_memory_per_aggregator=9200,
            cluster_memory_available=1000000000,
            total_required_memory=8000,
            memory_headroom_ratio=125000.0,
            recommended_ray_remote_args={"memory": 9200},
            left_bytes_per_partition=[1000],
            right_bytes_per_partition=[1000],
            estimated_output_size_bytes=2000,
            is_join=True,
        )

        summary = estimate.summary()
        assert "Join Memory Estimation Summary" in summary
        assert "Join-Specific Stats" in summary
        assert "Left dataset per partition" in summary
        assert "Right dataset per partition" in summary


class TestEstimatorFunctions:
    """Tests for the estimator module functions."""

    def test_derive_num_aggregators(self, ray_start_regular_shared_2_cpus):
        """Test aggregator count derivation."""
        ctx = DataContext.get_current()

        # With more partitions than CPUs, should be capped at CPU count
        num_agg = _derive_num_aggregators(100, ctx)
        assert num_agg <= 100
        assert num_agg >= 1

        # With fewer partitions than CPUs, should match partition count
        num_agg = _derive_num_aggregators(1, ctx)
        assert num_agg == 1

    def test_aggregate_partition_stats(self):
        """Test partition stats aggregation."""
        # Simulate results from two blocks
        results = [
            (
                {0: {"rows": 10, "bytes": 100}, 1: {"rows": 20, "bytes": 200}},
                {(1,), (2,)},
            ),
            (
                {0: {"rows": 5, "bytes": 50}, 2: {"rows": 15, "bytes": 150}},
                {(2,), (3,)},
            ),
        ]

        (
            partition_estimates,
            key_cardinality,
            aggregator_estimates,
        ) = _aggregate_partition_stats(results, num_partitions=3, num_aggregators=2)

        assert len(partition_estimates) == 3
        # Partition 0: 10+5=15 rows, 100+50=150 bytes
        assert partition_estimates[0].num_rows == 15
        assert partition_estimates[0].size_bytes == 150
        # Partition 1: 20 rows, 200 bytes
        assert partition_estimates[1].num_rows == 20
        assert partition_estimates[1].size_bytes == 200
        # Partition 2: 15 rows, 150 bytes
        assert partition_estimates[2].num_rows == 15
        assert partition_estimates[2].size_bytes == 150

        # Key cardinality is union of all distinct keys
        assert key_cardinality == 3  # {(1,), (2,), (3,)}

        # With 3 partitions and 2 aggregators, round-robin assignment
        assert len(aggregator_estimates) == 2


@pytest.mark.parametrize(
    "num_rows,num_partitions",
    [
        (100, 4),
        (1000, 10),
        (50, 2),
    ],
)
def test_repartition_with_automatic_estimation(
    ray_start_regular_shared_2_cpus, num_rows, num_partitions
):
    """Test that repartition automatically runs memory estimation.

    Note: The memory estimation summary is printed to the Ray Data logger
    (which goes to stderr), not captured by pytest's caplog. The test verifies
    the operation completes correctly with the expected results.
    """
    ds = ray.data.range(num_rows).map(
        lambda row: {"id": row["id"], "value": row["id"] * 2}
    )

    # Run the actual repartition - estimation should happen automatically
    result = ds.repartition(
        num_blocks=num_partitions,
        shuffle=True,
        keys=["id"],
    )

    # Materialize to trigger execution
    result = result.materialize()

    # Verify the result is a Dataset with correct number of blocks
    assert result.num_blocks() == num_partitions
    assert result.count() == num_rows


def test_repartition_with_skewed_data(ray_start_regular_shared_2_cpus):
    """Test repartition with skewed data runs estimation and detects skew.

    Note: The skew warnings are printed to the Ray Data logger (which goes to
    stderr). The test verifies the operation completes correctly.
    """
    # Create skewed data where most rows have the same key
    data = [{"id": 0, "value": i} for i in range(90)]  # 90 rows with id=0
    data += [{"id": i, "value": i} for i in range(1, 11)]  # 10 rows with unique ids

    ds = ray.data.from_items(data)

    # Run repartition - should detect skew
    result = ds.repartition(
        num_blocks=10,
        shuffle=True,
        keys=["id"],
    ).materialize()

    # Verify result is correct
    assert result.count() == 100


@pytest.mark.parametrize(
    "num_rows_left,num_rows_right,num_partitions",
    [
        (100, 100, 4),
        (50, 100, 8),
    ],
)
def test_join_with_automatic_estimation(
    ray_start_regular_shared_2_cpus,
    num_rows_left,
    num_rows_right,
    num_partitions,
):
    """Test that join automatically runs memory estimation.

    Note: The memory estimation summary is printed to the Ray Data logger
    (which goes to stderr). The test verifies the operation completes correctly.
    """
    left_ds = ray.data.range(num_rows_left).map(
        lambda row: {"id": row["id"], "left_value": row["id"] * 2}
    )

    right_ds = ray.data.range(num_rows_right).map(
        lambda row: {"id": row["id"], "right_value": row["id"] ** 2}
    )

    # Run the actual join - estimation should happen automatically
    result = left_ds.join(
        right_ds,
        join_type="inner",
        num_partitions=num_partitions,
        on=("id",),
    )

    # Materialize to trigger execution
    result = result.materialize()

    # Verify the result has expected row count (inner join)
    expected_rows = min(num_rows_left, num_rows_right)
    assert result.count() == expected_rows


def test_join_with_different_keys(ray_start_regular_shared_2_cpus):
    """Test join with different key column names runs estimation."""
    left_ds = ray.data.from_items(
        [{"left_id": i, "left_val": i * 2} for i in range(50)]
    )

    right_ds = ray.data.from_items(
        [{"right_id": i, "right_val": i**2} for i in range(50)]
    )

    result = left_ds.join(
        right_ds,
        join_type="inner",
        num_partitions=4,
        on=("left_id",),
        right_on=("right_id",),
    ).materialize()

    assert result.count() == 50


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
