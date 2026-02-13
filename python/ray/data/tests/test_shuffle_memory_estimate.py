"""Tests for shuffle memory estimation feature.

This module tests the estimate_only=True mode for repartition() and join()
operations, which provides detailed memory requirements and partition
distribution statistics without executing the actual shuffle.
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
            aggregator_heap_memory_bytes=0,
            aggregator_input_object_store_bytes=6000,
            aggregator_output_object_store_bytes=6000,
            required_memory_per_aggregator=12000,
            buffer_memory_bytes=1800,
            recommended_memory_per_aggregator=13800,
            cluster_memory_available=1000000000,
            total_required_memory=24000,
            memory_headroom_ratio=41666.67,
            recommended_ray_remote_args={"memory": 13800},
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
        assert d["recommended_memory_per_aggregator"] == 13800
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
def test_repartition_estimate_only_basic(
    ray_start_regular_shared_2_cpus, num_rows, num_partitions
):
    """Test basic estimate_only functionality for repartition."""
    ds = ray.data.range(num_rows).map(
        lambda row: {"id": row["id"], "value": row["id"] * 2}
    )

    estimate = ds.repartition(
        num_blocks=num_partitions,
        shuffle=True,
        keys=["id"],
        estimate_only=True,
    )

    assert isinstance(estimate, ShuffleMemoryEstimate)
    assert estimate.num_partitions == num_partitions
    assert estimate.total_rows == num_rows
    assert estimate.key_columns == ["id"]
    assert len(estimate.partition_estimates) == num_partitions
    assert estimate.recommended_memory_per_aggregator > 0


def test_repartition_estimate_only_requires_shuffle(ray_start_regular_shared_2_cpus):
    """Test that estimate_only requires shuffle=True."""
    ds = ray.data.range(100)

    with pytest.raises(ValueError, match="requires `shuffle=True`"):
        ds.repartition(num_blocks=4, shuffle=False, estimate_only=True)


def test_repartition_estimate_only_requires_keys(ray_start_regular_shared_2_cpus):
    """Test that estimate_only requires keys to be specified."""
    ds = ray.data.range(100)

    with pytest.raises(ValueError, match="requires `keys` to be specified"):
        ds.repartition(num_blocks=4, shuffle=True, estimate_only=True)


def test_repartition_estimate_only_not_with_target_rows(
    ray_start_regular_shared_2_cpus,
):
    """Test that estimate_only is not supported with target_num_rows_per_block."""
    ds = ray.data.range(100)

    # estimate_only requires shuffle=True, so without shuffle it fails first
    with pytest.raises(ValueError, match="requires `shuffle=True`"):
        ds.repartition(
            target_num_rows_per_block=10,
            estimate_only=True,
        )

    # If shuffle=True with target_num_rows_per_block, we get the incompatibility error
    # (target_num_rows_per_block and shuffle=True are not compatible)
    with pytest.raises(ValueError, match="`shuffle` must be False"):
        ds.repartition(
            target_num_rows_per_block=10,
            shuffle=True,
            estimate_only=True,
        )


def test_repartition_estimate_skew_detection(ray_start_regular_shared_2_cpus):
    """Test skew detection in repartition estimates."""
    # Create skewed data where most rows have the same key
    data = [{"id": 0, "value": i} for i in range(90)]  # 90 rows with id=0
    data += [{"id": i, "value": i} for i in range(1, 11)]  # 10 rows with unique ids

    ds = ray.data.from_items(data)

    estimate = ds.repartition(
        num_blocks=10,
        shuffle=True,
        keys=["id"],
        estimate_only=True,
    )

    assert isinstance(estimate, ShuffleMemoryEstimate)
    # With skewed data, the skew factor should be > 1
    assert estimate.skew_factor > 1.0
    # The hot partition should contain most of the data
    assert estimate.hot_partitions[0].num_rows > 50


def test_repartition_estimate_empty_partitions(ray_start_regular_shared_2_cpus):
    """Test detection of empty partitions."""
    # Create data with only a few distinct keys but many partitions
    data = [{"id": i % 3, "value": i} for i in range(30)]
    ds = ray.data.from_items(data)

    estimate = ds.repartition(
        num_blocks=100,  # Many more partitions than distinct keys
        shuffle=True,
        keys=["id"],
        estimate_only=True,
    )

    assert isinstance(estimate, ShuffleMemoryEstimate)
    # Most partitions should be empty
    assert estimate.empty_partition_count > 90


@pytest.mark.parametrize(
    "num_rows_left,num_rows_right,num_partitions",
    [
        (100, 100, 4),
        (50, 100, 8),
        (100, 50, 8),
    ],
)
def test_join_estimate_only_basic(
    ray_start_regular_shared_2_cpus, num_rows_left, num_rows_right, num_partitions
):
    """Test basic estimate_only functionality for join."""
    left_ds = ray.data.range(num_rows_left).map(
        lambda row: {"id": row["id"], "left_value": row["id"] * 2}
    )

    right_ds = ray.data.range(num_rows_right).map(
        lambda row: {"id": row["id"], "right_value": row["id"] ** 2}
    )

    estimate = left_ds.join(
        right_ds,
        join_type="inner",
        num_partitions=num_partitions,
        on=("id",),
        estimate_only=True,
    )

    assert isinstance(estimate, ShuffleMemoryEstimate)
    assert estimate.is_join is True
    assert estimate.num_partitions == num_partitions
    assert estimate.total_rows == num_rows_left + num_rows_right
    assert estimate.key_columns == ["id"]
    assert estimate.left_bytes_per_partition is not None
    assert estimate.right_bytes_per_partition is not None
    assert len(estimate.left_bytes_per_partition) == num_partitions
    assert len(estimate.right_bytes_per_partition) == num_partitions
    # Join should have heap memory for the in-memory join operation
    assert estimate.aggregator_heap_memory_bytes > 0


def test_join_estimate_only_with_different_keys(ray_start_regular_shared_2_cpus):
    """Test join estimation with different key column names."""
    left_ds = ray.data.from_items(
        [{"left_id": i, "left_val": i * 2} for i in range(50)]
    )

    right_ds = ray.data.from_items(
        [{"right_id": i, "right_val": i**2} for i in range(50)]
    )

    estimate = left_ds.join(
        right_ds,
        join_type="inner",
        num_partitions=4,
        on=("left_id",),
        right_on=("right_id",),
        estimate_only=True,
    )

    assert isinstance(estimate, ShuffleMemoryEstimate)
    assert estimate.key_columns == ["left_id"]


def test_join_estimate_uses_both_datasets(ray_start_regular_shared_2_cpus):
    """Test that join estimation considers both datasets."""
    # Create datasets of different sizes
    small_ds = ray.data.from_items([{"id": i, "val": i} for i in range(10)])

    large_ds = ray.data.from_items([{"id": i, "val": i} for i in range(100)])

    estimate = small_ds.join(
        large_ds,
        join_type="inner",
        num_partitions=4,
        on=("id",),
        estimate_only=True,
    )

    # Total bytes should account for both datasets
    assert estimate.total_bytes > 0
    assert estimate.total_rows == 110  # 10 + 100


def test_estimate_recommended_args_can_be_used(ray_start_regular_shared_2_cpus):
    """Test that recommended_ray_remote_args can be passed to actual operations."""
    ds = ray.data.range(100).map(lambda row: {"id": row["id"], "value": row["id"] * 2})

    # Get the estimate
    estimate = ds.repartition(
        num_blocks=4,
        shuffle=True,
        keys=["id"],
        estimate_only=True,
    )

    # Verify the recommended args have the expected structure
    assert "memory" in estimate.recommended_ray_remote_args
    assert isinstance(estimate.recommended_ray_remote_args["memory"], int)
    assert estimate.recommended_ray_remote_args["memory"] > 0


def test_estimate_summary_formatting(ray_start_regular_shared_2_cpus):
    """Test that summary() produces well-formatted output."""
    ds = ray.data.range(1000).map(
        lambda row: {"id": row["id"] % 10, "value": row["id"] * 2}
    )

    estimate = ds.repartition(
        num_blocks=20,
        shuffle=True,
        keys=["id"],
        estimate_only=True,
    )

    summary = estimate.summary()

    # Check key sections are present
    assert "Shuffle Memory Estimation Summary" in summary
    assert "Input:" in summary
    assert "Partitions:" in summary
    assert "Key Cardinality:" in summary
    assert "Partition Size Distribution:" in summary
    assert "Hot Partitions" in summary
    assert "Per-Aggregator Breakdown:" in summary
    assert "Memory Requirements" in summary
    assert "Cluster Resources:" in summary
    assert "Recommendations:" in summary
    assert "Skew Warnings:" in summary


def test_estimate_to_dict_completeness(ray_start_regular_shared_2_cpus):
    """Test that to_dict() returns all expected fields."""
    ds = ray.data.range(100).map(lambda row: {"id": row["id"], "value": row["id"] * 2})

    estimate = ds.repartition(
        num_blocks=4,
        shuffle=True,
        keys=["id"],
        estimate_only=True,
    )

    d = estimate.to_dict()

    # Check all expected keys are present
    expected_keys = [
        "total_rows",
        "total_bytes",
        "num_partitions",
        "num_aggregators",
        "key_columns",
        "partition_estimates",
        "partition_size_percentiles",
        "empty_partition_count",
        "key_cardinality",
        "hot_partitions",
        "partition_size_min_bytes",
        "partition_size_max_bytes",
        "partition_size_mean_bytes",
        "partition_size_std_bytes",
        "skew_factor",
        "skew_warnings",
        "aggregator_estimates",
        "worst_case_aggregator",
        "aggregator_heap_memory_bytes",
        "aggregator_input_object_store_bytes",
        "aggregator_output_object_store_bytes",
        "required_memory_per_aggregator",
        "buffer_memory_bytes",
        "recommended_memory_per_aggregator",
        "cluster_memory_available",
        "total_required_memory",
        "memory_headroom_ratio",
        "recommended_ray_remote_args",
        "left_bytes_per_partition",
        "right_bytes_per_partition",
        "estimated_output_size_bytes",
        "is_join",
    ]

    for key in expected_keys:
        assert key in d, f"Missing key: {key}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
