"""Shuffle and join memory estimation for Ray Data.

This module provides functions to estimate memory requirements for shuffle
and join operations by performing a full scan of the data to compute
actual partition sizes. This enables accurate skew detection and memory
estimates before executing expensive shuffle operations.
"""

import logging
import math
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import ray
from ray.data._internal.arrow_ops.transform_pyarrow import _hash_partition
from ray.data._internal.shuffle_memory_estimate import (
    AggregatorEstimate,
    PartitionEstimate,
    ShuffleMemoryEstimate,
)
from ray.data.block import Block, BlockAccessor
from ray.data.context import (
    DEFAULT_MAX_HASH_SHUFFLE_AGGREGATORS,
    DataContext,
)

logger = logging.getLogger(__name__)


# Default buffer ratio for memory recommendations
DEFAULT_BUFFER_RATIO = 0.15


@ray.remote
def _compute_partition_sizes(
    block: Block,
    key_columns: List[str],
    num_partitions: int,
) -> Tuple[Dict[int, Dict[str, int]], Set[Tuple]]:
    """Remote function to compute partition sizes for a single block.

    This function hash-partitions the block and counts rows and bytes per
    partition without actually materializing the partitioned data.

    Args:
        block: The input block to analyze.
        key_columns: Column names to use for hash partitioning.
        num_partitions: Target number of partitions.

    Returns:
        A tuple containing:
        - Dict mapping partition_id to {"rows": count, "bytes": estimated_bytes}
        - Set of distinct key tuples observed in this block
    """

    accessor = BlockAccessor.for_block(block)
    table = accessor.to_arrow()

    if table.num_rows == 0:
        return {}, set()

    # Project to key columns for hashing
    projected_table = table.select(key_columns)

    # Get partition assignments for each row
    partition_ids = _hash_partition(projected_table, num_partitions)

    # Estimate bytes per row (average across the block)
    total_bytes = table.nbytes
    bytes_per_row = total_bytes / table.num_rows if table.num_rows > 0 else 0

    # Count rows and estimate bytes per partition
    partition_stats: Dict[int, Dict[str, int]] = {}
    for pid in range(num_partitions):
        mask = partition_ids == pid
        count = int(np.sum(mask))
        if count > 0:
            partition_stats[pid] = {
                "rows": count,
                "bytes": int(count * bytes_per_row),
            }

    # Track distinct keys for cardinality estimation
    # Convert key columns to tuples for hashing
    distinct_keys: Set[Tuple] = set()
    key_arrays = [projected_table.column(col) for col in key_columns]

    # Sample keys if the block is very large to avoid memory issues
    sample_size = min(table.num_rows, 100000)
    if table.num_rows > sample_size:
        # Sample rows for distinct key tracking
        indices = np.random.choice(table.num_rows, sample_size, replace=False)
        for idx in indices:
            key_tuple = tuple(arr[int(idx)].as_py() for arr in key_arrays)
            distinct_keys.add(key_tuple)
    else:
        for i in range(table.num_rows):
            key_tuple = tuple(arr[i].as_py() for arr in key_arrays)
            distinct_keys.add(key_tuple)

    return partition_stats, distinct_keys


def _aggregate_partition_stats(
    results: List[Tuple[Dict[int, Dict[str, int]], Set[Tuple]]],
    num_partitions: int,
    num_aggregators: int,
) -> Tuple[List[PartitionEstimate], int, List[AggregatorEstimate]]:
    """Aggregate partition statistics from all blocks.

    Args:
        results: List of (partition_stats, distinct_keys) from each block.
        num_partitions: Total number of partitions.
        num_aggregators: Number of aggregator workers.

    Returns:
        Tuple of (partition_estimates, key_cardinality, aggregator_estimates).
    """
    # Aggregate partition stats across all blocks
    partition_totals: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "bytes": 0}
    )

    all_distinct_keys: Set[Tuple] = set()

    for partition_stats, distinct_keys in results:
        for pid, stats in partition_stats.items():
            partition_totals[pid]["rows"] += stats["rows"]
            partition_totals[pid]["bytes"] += stats["bytes"]
        all_distinct_keys.update(distinct_keys)

    # Build partition estimates list
    partition_estimates = []
    for pid in range(num_partitions):
        stats = partition_totals.get(pid, {"rows": 0, "bytes": 0})
        partition_estimates.append(
            PartitionEstimate(
                partition_id=pid,
                num_rows=stats["rows"],
                size_bytes=stats["bytes"],
            )
        )

    key_cardinality = len(all_distinct_keys)

    # Compute per-aggregator breakdown (round-robin assignment)
    aggregator_estimates = []
    for agg_id in range(num_aggregators):
        # Partitions are assigned round-robin to aggregators
        assigned_partitions = [
            p for p in partition_estimates if p.partition_id % num_aggregators == agg_id
        ]
        partition_ids = [p.partition_id for p in assigned_partitions]
        total_bytes = sum(p.size_bytes for p in assigned_partitions)
        total_rows = sum(p.num_rows for p in assigned_partitions)
        largest_partition_bytes = (
            max(p.size_bytes for p in assigned_partitions) if assigned_partitions else 0
        )

        aggregator_estimates.append(
            AggregatorEstimate(
                aggregator_id=agg_id,
                partition_ids=partition_ids,
                total_bytes=total_bytes,
                total_rows=total_rows,
                largest_partition_bytes=largest_partition_bytes,
            )
        )

    return partition_estimates, key_cardinality, aggregator_estimates


def _derive_num_aggregators(
    num_partitions: int,
    data_context: Optional[DataContext] = None,
) -> int:
    """Derive the number of aggregators that would be used.

    This mirrors the logic in HashShufflingOperatorBase.

    Args:
        num_partitions: Target number of partitions.
        data_context: Optional DataContext for configuration.

    Returns:
        Number of aggregators that would be used.
    """
    if data_context is None:
        data_context = DataContext.get_current()

    # Get cluster resources
    try:
        cluster_resources = ray.cluster_resources()
        total_cpus = cluster_resources.get("CPU", 1)
    except Exception:
        total_cpus = 1

    # Max aggregators is min of total CPUs and configured max
    max_aggregators = min(
        math.ceil(total_cpus),
        data_context.max_hash_shuffle_aggregators
        or DEFAULT_MAX_HASH_SHUFFLE_AGGREGATORS,
    )

    # Cap at number of partitions
    return min(num_partitions, max_aggregators)


def _get_cluster_memory() -> int:
    """Get total cluster memory available.

    Returns:
        Total memory in bytes, or 0 if unavailable.
    """
    try:
        cluster_resources = ray.cluster_resources()
        # Ray reports memory in bytes
        return int(cluster_resources.get("memory", 0))
    except Exception:
        return 0


def estimate_shuffle_memory(
    blocks: List[Block],
    key_columns: List[str],
    num_partitions: int,
    num_aggregators: Optional[int] = None,
    buffer_ratio: float = DEFAULT_BUFFER_RATIO,
    data_context: Optional[DataContext] = None,
) -> ShuffleMemoryEstimate:
    """Estimate memory requirements for a shuffle operation.

    Performs a full scan of all blocks to compute actual partition sizes,
    enabling accurate skew detection and memory estimates.

    Args:
        blocks: List of input blocks to analyze.
        key_columns: Column names to use for hash partitioning.
        num_partitions: Target number of output partitions.
        num_aggregators: Number of aggregator workers. If None, derived automatically.
        buffer_ratio: Safety buffer ratio (default 0.15 = 15%).
        data_context: Optional DataContext for configuration.

    Returns:
        ShuffleMemoryEstimate with detailed statistics and recommendations.
    """
    if data_context is None:
        data_context = DataContext.get_current()

    if num_aggregators is None:
        num_aggregators = _derive_num_aggregators(num_partitions, data_context)

    # Compute partition sizes in parallel across all blocks
    futures = [
        _compute_partition_sizes.remote(block, key_columns, num_partitions)
        for block in blocks
    ]
    results = ray.get(futures)

    # Aggregate results
    (
        partition_estimates,
        key_cardinality,
        aggregator_estimates,
    ) = _aggregate_partition_stats(results, num_partitions, num_aggregators)

    # Compute total input stats
    total_rows = sum(p.num_rows for p in partition_estimates)
    total_bytes = sum(p.size_bytes for p in partition_estimates)

    # Compute partition size distribution metrics
    sizes = [p.size_bytes for p in partition_estimates]
    non_empty_sizes = [s for s in sizes if s > 0]

    if non_empty_sizes:
        partition_size_min = min(sizes)
        partition_size_max = max(sizes)
        partition_size_mean = statistics.mean(sizes)
        partition_size_std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
    else:
        partition_size_min = 0
        partition_size_max = 0
        partition_size_mean = 0.0
        partition_size_std = 0.0

    # Compute percentiles
    if sizes:
        partition_size_percentiles = {
            50: int(np.percentile(sizes, 50)),
            90: int(np.percentile(sizes, 90)),
            95: int(np.percentile(sizes, 95)),
            99: int(np.percentile(sizes, 99)),
        }
    else:
        partition_size_percentiles = {50: 0, 90: 0, 95: 0, 99: 0}

    # Count empty partitions
    empty_partition_count = sum(1 for s in sizes if s == 0)

    # Identify hot partitions (top 10 by size)
    hot_partitions = sorted(
        partition_estimates, key=lambda p: p.size_bytes, reverse=True
    )[:10]

    # Compute skew metrics
    skew_factor = (
        partition_size_max / partition_size_mean if partition_size_mean > 0 else 0.0
    )

    # Generate skew warnings (partitions > 2x mean)
    skew_warnings = []
    if partition_size_mean > 0:
        for p in partition_estimates:
            ratio = p.size_bytes / partition_size_mean
            if ratio > 2.0:
                skew_warnings.append(
                    f"Partition {p.partition_id}: {ratio:.1f}x mean "
                    f"({p.size_bytes / (1024 * 1024):.1f} MiB)"
                )

    # Find worst-case aggregator
    worst_case_aggregator = max(aggregator_estimates, key=lambda a: a.total_bytes)

    # Calculate memory requirements for shuffle
    # For shuffle: input + output object store + heap for Arrow operations
    aggregator_input_object_store_bytes = worst_case_aggregator.total_bytes
    aggregator_output_object_store_bytes = worst_case_aggregator.total_bytes
    # Heap memory: 1.0x largest partition for Arrow concatenation/processing
    aggregator_heap_memory_bytes = worst_case_aggregator.largest_partition_bytes

    required_memory_per_aggregator = (
        aggregator_input_object_store_bytes
        + aggregator_output_object_store_bytes
        + aggregator_heap_memory_bytes
    )
    buffer_memory_bytes = int(required_memory_per_aggregator * buffer_ratio)
    recommended_memory_per_aggregator = (
        required_memory_per_aggregator + buffer_memory_bytes
    )

    # Get cluster resources
    cluster_memory_available = _get_cluster_memory()
    total_required_memory = required_memory_per_aggregator * num_aggregators
    memory_headroom_ratio = (
        cluster_memory_available / total_required_memory
        if total_required_memory > 0
        else float("inf")
    )

    # Generate recommended ray remote args
    recommended_ray_remote_args = {"memory": recommended_memory_per_aggregator}

    return ShuffleMemoryEstimate(
        # Input statistics
        total_rows=total_rows,
        total_bytes=total_bytes,
        num_partitions=num_partitions,
        num_aggregators=num_aggregators,
        key_columns=key_columns,
        # Partition distribution
        partition_estimates=partition_estimates,
        partition_size_percentiles=partition_size_percentiles,
        empty_partition_count=empty_partition_count,
        key_cardinality=key_cardinality,
        hot_partitions=hot_partitions,
        # Skew metrics
        partition_size_min_bytes=partition_size_min,
        partition_size_max_bytes=partition_size_max,
        partition_size_mean_bytes=partition_size_mean,
        partition_size_std_bytes=partition_size_std,
        skew_factor=skew_factor,
        skew_warnings=skew_warnings,
        # Per-aggregator breakdown
        aggregator_estimates=aggregator_estimates,
        worst_case_aggregator=worst_case_aggregator,
        # Memory requirements
        aggregator_heap_memory_bytes=aggregator_heap_memory_bytes,
        aggregator_input_object_store_bytes=aggregator_input_object_store_bytes,
        aggregator_output_object_store_bytes=aggregator_output_object_store_bytes,
        required_memory_per_aggregator=required_memory_per_aggregator,
        buffer_memory_bytes=buffer_memory_bytes,
        recommended_memory_per_aggregator=recommended_memory_per_aggregator,
        # Cluster comparison
        cluster_memory_available=cluster_memory_available,
        total_required_memory=total_required_memory,
        memory_headroom_ratio=memory_headroom_ratio,
        # Configuration
        recommended_ray_remote_args=recommended_ray_remote_args,
        # Join-specific (not applicable for shuffle)
        left_bytes_per_partition=None,
        right_bytes_per_partition=None,
        estimated_output_size_bytes=None,
        is_join=False,
        _buffer_ratio=buffer_ratio,
    )


def estimate_join_memory(
    left_blocks: List[Block],
    right_blocks: List[Block],
    left_key_columns: List[str],
    right_key_columns: List[str],
    num_partitions: int,
    num_aggregators: Optional[int] = None,
    buffer_ratio: float = DEFAULT_BUFFER_RATIO,
    data_context: Optional[DataContext] = None,
) -> ShuffleMemoryEstimate:
    """Estimate memory requirements for a join operation.

    Performs a full scan of both datasets to compute actual partition sizes,
    enabling accurate skew detection and memory estimates.

    Args:
        left_blocks: List of blocks from the left dataset.
        right_blocks: List of blocks from the right dataset.
        left_key_columns: Key column names for the left dataset.
        right_key_columns: Key column names for the right dataset.
        num_partitions: Target number of output partitions.
        num_aggregators: Number of aggregator workers. If None, derived automatically.
        buffer_ratio: Safety buffer ratio (default 0.15 = 15%).
        data_context: Optional DataContext for configuration.

    Returns:
        ShuffleMemoryEstimate with detailed statistics and recommendations.
    """
    if data_context is None:
        data_context = DataContext.get_current()

    if num_aggregators is None:
        num_aggregators = _derive_num_aggregators(num_partitions, data_context)

    # Compute partition sizes for left dataset
    left_futures = [
        _compute_partition_sizes.remote(block, left_key_columns, num_partitions)
        for block in left_blocks
    ]

    # Compute partition sizes for right dataset
    right_futures = [
        _compute_partition_sizes.remote(block, right_key_columns, num_partitions)
        for block in right_blocks
    ]

    # Wait for all results
    left_results = ray.get(left_futures)
    right_results = ray.get(right_futures)

    # Aggregate left dataset stats
    left_partition_totals: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "bytes": 0}
    )
    left_distinct_keys: Set[Tuple] = set()

    for partition_stats, distinct_keys in left_results:
        for pid, stats in partition_stats.items():
            left_partition_totals[pid]["rows"] += stats["rows"]
            left_partition_totals[pid]["bytes"] += stats["bytes"]
        left_distinct_keys.update(distinct_keys)

    # Aggregate right dataset stats
    right_partition_totals: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "bytes": 0}
    )
    right_distinct_keys: Set[Tuple] = set()

    for partition_stats, distinct_keys in right_results:
        for pid, stats in partition_stats.items():
            right_partition_totals[pid]["rows"] += stats["rows"]
            right_partition_totals[pid]["bytes"] += stats["bytes"]
        right_distinct_keys.update(distinct_keys)

    # Combined partition estimates (sum of left and right)
    partition_estimates = []
    left_bytes_per_partition = []
    right_bytes_per_partition = []

    for pid in range(num_partitions):
        left_stats = left_partition_totals.get(pid, {"rows": 0, "bytes": 0})
        right_stats = right_partition_totals.get(pid, {"rows": 0, "bytes": 0})

        combined_rows = left_stats["rows"] + right_stats["rows"]
        combined_bytes = left_stats["bytes"] + right_stats["bytes"]

        partition_estimates.append(
            PartitionEstimate(
                partition_id=pid,
                num_rows=combined_rows,
                size_bytes=combined_bytes,
            )
        )
        left_bytes_per_partition.append(left_stats["bytes"])
        right_bytes_per_partition.append(right_stats["bytes"])

    # Key cardinality is intersection of both sets (for joins)
    key_cardinality = len(left_distinct_keys | right_distinct_keys)

    # Compute per-aggregator breakdown
    aggregator_estimates = []
    for agg_id in range(num_aggregators):
        assigned_partitions = [
            p for p in partition_estimates if p.partition_id % num_aggregators == agg_id
        ]
        partition_ids = [p.partition_id for p in assigned_partitions]
        total_bytes = sum(p.size_bytes for p in assigned_partitions)
        total_rows = sum(p.num_rows for p in assigned_partitions)
        largest_partition_bytes = (
            max(p.size_bytes for p in assigned_partitions) if assigned_partitions else 0
        )

        aggregator_estimates.append(
            AggregatorEstimate(
                aggregator_id=agg_id,
                partition_ids=partition_ids,
                total_bytes=total_bytes,
                total_rows=total_rows,
                largest_partition_bytes=largest_partition_bytes,
            )
        )

    # Compute totals
    total_rows = sum(p.num_rows for p in partition_estimates)
    total_bytes = sum(p.size_bytes for p in partition_estimates)

    # Compute partition size distribution metrics
    sizes = [p.size_bytes for p in partition_estimates]

    if sizes:
        partition_size_min = min(sizes)
        partition_size_max = max(sizes)
        partition_size_mean = statistics.mean(sizes)
        partition_size_std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
        partition_size_percentiles = {
            50: int(np.percentile(sizes, 50)),
            90: int(np.percentile(sizes, 90)),
            95: int(np.percentile(sizes, 95)),
            99: int(np.percentile(sizes, 99)),
        }
    else:
        partition_size_min = 0
        partition_size_max = 0
        partition_size_mean = 0.0
        partition_size_std = 0.0
        partition_size_percentiles = {50: 0, 90: 0, 95: 0, 99: 0}

    empty_partition_count = sum(1 for s in sizes if s == 0)

    hot_partitions = sorted(
        partition_estimates, key=lambda p: p.size_bytes, reverse=True
    )[:10]

    skew_factor = (
        partition_size_max / partition_size_mean if partition_size_mean > 0 else 0.0
    )

    skew_warnings = []
    if partition_size_mean > 0:
        for p in partition_estimates:
            ratio = p.size_bytes / partition_size_mean
            if ratio > 2.0:
                skew_warnings.append(
                    f"Partition {p.partition_id}: {ratio:.1f}x mean "
                    f"({p.size_bytes / (1024 * 1024):.1f} MiB)"
                )

    worst_case_aggregator = max(aggregator_estimates, key=lambda a: a.total_bytes)

    # Calculate memory requirements for join
    # For joins: input + heap (for join operation) + output
    aggregator_input_object_store_bytes = worst_case_aggregator.total_bytes

    # Join heap memory: largest partition * 2 (100% Arrow join overhead)
    aggregator_heap_memory_bytes = worst_case_aggregator.largest_partition_bytes * 2

    # Output: estimated as largest partition size
    aggregator_output_object_store_bytes = worst_case_aggregator.largest_partition_bytes

    required_memory_per_aggregator = (
        aggregator_input_object_store_bytes
        + aggregator_heap_memory_bytes
        + aggregator_output_object_store_bytes
    )
    buffer_memory_bytes = int(required_memory_per_aggregator * buffer_ratio)
    recommended_memory_per_aggregator = (
        required_memory_per_aggregator + buffer_memory_bytes
    )

    # Get cluster resources
    cluster_memory_available = _get_cluster_memory()
    total_required_memory = required_memory_per_aggregator * num_aggregators
    memory_headroom_ratio = (
        cluster_memory_available / total_required_memory
        if total_required_memory > 0
        else float("inf")
    )

    # Estimate output size (conservative: sum of both inputs for inner join)
    estimated_output_size_bytes = total_bytes

    # Generate recommended ray remote args
    recommended_ray_remote_args = {"memory": recommended_memory_per_aggregator}

    return ShuffleMemoryEstimate(
        # Input statistics
        total_rows=total_rows,
        total_bytes=total_bytes,
        num_partitions=num_partitions,
        num_aggregators=num_aggregators,
        key_columns=left_key_columns,  # Use left key columns as primary
        # Partition distribution
        partition_estimates=partition_estimates,
        partition_size_percentiles=partition_size_percentiles,
        empty_partition_count=empty_partition_count,
        key_cardinality=key_cardinality,
        hot_partitions=hot_partitions,
        # Skew metrics
        partition_size_min_bytes=partition_size_min,
        partition_size_max_bytes=partition_size_max,
        partition_size_mean_bytes=partition_size_mean,
        partition_size_std_bytes=partition_size_std,
        skew_factor=skew_factor,
        skew_warnings=skew_warnings,
        # Per-aggregator breakdown
        aggregator_estimates=aggregator_estimates,
        worst_case_aggregator=worst_case_aggregator,
        # Memory requirements
        aggregator_heap_memory_bytes=aggregator_heap_memory_bytes,
        aggregator_input_object_store_bytes=aggregator_input_object_store_bytes,
        aggregator_output_object_store_bytes=aggregator_output_object_store_bytes,
        required_memory_per_aggregator=required_memory_per_aggregator,
        buffer_memory_bytes=buffer_memory_bytes,
        recommended_memory_per_aggregator=recommended_memory_per_aggregator,
        # Cluster comparison
        cluster_memory_available=cluster_memory_available,
        total_required_memory=total_required_memory,
        memory_headroom_ratio=memory_headroom_ratio,
        # Configuration
        recommended_ray_remote_args=recommended_ray_remote_args,
        # Join-specific
        left_bytes_per_partition=left_bytes_per_partition,
        right_bytes_per_partition=right_bytes_per_partition,
        estimated_output_size_bytes=estimated_output_size_bytes,
        is_join=True,
        _buffer_ratio=buffer_ratio,
    )
