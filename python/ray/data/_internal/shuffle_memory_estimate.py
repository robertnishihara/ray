"""Dataclasses for shuffle/join memory estimation results.

This module provides structured output for the estimate_only=True mode of
repartition() and join() operations, enabling users to understand memory
requirements before executing expensive shuffle operations.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ray.data._internal.util import GiB, MiB


@dataclass
class PartitionEstimate:
    """Statistics for a single partition after hash-partitioning.

    Attributes:
        partition_id: The partition index (0 to num_partitions-1).
        num_rows: Number of rows in this partition.
        size_bytes: Exact size in bytes of this partition.
        distinct_keys: Number of distinct key values in this partition.
    """

    partition_id: int
    num_rows: int
    size_bytes: int
    distinct_keys: int = 0


@dataclass
class AggregatorEstimate:
    """Estimated resource usage for a single aggregator worker.

    In a hash shuffle, partitions are assigned to aggregators in round-robin
    fashion. This class captures the estimated load for one aggregator.

    Attributes:
        aggregator_id: The aggregator index (0 to num_aggregators-1).
        partition_ids: List of partition IDs assigned to this aggregator.
        total_bytes: Sum of sizes of all assigned partitions.
        total_rows: Sum of rows across all assigned partitions.
        largest_partition_bytes: Size of the largest partition assigned.
    """

    aggregator_id: int
    partition_ids: List[int]
    total_bytes: int
    total_rows: int
    largest_partition_bytes: int


@dataclass
class ShuffleMemoryEstimate:
    """Complete memory estimation for a shuffle or join operation.

    This class provides detailed statistics about partition distribution,
    memory requirements, and configuration recommendations. It is returned
    when calling repartition() or join() with estimate_only=True.

    The estimation is based on a full scan of the data to compute actual
    partition sizes, enabling accurate skew detection and memory estimates.

    Note:
        Estimates are based on the current dataset size and do not account
        for optimizations like map-side aggregation in groupby operations.
        For pipelines with combiners, the actual shuffle may be smaller
        than estimated.

    Attributes:
        # Input statistics
        total_rows: Total number of rows in the input dataset.
        total_bytes: Total size in bytes of the input dataset.
        num_partitions: Number of output partitions requested.
        num_aggregators: Number of aggregator workers that will be used.
        key_columns: Column names used for hash partitioning.

        # Partition distribution
        partition_estimates: Per-partition size and row counts.
        partition_size_percentiles: Size percentiles (p50, p90, p95, p99) in bytes.
        empty_partition_count: Number of partitions with zero rows.
        key_cardinality: Number of distinct key values observed.
        hot_partitions: Top 10 largest partitions by size.

        # Skew metrics
        partition_size_min_bytes: Smallest partition size.
        partition_size_max_bytes: Largest partition size.
        partition_size_mean_bytes: Average partition size.
        partition_size_std_bytes: Standard deviation of partition sizes.
        skew_factor: Ratio of max partition size to mean (>2.0 indicates significant skew).
        skew_warnings: List of warnings for partitions >2x mean size.

        # Per-aggregator breakdown
        aggregator_estimates: Resource estimates for each aggregator.
        worst_case_aggregator: The aggregator with the most data to process.

        # Memory requirements (per aggregator, worst-case)
        aggregator_heap_memory_bytes: Memory needed for in-memory operations (joins).
        aggregator_input_object_store_bytes: Object store memory for incoming data.
        aggregator_output_object_store_bytes: Object store memory for output blocks.
        required_memory_per_aggregator: Minimum memory needed (sum of above).
        buffer_memory_bytes: Safety buffer for fragmentation/temp allocations.
        recommended_memory_per_aggregator: Required + buffer.

        # Cluster comparison
        cluster_memory_available: Total cluster memory in bytes.
        total_required_memory: Required memory across all aggregators.
        memory_headroom_ratio: Available/required ratio (>1.0 means sufficient).

        # Ready-to-use configuration
        recommended_ray_remote_args: Dict that can be passed to aggregator_ray_remote_args.

        # Join-specific fields (only populated for join estimates)
        left_bytes_per_partition: Size distribution for left dataset partitions.
        right_bytes_per_partition: Size distribution for right dataset partitions.
        estimated_output_size_bytes: Estimated output size after join.
        is_join: Whether this estimate is for a join operation.
    """

    # Input statistics
    total_rows: int
    total_bytes: int
    num_partitions: int
    num_aggregators: int
    key_columns: List[str]

    # Partition distribution
    partition_estimates: List[PartitionEstimate]
    partition_size_percentiles: Dict[int, int]
    empty_partition_count: int
    key_cardinality: int
    hot_partitions: List[PartitionEstimate]

    # Skew metrics
    partition_size_min_bytes: int
    partition_size_max_bytes: int
    partition_size_mean_bytes: float
    partition_size_std_bytes: float
    skew_factor: float
    skew_warnings: List[str]

    # Per-aggregator breakdown
    aggregator_estimates: List[AggregatorEstimate]
    worst_case_aggregator: AggregatorEstimate

    # Memory requirements (per aggregator, worst-case)
    aggregator_heap_memory_bytes: int
    aggregator_input_object_store_bytes: int
    aggregator_output_object_store_bytes: int
    required_memory_per_aggregator: int
    buffer_memory_bytes: int
    recommended_memory_per_aggregator: int

    # Cluster comparison
    cluster_memory_available: int
    total_required_memory: int
    memory_headroom_ratio: float

    # Ready-to-use configuration
    recommended_ray_remote_args: Dict[str, Any]

    # Join-specific fields
    left_bytes_per_partition: Optional[List[int]] = None
    right_bytes_per_partition: Optional[List[int]] = None
    estimated_output_size_bytes: Optional[int] = None
    is_join: bool = False

    # Buffer ratio used for recommendations
    _buffer_ratio: float = field(default=0.15, repr=False)

    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes into human-readable form."""
        if size_bytes >= GiB:
            return f"{size_bytes / GiB:.1f} GiB"
        elif size_bytes >= MiB:
            return f"{size_bytes / MiB:.1f} MiB"
        else:
            return f"{size_bytes / 1024:.1f} KiB"

    def summary(self) -> str:
        """Generate a human-readable summary of the memory estimation.

        Returns:
            A formatted string containing key statistics, skew analysis,
            memory requirements, and recommendations.
        """
        lines = []
        operation = "Join" if self.is_join else "Shuffle"
        lines.append(f"{operation} Memory Estimation Summary")
        lines.append("=" * (len(lines[0])))
        lines.append(
            f"Input: {self.total_rows:,} rows, {self._format_bytes(self.total_bytes)}"
        )
        lines.append(
            f"Partitions: {self.num_partitions}, Aggregators: {self.num_aggregators}"
        )
        lines.append(f"Key Columns: {', '.join(self.key_columns)}")
        lines.append(f"Key Cardinality: {self.key_cardinality:,} distinct values")
        lines.append("")

        # Partition size distribution
        lines.append("Partition Size Distribution:")
        p50 = self.partition_size_percentiles.get(50, 0)
        p90 = self.partition_size_percentiles.get(90, 0)
        p99 = self.partition_size_percentiles.get(99, 0)
        lines.append(
            f"  Min: {self._format_bytes(self.partition_size_min_bytes):>12}    "
            f"p50: {self._format_bytes(p50)}"
        )
        lines.append(
            f"  Max: {self._format_bytes(self.partition_size_max_bytes):>12}    "
            f"p90: {self._format_bytes(p90)}"
        )
        lines.append(
            f"  Mean: {self._format_bytes(int(self.partition_size_mean_bytes)):>11}   "
            f"p99: {self._format_bytes(p99)}"
        )
        lines.append(
            f"  Std: {self._format_bytes(int(self.partition_size_std_bytes)):>12}"
        )
        lines.append(f"  Empty Partitions: {self.empty_partition_count}")
        lines.append(f"  Skew Factor: {self.skew_factor:.2f}x")
        lines.append("")

        # Hot partitions
        if self.hot_partitions:
            lines.append("Hot Partitions (top 5):")
            for p in self.hot_partitions[:5]:
                ratio = (
                    p.size_bytes / self.partition_size_mean_bytes
                    if self.partition_size_mean_bytes > 0
                    else 0
                )
                keys_info = f", {p.distinct_keys:,} keys" if p.distinct_keys > 0 else ""
                lines.append(
                    f"  Partition {p.partition_id}: "
                    f"{self._format_bytes(p.size_bytes)} ({ratio:.2f}x mean{keys_info})"
                )
            lines.append("")

        # Per-aggregator breakdown
        lines.append("Per-Aggregator Breakdown:")
        partitions_per_agg = self.num_partitions // self.num_aggregators
        lines.append(
            f"  Aggregators handle ~{partitions_per_agg} partitions each "
            f"({self.num_partitions} / {self.num_aggregators})"
        )
        lines.append(
            f"  Worst-case aggregator #{self.worst_case_aggregator.aggregator_id}: "
            f"{self._format_bytes(self.worst_case_aggregator.total_bytes)} "
            f"(partitions {self.worst_case_aggregator.partition_ids[:5]}"
            f"{'...' if len(self.worst_case_aggregator.partition_ids) > 5 else ''})"
        )
        # Find best-case aggregator
        best_agg = min(self.aggregator_estimates, key=lambda a: a.total_bytes)
        lines.append(
            f"  Best-case aggregator #{best_agg.aggregator_id}: "
            f"{self._format_bytes(best_agg.total_bytes)}"
        )
        lines.append("")

        # Memory requirements
        lines.append("Memory Requirements (per aggregator, worst-case):")
        lines.append(
            f"  Object Store (input):  {self._format_bytes(self.aggregator_input_object_store_bytes)}"
        )
        lines.append(
            f"  Object Store (output): {self._format_bytes(self.aggregator_output_object_store_bytes)}"
        )
        lines.append(
            f"  Heap Memory: {self._format_bytes(self.aggregator_heap_memory_bytes)}"
        )
        lines.append("  " + "-" * 35)
        lines.append(
            f"  Required:     {self._format_bytes(self.required_memory_per_aggregator)}  "
            f"(minimum based on actual partition sizes)"
        )
        buffer_pct = int(self._buffer_ratio * 100)
        lines.append(
            f"  Buffer ({buffer_pct}%):  {self._format_bytes(self.buffer_memory_bytes)}  "
            f"(for fragmentation/temp allocations)"
        )
        lines.append(
            f"  Recommended:  {self._format_bytes(self.recommended_memory_per_aggregator)}"
        )
        lines.append("")

        # Cluster resources
        lines.append("Cluster Resources:")
        lines.append(
            f"  Available memory: {self._format_bytes(self.cluster_memory_available)}"
        )
        lines.append(
            f"  Required (all aggregators): {self._format_bytes(self.total_required_memory)}"
        )
        status = "\u2713" if self.memory_headroom_ratio >= 1.0 else "\u2717"
        lines.append(f"  Headroom: {self.memory_headroom_ratio:.1f}x {status}")
        lines.append("")

        # Join-specific stats
        if self.is_join and self.left_bytes_per_partition is not None:
            lines.append("Join-Specific Stats:")
            left_sizes = self.left_bytes_per_partition
            right_sizes = self.right_bytes_per_partition or []
            if left_sizes:
                left_avg = statistics.mean(left_sizes) if left_sizes else 0
                left_max = max(left_sizes) if left_sizes else 0
                lines.append(
                    f"  Left dataset per partition:  "
                    f"avg {self._format_bytes(int(left_avg))}, "
                    f"max {self._format_bytes(left_max)}"
                )
            if right_sizes:
                right_avg = statistics.mean(right_sizes) if right_sizes else 0
                right_max = max(right_sizes) if right_sizes else 0
                lines.append(
                    f"  Right dataset per partition: "
                    f"avg {self._format_bytes(int(right_avg))}, "
                    f"max {self._format_bytes(right_max)}"
                )
            if self.estimated_output_size_bytes:
                lines.append(
                    f"  Estimated output size: "
                    f"{self._format_bytes(self.estimated_output_size_bytes)}"
                )
            lines.append("")

        # Recommendations
        lines.append("Recommendations:")
        memory_arg = self.recommended_ray_remote_args.get("memory", 0)
        lines.append(f'  aggregator_ray_remote_args={{"memory": {memory_arg}}}')
        lines.append("")

        # Skew warnings
        if self.skew_warnings:
            lines.append("\u26a0 Skew Warnings:")
            for warning in self.skew_warnings[:10]:
                lines.append(f"  {warning}")
            if len(self.skew_warnings) > 10:
                lines.append(f"  ... and {len(self.skew_warnings) - 10} more")
        else:
            lines.append("\u26a0 Skew Warnings:")
            lines.append("  None (all partitions within 2x of mean)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the estimate to a dictionary for programmatic access.

        Returns:
            A dictionary containing all estimation fields with their values.
        """
        return {
            # Input statistics
            "total_rows": self.total_rows,
            "total_bytes": self.total_bytes,
            "num_partitions": self.num_partitions,
            "num_aggregators": self.num_aggregators,
            "key_columns": self.key_columns,
            # Partition distribution
            "partition_estimates": [
                {
                    "partition_id": p.partition_id,
                    "num_rows": p.num_rows,
                    "size_bytes": p.size_bytes,
                }
                for p in self.partition_estimates
            ],
            "partition_size_percentiles": self.partition_size_percentiles,
            "empty_partition_count": self.empty_partition_count,
            "key_cardinality": self.key_cardinality,
            "hot_partitions": [
                {
                    "partition_id": p.partition_id,
                    "num_rows": p.num_rows,
                    "size_bytes": p.size_bytes,
                }
                for p in self.hot_partitions
            ],
            # Skew metrics
            "partition_size_min_bytes": self.partition_size_min_bytes,
            "partition_size_max_bytes": self.partition_size_max_bytes,
            "partition_size_mean_bytes": self.partition_size_mean_bytes,
            "partition_size_std_bytes": self.partition_size_std_bytes,
            "skew_factor": self.skew_factor,
            "skew_warnings": self.skew_warnings,
            # Per-aggregator breakdown
            "aggregator_estimates": [
                {
                    "aggregator_id": a.aggregator_id,
                    "partition_ids": a.partition_ids,
                    "total_bytes": a.total_bytes,
                    "total_rows": a.total_rows,
                    "largest_partition_bytes": a.largest_partition_bytes,
                }
                for a in self.aggregator_estimates
            ],
            "worst_case_aggregator": {
                "aggregator_id": self.worst_case_aggregator.aggregator_id,
                "partition_ids": self.worst_case_aggregator.partition_ids,
                "total_bytes": self.worst_case_aggregator.total_bytes,
                "total_rows": self.worst_case_aggregator.total_rows,
                "largest_partition_bytes": self.worst_case_aggregator.largest_partition_bytes,
            },
            # Memory requirements
            "aggregator_heap_memory_bytes": self.aggregator_heap_memory_bytes,
            "aggregator_input_object_store_bytes": self.aggregator_input_object_store_bytes,
            "aggregator_output_object_store_bytes": self.aggregator_output_object_store_bytes,
            "required_memory_per_aggregator": self.required_memory_per_aggregator,
            "buffer_memory_bytes": self.buffer_memory_bytes,
            "recommended_memory_per_aggregator": self.recommended_memory_per_aggregator,
            # Cluster comparison
            "cluster_memory_available": self.cluster_memory_available,
            "total_required_memory": self.total_required_memory,
            "memory_headroom_ratio": self.memory_headroom_ratio,
            # Configuration
            "recommended_ray_remote_args": self.recommended_ray_remote_args,
            # Join-specific
            "left_bytes_per_partition": self.left_bytes_per_partition,
            "right_bytes_per_partition": self.right_bytes_per_partition,
            "estimated_output_size_bytes": self.estimated_output_size_bytes,
            "is_join": self.is_join,
        }

    def __repr__(self) -> str:
        """Return a brief representation of the estimate."""
        return (
            f"ShuffleMemoryEstimate("
            f"partitions={self.num_partitions}, "
            f"aggregators={self.num_aggregators}, "
            f"recommended_memory={self._format_bytes(self.recommended_memory_per_aggregator)}, "
            f"skew_factor={self.skew_factor:.2f}x"
            f")"
        )
