.. _shuffle-memory-estimation:

Shuffle Memory Estimation
=========================

This page explains how Ray Data estimates and manages memory for shuffle operations
(repartition, groupby, and join). Understanding this system helps you configure
shuffle operations to avoid out-of-memory errors and optimize performance.

Overview
--------

Shuffle operations redistribute data across partitions based on key columns. This
requires buffering data in memory, which can lead to OOM errors if not properly
managed. Ray Data includes an automatic memory estimation system that:

1. **Estimates partition sizes** before execution by scanning input data
2. **Allocates per-worker memory** based on actual partition assignments
3. **Validates estimates** against actual usage (optional strict mode)
4. **Adjusts heap memory** for operations that require in-memory processing

Memory Components
-----------------

Each shuffle aggregator (worker) requires memory for three components:

**Object Store Memory (Input)**
    Buffered input data waiting to be processed. Size depends on how many
    partitions are assigned to this aggregator.

**Heap Memory**
    Memory for in-memory operations like join hash tables or aggregation
    state. This varies significantly by operation type.

**Object Store Memory (Output)**
    Buffered output blocks waiting to be consumed by downstream operators.

The total memory requirement per aggregator is::

    total = input_object_store + heap + output_object_store + buffer

Where ``buffer`` is a safety margin (default 15%) for fragmentation and
temporary allocations.

Estimation Methods
------------------

Partition Size Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

Ray Data estimates partition sizes by scanning all input blocks and computing
the hash partition assignment for each row. This uses the same ``_hash_partition()``
function as the actual shuffle, ensuring accurate predictions.

.. code-block:: python

    # For each input block (in parallel):
    partition_ids = _hash_partition(block.select(key_columns), num_partitions)

    # Count rows and bytes per partition
    for pid in range(num_partitions):
        partition_sizes[pid] += count_rows_with_id(pid) * bytes_per_row

This full-scan approach (no sampling) ensures exact partition size predictions
for operations without map-side reduction.

Key Cardinality Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The system also tracks the number of distinct key values (cardinality), which
is used to estimate reduction ratios for aggregation operations. Large blocks
are sampled (up to 100K rows) to avoid memory issues during cardinality tracking.

Heap Memory Estimation
~~~~~~~~~~~~~~~~~~~~~~

Heap memory requirements depend on the operation type:

**Repartition**: No heap memory needed (just shuffling data)
    ``heap = 0``

**Groupby/Aggregation**: Memory for partial aggregation state
    ``heap = largest_partition_size * 1.0``  (100% of largest partition)

**Join**: Memory for join hash tables and intermediate results
    ``heap = largest_partition_size * 2.0``  (200% of largest partition)

Per-Aggregator Memory Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory is allocated per-aggregator based on actual partition assignments,
not worst-case estimates. This allows more efficient memory utilization:

.. code-block:: python

    for aggregator_id in range(num_aggregators):
        # Get partitions assigned to this aggregator (round-robin)
        assigned_partitions = [p for p in partitions
                               if p.partition_id % num_aggregators == aggregator_id]

        # Compute memory based on assigned partitions
        input_bytes = sum(p.size_bytes for p in assigned_partitions)
        largest_partition = max(p.size_bytes for p in assigned_partitions)
        heap_bytes = largest_partition * heap_multiplier
        output_bytes = largest_partition

        aggregator_memory[aggregator_id] = input_bytes + heap_bytes + output_bytes

Validation Modes
----------------

Non-Strict Mode (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~

In normal operation, estimates are compared against actual usage and
differences are logged but don't cause failures:

- **< 10% error**: Info log confirming accuracy
- **> 10% error**: Warning log about estimation discrepancy

Strict Mode
~~~~~~~~~~~

Enable strict validation for testing or debugging estimation accuracy:

.. code-block:: python

    from ray.data.context import DataContext
    ctx = DataContext.get_current()
    ctx.strict_shuffle_estimation = True

In strict mode, validation failures raise ``RuntimeError``.

Operation-Specific Validation
-----------------------------

Repartition (Exact Match Expected)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For repartition operations, partition size estimates should exactly match
actual sizes because:

1. Estimation uses the same ``hash_partition()`` logic as actual shuffle
2. All rows are scanned (no sampling)
3. Arrow's ``nbytes`` provides consistent byte counting

Any non-zero error indicates a bug in the estimation logic.

**Validation**: ``actual == estimated`` (0% error tolerance)

Groupby/Aggregation (Upper Bound)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregation operators have **map-side partial aggregation** that reduces data
before the shuffle. This means actual partition sizes will be much smaller than
estimated (which is based on input data).

**Key insight**: The reduction ratio depends on key cardinality::

    reduction_ratio ≈ num_distinct_keys / num_rows

For example, grouping 1 billion rows into 50K groups:

- Input: 100 GB
- Expected shuffle: 100 GB × (50K / 1B) = 5 MB
- Reduction: 99.995%

**Validation**:

1. **Upper bound**: ``actual <= estimated * 1.05`` (5% tolerance)

   - Violation indicates a bug (actual should never exceed input-based estimate)

2. **Expected range**: ``actual ≈ estimated * (cardinality / rows)``

   - Large deviation may indicate cardinality sampling issues

Join (Shuffle Exact, Heap Variable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For join operations:

- **Shuffle partition sizes**: Should be exact (sum of left + right inputs)
- **Heap memory**: Harder to predict due to output size variability

Join output can "explode" when many rows share the same key::

    # If each key appears M times in left and N times in right:
    output_rows = num_keys * M * N

**Current limitation**: Heap estimation for joins uses a fixed 2x multiplier
based on largest input partition, which may be inaccurate for highly skewed
key distributions.

**Validation**: Exact match for shuffle partition sizes

Automatic Memory Configuration
------------------------------

When memory estimation is enabled, Ray Data automatically sets
``aggregator_ray_remote_args`` with the ``memory`` parameter:

.. code-block:: python

    # Automatically computed
    aggregator_ray_remote_args = {
        "memory": per_aggregator_memory_bytes
    }

This reserves the appropriate amount of memory for each aggregator actor.

You can override this with explicit configuration:

.. code-block:: python

    ds.repartition(
        num_blocks=100,
        shuffle=True,
        keys=["user_id"],
        aggregator_ray_remote_args={"memory": 2 * 1024**3},  # 2 GiB per worker
    )

Known Limitations and Caveats
-----------------------------

1. **Aggregation Estimates Are Upper Bounds**

   For groupby operations, estimates are based on input data size, not the
   reduced data after map-side aggregation. The actual shuffle will be much
   smaller. This is expected behavior, not a bug.

2. **Join Output Size Unpredictable**

   Join output can be much larger or smaller than inputs depending on key
   distribution and join type. The current 2x heap multiplier is a heuristic.

3. **max_concurrency > 1 Affects Heap Estimation**

   If ``max_concurrency > 1``, multiple partitions can be finalized
   concurrently, potentially requiring more heap memory. The default is
   ``max_concurrency=1`` for accurate estimation. A warning is logged if
   this is overridden.

4. **Cardinality Sampling for Large Blocks**

   Blocks larger than 100K rows are sampled for key cardinality tracking.
   This may underestimate cardinality for highly skewed distributions.

5. **Arrow Allocator Fragmentation**

   The Arrow memory allocator may retain memory after use, causing measured
   heap usage to exceed theoretical requirements. The 15% buffer helps
   accommodate this.

Environment Variables
---------------------

``RAY_DATA_DEFAULT_HASH_SHUFFLE_AGGREGATOR_MAX_CONCURRENCY``
    Maximum concurrent partition finalizations per aggregator.
    Default: 1 (for accurate heap estimation)

Configuration Options
---------------------

Via ``DataContext``:

.. code-block:: python

    from ray.data.context import DataContext
    ctx = DataContext.get_current()

    # Enable strict validation (fails on estimation errors)
    ctx.strict_shuffle_estimation = True

    # Set maximum number of shuffle aggregators
    ctx.max_hash_shuffle_aggregators = 256

Debugging Estimation Issues
---------------------------

If you encounter OOM errors or unexpected memory usage:

1. **Enable logging**: Check for estimation warnings in logs
2. **Enable strict mode**: Identify where estimates diverge from actual
3. **Check key cardinality**: High cardinality reduces aggregation efficiency
4. **Monitor skew**: Highly skewed partitions may exceed estimates
5. **Increase buffer**: If fragmentation is an issue, try a larger buffer ratio

Example output from estimation logging::

    INFO: Shuffle Memory Estimation
    Input: 1,000,000,000 rows, 36.8 GiB
    Output: 500 partitions across 128 workers
    Memory Allocation:
      Total reserved: 95.6 GiB for 128 workers
      Per worker: 600.4 MiB to 798.0 MiB

    INFO: Partition size estimation: predictions match actual sizes (max error 0.0%)
    INFO: Heap memory estimation: predicted 9824.8 MiB, actual 9361.8 MiB (error 5%)

Summary Table
-------------

.. list-table:: Memory Estimation by Operation Type
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Operation
     - Partition Estimate
     - Heap Estimate
     - Validation
     - Notes
   * - Repartition
     - Exact (scan-based)
     - 0
     - Exact match
     - Most predictable
   * - Groupby
     - Upper bound
     - 1x largest partition
     - Upper bound only
     - Map-side reduction
   * - Join
     - Exact (sum of inputs)
     - 2x largest partition
     - Exact for partitions
     - Output size varies
