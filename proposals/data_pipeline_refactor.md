# Pretraining Data Pipeline Analysis & Refactoring Proposal

## 1. Current Architecture Analysis

### 1.1 Components & Flow

The current data pipeline consists of three main stages:

1.  **Collection (`src/modes/collect.py` & `src/data_collector.py`)**:
    *   **Process**: Multiple worker processes run simulations.
    *   **Buffering**: `DataCollector` accumulates episodes in memory (list of `EpisodeData`).
    *   **Checkpoints**: When `save_frequency` is reached, `DataCollector` consolidates the buffered episodes into large tensors (Shape: `[TotalTimeSteps, ...]`) and dumps them as a `data_checkpoint_X.pkl`.
    *   **Format**: Pickle containing a dictionary with aggregated tensors for Team 0 and Team 1.

2.  **Aggregation (`src/modes/collect.py` -> `aggregate_worker_data`)**:
    *   **Process**: After all workers finish, the main process scans for all `data_checkpoint_*.pkl` files.
    *   **Loading**: It loads **every single checkpoint** into memory simultaneously.
    *   **Merging**: It concatenates all tensors into one massive dictionary.
    *   **Saving**: It dumps the result to `aggregated_data.pkl` (and `metadata.yaml`).

3.  **Loading & Preprocessing (`src/train/data_loader.py` & `src/train/unified_dataset.py`)**:
    *   **Loading**: `load_bc_data` reads the entire `aggregated_data.pkl` into RAM.
    *   **Preprocessing**: `_compute_discounted_returns` iterates over all episodes to calculate returns. This happens *every time* the training script starts.
    *   **Dataset**: `UnifiedEpisodeDataset` wraps the in-memory tensors. `ShortView` and `LongView` index into this dataset.

### 1.2 Data Format

**Current Format (`aggregated_data.pkl`)**:
```python
{
    "team_0": {
        "tokens": Tensor(TotalSteps, MaxShips, TokenDim),
        "actions": Tensor(TotalSteps, MaxShips, NumActions),
        "rewards": Tensor(TotalSteps,),
        # ...
    },
    "team_1": { ... },
    "episode_ids": Tensor(TotalSteps,),
    "episode_lengths": Tensor(NumEpisodes,)
}
```

### 1.3 Critical Pain Points

1.  **Memory Bottleneck (Aggregation)**: The aggregation step loads all checkpoints into RAM before concatenation. For large datasets (e.g., >10M steps), this will cause an Out-Of-Memory (OOM) crash.
2.  **Memory Bottleneck (Training)**: The trainer must load the entire dataset into RAM. This limits the dataset size to available physical RAM.
3.  **Startup Latency**: Calculating discounted returns on every startup is redundant and slow for large datasets.
4.  **Inefficient Storage**: Python's `pickle` is not optimized for large numerical arrays and allows no partial reading (random access).

---

## 2. Decision: Aggregation Format HDF5 vs Zarr

We compared **HDF5** and **Zarr** for this pipeline:

| Feature | HDF5 (`h5py`) | Zarr |
| :--- | :--- | :--- |
| **Structure** | Single file (`.h5`) | Directory of chunk files |
| **Parallel Writes** | Difficult (Global Lock / SWMR) | Excellent (File-level locking) |
| **Parallel Reads** | Good (with SWMR / Careful usage) | Excellent |
| **Ecosystem** | Standard in Scientific Computing / DL | Standard in Cloud / Big Data |
| **Local Usage** | Clean (1 file to move) | Messy (100k+ files on disk) |

**Verdict**: We will use **HDF5**.
*   **Reasoning**: Since we are primarily running locally, managing a Zarr directory with 100,000+ tiny chunk files can severely impact file system performance (NTFS/ext4) and makes moving datasets (e.g., via `scp` or USB) painful. HDF5 offers a single-file abstraction that is easier to manage personally.
*   **Concurrency**: Our *Aggregation* step is serial (or can be made serial-streaming), and our *Training* step involves multiple workers reading read-only. HDF5 handles multiple readers fine (with `swmr` mode or just careful file opening).

---

## 3. Proposal: Unified HDF5 Pipeline

### 3.1 Streaming Aggregation
Instead of loading all data at once:
1.  Create a target `aggregated_data.h5` file with **resizable datasets** (using `maxshape`).
2.  Iterate through worker checkpoints ONE BY ONE.
3.  For each checkpoint:
    *   Load it.
    *   **Discard Team 2**: As per user request, we completely ignore `team_1` data keys.
    *   **Merge & Normalize**: Flatten the `team_0` data into normalized keys (`tokens`, `actions`, etc.).
    *   **Precompute**: Compute Discounted Returns / GAE *now*.
    *   Append all tensors to the HDF5 datasets.
    *   Discard checkpoint data to free RAM.
4.  This reduces Aggregation peak memory from $O(TotalData)$ to $O(CheckpointSize)$.

### 3.2 Data Structure Simplification (Removing Team 2)
We will **remove the `team_0` / `team_1` distinction**. The logic is as follows:
*   We only save `team_0` data from the collector.
*   The final HDF5 file will look like a standard single-agent dataset (tokens, actions, etc.).
*   **Benefit**: This halves the size of the checkpoints and the final dataset, and removes all complexity related to team keys in the `UnifiedEpisodeDataset`.

**New Simplified HDF5 Schema (`dataset.h5`)**:
```
/
├── tokens       (Shape: N_total x S x D)  [Chunks: 1024 x S x D]
├── actions      (Shape: N_total x S x A)
├── rewards      (Shape: N_total)
├── returns      (Shape: N_total)           <-- Precomputed!
├── term_mask    (Shape: N_total)           <-- Boolean mask for end-of-ep
├── meta/
│   ├── episode_lengths (Shape: E_total)
│   └── episode_ids     (Shape: N_total)
```
*Note: `N_total` will be `TotalTimeSteps` (just Team 0).*

### 3.3 Zero-Copy Loading
In `UnifiedEpisodeDataset`:
1.  Open the HDF5 file in read mode.
2.  Wrap the HDF5 datasets in a thin wrapper that behaves like a Tensor.
3.  Use **Memory Mapping** (or direct slicing) so that data is only fetched from disk when a batch is requested.
4.  Remove the on-the-fly return calculation.

---

## 4. Implementation Plan

### Step 1: Dependencies
Add `h5py` to the project dependencies.

### Step 2: Refactor `collect.py` (Aggregation)
Rewrite `aggregate_worker_data` to perform streaming aggregation + merging + precomputation.

### Step 3: Refactor `DataLoader` & `UnifiedEpisodeDataset`
Update `UnifiedEpisodeDataset` to accept a path to an `.h5` file.
*   Implement `__getitem__` by slicing the HDF5 object.
*   Ensure thread-safety for `num_workers > 0`.

### Step 4: Verification
Verify that the training loop receives data identically to before (same shapes, correct return values).
