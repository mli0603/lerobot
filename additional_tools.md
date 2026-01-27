# Streaming Optimization Tools

This document covers new utilities added to `lerobot.datasets.dataset_tools` for optimizing datasets for efficient streaming from remote storage (e.g., S3, Hugging Face Hub).

## New Functions

### `realign_parquets_to_videos()`

Rewrites parquet files so they match the video file structure **without re-encoding videos**. Parquet files are usually smaller than video files, resulting in fewer parquet shards than video shards. Since parquets determine the number of shards for streaming, fewer shards means less parallelization. This function splits parquets to match the video file structure, maximizing the number of shards for parallel streaming.

```python
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import realign_parquets_to_videos

# Load source dataset
dataset = LeRobotDataset(repo_id="my_dataset", root="/path/to/dataset")

# Realign parquets to match video file structure
aligned = realign_parquets_to_videos(
    dataset,
    output_dir="/path/to/output",
    copy_videos=True,  # Set to False if videos are already in place
)

print(f"Created aligned dataset at {aligned.root}")
print(f"  Episodes: {aligned.meta.total_episodes}")
print(f"  Frames: {aligned.meta.total_frames}")
```

After alignment, each episode's data file location (`data/chunk_index`, `data/file_index`) will match its video file location (`videos/{key}/chunk_index`, `videos/{key}/file_index`) and metadata file location (`meta/episodes/chunk_index`, `meta/episodes/file_index`).

Example script: `split_parquet.py`

### `shuffle_episodes()`

Physically reorders episodes in both parquet and video files. This **re-encodes videos** with episodes in the new shuffled order, ensuring optimal streaming performance where sequential episode reads map to sequential bytes in the files.

This is particularly useful for datasets with multiple tasks where episodes are grouped by task, and you want to interleave episodes from different tasks for training.

```python
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import shuffle_episodes

# Load source dataset
dataset = LeRobotDataset(repo_id="my_dataset", root="/path/to/dataset")

# Shuffle episodes with reproducible seed
shuffled = shuffle_episodes(
    dataset,
    output_dir="/path/to/output",
    seed=42,  # For reproducibility
)

print(f"Created shuffled dataset at {shuffled.root}")
```

**Note:** Video re-encoding can be time-consuming for large datasets. The function uses `libsvtav1` codec by default.

Example script: `split_parquet_and_shuffle.py`

### `update_info_with_data_files()`

Adds a `data_files` list to `info.json`. This allows specifying multiple parquet file shards, enabling parallel data loading across shards and avoiding directory scanning at runtime (useful for remote storage like S3 where directory listing operations can be slow).

```python
from lerobot.datasets.dataset_tools import update_info_with_data_files

# Update info.json with data_files list
data_files = update_info_with_data_files("/path/to/dataset")

print(f"Found {len(data_files)} parquet files:")
for f in data_files:
    print(f"  - {f}")
```

After running this, your `meta/info.json` will include:

```json
{
  "data_files": [
    "data/chunk-000/file-000.parquet",
    "data/chunk-000/file-001.parquet"
  ]
}
```

Example script: `update_parquet_info.py`

## Validation

After realigning or shuffling a dataset, validate the result:

```python
# 1. Check metadata counts match
assert aligned.meta.total_episodes == dataset.meta.total_episodes
assert aligned.meta.total_frames == dataset.meta.total_frames
assert aligned.meta.total_tasks == dataset.meta.total_tasks

# 2. Verify file alignment (data, video, meta have same indices)
first_video_key = aligned.meta.video_keys[0]
for ep_idx in range(min(20, aligned.meta.total_episodes)):
    data_chunk = aligned.meta.episodes[ep_idx]["data/chunk_index"]
    data_file = aligned.meta.episodes[ep_idx]["data/file_index"]
    video_chunk = aligned.meta.episodes[ep_idx][f"videos/{first_video_key}/chunk_index"]
    video_file = aligned.meta.episodes[ep_idx][f"videos/{first_video_key}/file_index"]
    meta_chunk = aligned.meta.episodes[ep_idx]["meta/episodes/chunk_index"]
    meta_file = aligned.meta.episodes[ep_idx]["meta/episodes/file_index"]
    
    assert (data_chunk, data_file) == (video_chunk, video_file)
    assert (data_chunk, data_file) == (meta_chunk, meta_file)

# 3. Sample frame data integrity
for idx in [0, aligned.meta.total_frames // 2, aligned.meta.total_frames - 1]:
    frame = aligned[idx]
    for video_key in aligned.meta.video_keys:
        img = frame[video_key]
        assert img.dim() == 3 and img.shape[0] == 3
        assert img.min() >= 0 and img.max() <= 1

print("All validation checks passed!")
```
