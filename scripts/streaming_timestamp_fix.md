# Streaming Timestamp Fix

## Dataset Format

### Video Timestamps (`from_timestamp`, `to_timestamp`)

These are **absolute seek positions** within each video file. 

When multiple episodes are concatenated into a single video file, timestamps accumulate. When a new video file starts (`file_index` changes), timestamps **reset to 0**:

| Video File | file_index | Episode | from_ts | to_ts | Notes |
|------------|------------|---------|---------|-------|-------|
| `file-000.mp4` | 0 | 0 | 0.00 | 10.70 | |
| `file-000.mp4` | 0 | 1 | 10.70 | 24.90 | |
| `file-000.mp4` | 0 | ... | ... | ... | |
| `file-000.mp4` | 0 | 207 | 1402.40 | 1414.05 | |
| `file-001.mp4` | 1 | 208 | **0.00** | 12.50 | ← RESETS to 0 for new file |
| `file-001.mp4` | 1 | 209 | 12.50 | 20.95 | |
| `file-001.mp4` | 1 | ... | ... | ... | |
| `file-001.mp4` | 1 | 318 | 1448.85 | 1463.35 | |

Different video keys can split at different episodes, so the same episode may have different `from_timestamp` values per video key.

### Parquet Fields (`index`, `frame_index`, `timestamp`)

| Field | Scope | Resets? | Example for Episode 318, Frame 0 |
|-------|-------|---------|----------------------------------|
| `index` | **Global** across all parquets | Never | 85429 |
| `frame_index` | Per episode | Each episode | 0 |
| `timestamp` | Per episode | Each episode | 0.0 |

```
data/chunk-000/file-000.parquet
├── index: 0 - 85428
├── episode_index: 0 - 317
└── Episodes 0-317 data

data/chunk-000/file-001.parquet
├── index: 85429 - 101468    ← CONTINUES from file-000, never resets
├── episode_index: 318 - 378
└── Episodes 318-378 data
```

---

## Why Local Reading Was NOT Impacted

The local [`LeRobotDataset.__getitem__()`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py#L1048-L1078) works correctly due to **three key differences** from streaming:

### Difference 1: Uses `item["timestamp"]` instead of `item["index"] / fps`

```python
# LeRobotDataset.__getitem__()
current_ts = item["timestamp"].item()  # ← Episode-relative: 0.0, 0.05, ...
# NOT: item["index"] / self.fps       # ← Would give meaningless 4271s
```

### Difference 2: Clamps **indices** (not timestamps) in `_get_query_indices()`

```python
# LeRobotDataset._get_query_indices()
# https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py#L938-L958
ep_start = ep["dataset_from_index"]
ep_end = ep["dataset_to_index"]
query_indices = {
    key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]  # ← Clamps indices
    for key, delta_idx in self.delta_indices.items()
}
```

This clamps to valid episode indices first, then looks up actual timestamps from those indices. No timestamp clamping needed.

### Difference 3: `_query_videos()` adds `from_timestamp`

```python
# LeRobotDataset._query_videos()
# https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py#L1014-L1027
from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
shifted_query_ts = [from_timestamp + ts for ts in query_ts]  # ← Converts relative → absolute
```

---

## Upstream Streaming Implementation (Buggy)

The [upstream streaming code](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/streaming_dataset.py) has two bugs compared to local reading:

1. **Uses `item["index"] / fps`** instead of `item["timestamp"]` - gives meaningless global time
2. **Clamps to absolute bounds** - which constrains the meaningless value but creates coordinate mismatch

### Step 1: `make_frame()` - BUG: Uses wrong timestamp source

```python
current_ts = item["index"] / self.fps  # ← BUG: Global index gives meaningless 4271s

episode_boundaries_ts = {
    key: (
        self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],  # absolute w.r.t current video
        self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],    # absolute w.r.t current video
    )
    for key in self.meta.video_keys
}

# Pass to _get_query_timestamps
query_timestamps = self._get_query_timestamps(current_ts, self.delta_indices, episode_boundaries_ts)
```

### Step 2: `_get_query_timestamps()` - Clamp to absolute boundaries

```python
def _get_query_timestamps(self, current_ts, query_indices, episode_boundaries_ts):
    for key in self.meta.video_keys:
        timestamps = [current_ts + delta/fps for delta in query_indices[key]]
        
        # Clamping constrains the meaningless value (4271s) to episode bounds
        query_timestamps[key] = torch.clamp(
            torch.tensor(timestamps), 
            *episode_boundaries_ts[key]  # Clamps 4271s → 1463.35s (max bound)
        ).tolist()
    
    return query_timestamps
```

### Step 3: `_query_videos()` - BUG: No offset added

```python
def _query_videos(self, query_timestamps, ep_idx):
    for video_key, query_ts in query_timestamps.items():
        # ← BUG: Passes clamped value directly without adding from_timestamp
        frames = decode_video_frames_torchcodec(video_path, query_ts, ...)
```

**Summary of bugs:**
1. **Step 1:** Uses `item["index"] / fps` (meaningless global time) instead of `item["timestamp"]` (episode-relative)
2. **Step 3:** Does not add `from_timestamp` offset in `_query_videos()`

## Proposed Fix

Make streaming behave like local reading by applying the same two fixes:

### Fix 1: Use `item["timestamp"]` instead of `item["index"] / fps`

```python
# StreamingLeRobotDataset.make_frame()
current_ts = item["timestamp"]  # ← Episode-relative: 0.0, 0.05, ...
# NOT: item["index"] / self.fps  # ← Would give meaningless 4271s
```

Also update `episode_boundaries_ts` to use relative bounds (0 to duration):

```python
episode_boundaries_ts = {
    key: (
        0.0,  # Relative start
        to_timestamp - from_timestamp,  # Duration
    )
    for key in self.meta.video_keys
}
```

### Fix 2: `_query_videos()` adds `from_timestamp` (like local)

```python
# StreamingLeRobotDataset._query_videos()
from_timestamp = ep[f"videos/{video_key}/from_timestamp"]
shifted_query_ts = [from_timestamp + ts for ts in query_ts]  # ← Converts relative → absolute
```

**Result:** For Episode 408 (libero_90), `wrist_image` key:
- OLD: clamp(0, (2932.70, 2938.30)) = 2932.70 → seek **5865.40s** (double-add!)
- NEW: clamp(0, (0, 5.60)) = 0.00 → seek **2932.70s** ✓

---

## Comparison

| Aspect | Local (works) | Upstream Streaming (buggy) | Proposed Fix |
|--------|---------------|---------------------------|--------------|
| `current_ts` | `item["timestamp"]` | `item["index"] / fps` | `item["timestamp"]` |
| Clamping | None | Absolute bounds | Relative bounds |
| `_query_videos` | Adds `from_timestamp` | No offset | Adds `from_timestamp` |

---

## Concrete Example: Episode 318, Frame 0

This walkthrough shows exactly what happens at each step for `observation.images.image` in episode 318.

### Data from Files

**From `data/chunk-000/file-001.parquet`:**
```
episode_index: 318
frame_index: 0
index: 85429          ← Global frame counter (never resets)
timestamp: 0.0        ← Episode-relative (resets each episode)
```

**From `meta/episodes/chunk-000/file-001.parquet`:**
```
videos/observation.images.image/file_index: 1
videos/observation.images.image/from_timestamp: 1448.85   ← Seek position in file-001.mp4
videos/observation.images.image/to_timestamp: 1463.35
```

**Video file:** `videos/chunk-000/observation.images.image/file-001.mp4`

---

### Upstream (BUGGY) Flow

```
Step 1: make_frame() computes current_ts
─────────────────────────────────────────
current_ts = item["index"] / fps
           = 85429 / 20
           = 4271.45 seconds    ← Meaningless! No relation to video file.

episode_boundaries_ts = (1448.85, 1463.35)   ← Absolute bounds

Step 2: _get_query_timestamps() clamps
──────────────────────────────────────
clamp(4271.45, min=1448.85, max=1463.35) = 1463.35   ← Clamped to max bound

query_timestamps = [1463.35]

Step 3: _query_videos() - NO offset added
─────────────────────────────────────────
seek_position = 1463.35 seconds   ← Seeks to END of episode (last frame)!

Result: WRONG FRAME - returns last frame instead of first frame
```

---

### Fixed Flow

```
Step 1: make_frame() computes current_ts
─────────────────────────────────────────
current_ts = item["timestamp"]
           = 0.0 seconds        ← Episode-relative, correct!

episode_boundaries_ts = (0.0, 14.5)   ← Relative bounds (duration = 1463.35 - 1448.85)

Step 2: _get_query_timestamps() clamps
──────────────────────────────────────
clamp(0.0, min=0.0, max=14.5) = 0.0   ← Already valid, unchanged

query_timestamps = [0.0]

Step 3: _query_videos() adds from_timestamp
───────────────────────────────────────────
from_timestamp = 1448.85
seek_position = 1448.85 + 0.0 = 1448.85 seconds   ← Correct position in video!

Result: CORRECT FRAME - seeks to start of episode 318 in file-001.mp4
```

---

### Visual Summary

```
                    Episode 318 in file-001.mp4
                    ┌─────────────────────────────────┐
                    │  from_ts=1448.85   to_ts=1463.35│
                    │     ↓                     ↓     │
Video timeline: ════╪═════════════════════════════════╪════
                    │← frame 0              frame 289→│
                    └─────────────────────────────────┘

UPSTREAM (buggy):
  current_ts = 85429/20 = 4271s → clamp to 1463.35 → seek 1463.35
                                                          ↓
                                                    (last frame!)

FIXED:
  current_ts = 0.0 → clamp to 0.0 → add 1448.85 → seek 1448.85
                                                       ↓
                                                 (first frame!)
```

---

## Affected Datasets

| Dataset | Episodes | Mismatched File Boundaries | Large Offsets (>100s) |
|---------|----------|---------------------------|----------------------|
| `libero_10_no_noops_1.0.0_lerobot_aligned` | 379 | 110 | 371 |
| `libero_90_no_noops_lerobot` | 3921 | 3233 | 3906 |

**Example:** Episode 408 in `libero_90_no_noops_lerobot`:
- `observation.images.image`: file_index=1, from_ts=0.00 (just started new file)
- `observation.images.wrist_image`: file_index=0, from_ts=2932.70 (still in old file)
