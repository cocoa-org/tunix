"""SWE-Gym filtered data module for nanorollout agentic GRPO (Phase X).

Migration of examples/tinyflow_swe/data/swegym_filtered_data.py to use
NanoRollout symbols (api_mapping.md §1 #7-8). Loads HF SWE-Gym/SWE-Gym,
filters by jsonl of instance_ids, augments with docker_image, returns a
`grain.MapDataset` ready for `AgenticGrpoPipeline.post_init_dataset`.
"""

import json
import os
from typing import Optional

import grain
import numpy as np

# NEW: nanorollout imports (api_mapping.md §1, symbols #7-8).
# These live under nanorollout.harness.runner.swe.common (moved from
# harness.utils.artifacts). No `__init__` re-export — must import from
# `common` directly.
from nanorollout.harness.runner.swe.common import (
    NamingStrategy,
    get_swebench_docker_image_name,
)


def _read_filter_ids(filter_jsonl_path: str) -> set[str]:
    """Parse jsonl and return set of metadata.instance_id values."""
    if not os.path.isfile(filter_jsonl_path):
        raise FileNotFoundError(f"filter_jsonl_path not found: {filter_jsonl_path}")
    wanted: set[str] = set()
    with open(filter_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{filter_jsonl_path}:{line_num} not valid JSON: {exc}"
                ) from exc
            iid = rec.get("metadata", {}).get("instance_id")
            if not iid:
                raise ValueError(
                    f"{filter_jsonl_path}:{line_num} missing metadata.instance_id"
                )
            wanted.add(iid)
    if not wanted:
        raise ValueError(f"{filter_jsonl_path} contained zero instance_ids")
    return wanted


def create_dataset(
    dataset_name: str = "SWE-Gym/SWE-Gym",
    dataset_split: str = "train",
    filter_jsonl_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> grain.MapDataset:
    """Load SWE-Gym, filter by jsonl instance_ids, augment with docker_image.

    Args:
        dataset_name: HF dataset id. Defaults to `SWE-Gym/SWE-Gym`.
        dataset_split: HF split. SWE-Gym only ships `train`.
        filter_jsonl_path: Path to jsonl of `{"metadata": {"instance_id": ...}}`.
        cache_dir: HF datasets cache. Defaults to
            `/mnt/disks/tunix-data/dataset_cache` (project's persistent disk).
        shuffle: Shuffle filtered split before returning.
        seed: Shuffle seed.

    Returns:
        A `grain.MapDataset` whose elements are SWE-Gym entries with
        `docker_image` injected by `get_swebench_docker_image_name` with
        `naming_strategy=SWE_GYM`, and list fields JSON-serialised to strings.
    """
    if filter_jsonl_path is None:
        raise ValueError("filter_jsonl_path is required")

    wanted = _read_filter_ids(filter_jsonl_path)

    if cache_dir is None:
        cache_dir = "/mnt/disks/tunix-data/dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)

    from datasets import load_dataset  # pylint: disable=g-import-not-at-top
    full = load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    filtered = full.filter(lambda ex: ex["instance_id"] in wanted)
    if len(filtered) != len(wanted):
        missing = wanted - {ex["instance_id"] for ex in filtered}
        raise ValueError(
            f"Filtered size {len(filtered)} != wanted {len(wanted)}; "
            f"missing {len(missing)} instance_ids (sample: {list(missing)[:5]})"
        )

    def _transform(entry):
        # 1. Augment with docker_image (NanoRolloutGCEEnv requires this).
        entry["docker_image"] = get_swebench_docker_image_name(
            instance=entry,
            env_class="docker",
            naming_strategy=NamingStrategy.SWE_GYM,
        )
        # 2. JSON-serialise list fields so grain default batcher can stack them.
        for k, v in list(entry.items()):
            if isinstance(v, list):
                entry[k] = json.dumps(v)
        return entry

    filtered = filtered.map(_transform, keep_in_memory=True)

    if shuffle:
        filtered = filtered.shuffle(seed)

    return grain.MapDataset.source(filtered)


# Re-export deepswe batch_fn: SWE-Gym fields not in deepswe's _STR_KEYS /
# _DICT_KEYS / _ARRAY_KEYS (instance_id, repo, base_commit, version, patch,
# test_patch, hints_text, created_at) fall through deepswe.batch_fn's `else`
# branch and become plain lists, which the grain default batcher handles.
# `problem_statement` is in _ARRAY_KEYS → np.array, then _unpack_entry in
# nanorollout_env reduces to scalar. `docker_image` is in _STR_KEYS → list-of-str,
# then _unpack_entry picks v[0].
from examples.deepswe.deepswe_data import batch_fn  # noqa: E402,F401  pylint: disable=g-import-not-at-top
