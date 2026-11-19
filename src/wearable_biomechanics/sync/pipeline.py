"""End-to-end and batch orchestration pipelines for temporal synchronization."""

from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from wearable_biomechanics.sync.temporal import (
    calculate_front_to_raw_sync_lag,
    calculate_sync_lag,
)
from wearable_biomechanics.sync.video import (
    analyze_keypoints,
    ensure_30fps_video,
    generate_annotated_video,
    generate_synced_video,
)

_store_lock = threading.Lock()


def visual_estimation_pipeline(
    entry_id: str,
    store,
    generate_annotated: bool = True,
    generate_synced: bool = True,
    plot: bool = True,
    work_dir: str = "/content",
):
    """Run keypoint extraction, camera sync lag, and video generation for one experiment."""
    entry_type = "experiments"
    exp_folder = os.path.join(work_dir, entry_id)
    os.makedirs(exp_folder, exist_ok=True)
    print(f"Setting up directory: {exp_folder}")

    def safe_download(label: str):
        try:
            path = store.download_related_file(
                entry_type=entry_type,
                entry_id=entry_id,
                label=label,
                dest_folder=exp_folder,
            )
            if path and os.path.exists(path):
                return path
        except Exception as e:
            print(f"ℹ {label} not available for {entry_id} ({e})")
        return None

    front_video_path = safe_download("front_video")
    side_video_path = safe_download("side_video")
    has_front = front_video_path is not None
    has_side = side_video_path is not None

    if not has_front and not has_side:
        print(f"Error: Neither front_video nor side_video found for {entry_id}.")
        return None, None, None, None, None, None

    for label, path in [("front_video", front_video_path), ("side_video", side_video_path)]:
        if not path:
            continue
        resampled_path, was_resampled = ensure_30fps_video(path, target_fps=30.0)
        if was_resampled:
            with _store_lock:
                try:
                    store.add_related_file(
                        entry_type=entry_type,
                        entry_id=entry_id,
                        label=label,
                        local_path=resampled_path,
                    )
                    print(f"✓ Updated 30fps {label} in OpenDatasetStore for {entry_id}")
                except Exception as e:
                    print(f"⚠ Could not overwrite resampled {label} in store: {e}")

    print("\n--- Analyzing Keypoints ---")
    front_df = analyze_keypoints(front_video_path, batch_size=32) if has_front else None
    side_df = analyze_keypoints(side_video_path, batch_size=32) if has_side else None

    side_lag = None
    if has_front and has_side:
        print("\n--- Calculating Sync Lag ---")
        side_lag = calculate_sync_lag(front_df, side_df, plot=plot)
        print(f"Calculated side lag: {side_lag} frames")
    else:
        missing_cam = "side_video" if has_front else "front_video"
        print(f"\n--- Skipping Sync Lag Calculation ({missing_cam} is missing) ---")

    front_annotated = None
    side_annotated = None
    synced_video = None

    if generate_annotated:
        print("\n--- Generating Annotated Video(s) ---")
        with ThreadPoolExecutor(max_workers=2) as render_pool:
            f_front = (
                render_pool.submit(generate_annotated_video, front_video_path, front_df)
                if has_front
                else None
            )
            f_side = (
                render_pool.submit(generate_annotated_video, side_video_path, side_df)
                if has_side
                else None
            )
            if f_front:
                front_annotated = f_front.result()
            if f_side:
                side_annotated = f_side.result()

    if generate_synced:
        print("\n--- Generating Synchronized Side-by-Side Video (FFmpeg Ultrafast) ---")
        synced_video = generate_synced_video(front_video_path, side_video_path, side_lag)

    print("\nPipeline complete!")
    return front_df, side_df, side_lag, front_annotated, side_annotated, synced_video


def save_visual_estimation_results(
    entry_id: str,
    store,
    front_df,
    side_df,
    side_lag,
    front_annotated_path: str | None = None,
    side_annotated_path: str | None = None,
    synced_video_path: str | None = None,
) -> None:
    """Thread-safe saving of visual estimation outputs back to OpenDatasetStore."""
    entry_type = "experiments"
    with _store_lock:
        print(f"--- Saving results for {entry_id} to OpenDatasetStore ---")
        if side_lag is not None:
            store.edit_entry(
                entry_type=entry_type,
                entry_id=entry_id,
                **{"video_sync_side_lag_frames": int(side_lag)},
            )
            print(f"✓ Saved metadata: video_sync_side_lag_frames = {int(side_lag)}")

        if front_df is not None and not front_df.empty:
            store.add_processed_data(
                entry_type=entry_type,
                entry_id=entry_id,
                tag="front_video_keypoints",
                df=front_df,
            )
            print("✓ Saved front_video_keypoints dataframe")

        if side_df is not None and not side_df.empty:
            store.add_processed_data(
                entry_type=entry_type,
                entry_id=entry_id,
                tag="side_video_keypoints",
                df=side_df,
            )
            print("✓ Saved side_video_keypoints dataframe")

        for file_path, label in [
            (synced_video_path, "synced_video"),
            (front_annotated_path, "front_annotated_video"),
            (side_annotated_path, "side_annotated_video"),
        ]:
            if file_path and os.path.exists(file_path):
                try:
                    store.add_related_file(
                        entry_type=entry_type,
                        entry_id=entry_id,
                        label=label,
                        local_path=file_path,
                    )
                    print(f"✓ Saved video file for label: {label}")
                except Exception as e:
                    print(f"⚠ Error saving {label}: {e}")

        print("--- Save Complete! ---")


def run_single_experiment(
    entry_id: str,
    store,
    generate_annotated: bool = True,
    generate_synced: bool = True,
    plot: bool = False,
    cleanup_scratch: bool = True,
    work_dir: str = "/content",
) -> tuple[str, bool]:
    """Process and save a single experiment end-to-end."""
    print(f"\n[START] Pipeline started for: {entry_id}")
    t0 = time.time()

    results = visual_estimation_pipeline(
        entry_id=entry_id,
        store=store,
        generate_annotated=generate_annotated,
        generate_synced=generate_synced,
        plot=plot,
        work_dir=work_dir,
    )
    if not results or (results[0] is None and results[1] is None):
        print(f"\n[FAIL] Processing failed for: {entry_id} (no valid videos found)")
        return entry_id, False

    front_df, side_df, side_lag, front_ann, side_ann, synced = results
    save_visual_estimation_results(
        entry_id=entry_id,
        store=store,
        front_df=front_df,
        side_df=side_df,
        side_lag=side_lag,
        front_annotated_path=front_ann,
        side_annotated_path=side_ann,
        synced_video_path=synced,
    )

    if cleanup_scratch:
        exp_folder = os.path.join(work_dir, entry_id)
        if os.path.exists(exp_folder):
            shutil.rmtree(exp_folder, ignore_errors=True)

    print(f"\n[DONE] Finished & saved {entry_id} in {time.time() - t0:.1f}s")
    return entry_id, True


def run_batch_visual_estimation(
    experiments_to_run: list[str],
    store,
    max_workers: int = 3,
    plot: bool = False,
) -> None:
    """Run `run_single_experiment` in parallel across a list of experiment IDs."""
    print(f"Launching batch execution with {max_workers} parallel workers...")
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_experiment, exp_id, store, True, True, plot): exp_id
            for exp_id in experiments_to_run
        }
        for future in as_completed(futures):
            exp_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Exception during processing of {exp_id}: {e}")

    print(f"\n{'=' * 50}\nAll experiments completed in {time.time() - start_all:.1f}s\n{'=' * 50}")


def sync_front_to_raw_experiment(entry_id: str, store, plot: bool = True) -> dict:
    """Download keypoints/video/raw IMU for one experiment, compute lag, and save metadata."""
    raw_df = store.get_entry_raw_data(entry_type="experiments", entry_id=entry_id)
    front_df = store.get_entry_processed_data(
        entry_type="experiments", entry_id=entry_id, tag="front_video_keypoints"
    )
    front_video_path = store.download_related_file(
        entry_type="experiments",
        entry_id=entry_id,
        label="front_video",
        dest_folder="/tmp",
    )

    res = calculate_front_to_raw_sync_lag(
        front_df,
        raw_df,
        front_video_path=front_video_path,
        plot=plot,
    )

    metadata_payload = {
        "video_sync_front_delta_t_frames": int(res["delta_t_frames"]),
        "video_sync_front_delta_t_imu_samples": int(res["delta_t_imu_samples"]),
        "video_sync_front_delta_t_sec": float(res["delta_t_sec"]),
    }
    store.edit_entry(entry_type="experiments", entry_id=entry_id, **metadata_payload)
    print(f"✓ Saved metadata for {entry_id}: {metadata_payload}")
    return res


def run_batch_front_to_raw_sync(
    start_idx: int, end_idx: int, store, plot: bool = True
) -> None:
    """Run front-to-raw IMU synchronization across `experiments_{i:04d}`."""
    for i in range(start_idx, end_idx + 1):
        entry_id = f"experiments_{i:04d}"
        print(f"\n[{i}/{end_idx}] Starting front-to-raw sync for {entry_id}...")
        try:
            sync_front_to_raw_experiment(entry_id, store, plot=plot)
        except Exception as e:
            print(f"Error processing {entry_id}: {e}")

