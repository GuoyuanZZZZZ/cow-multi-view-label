#!/usr/bin/env python3
"""Utilities to normalize calibration input from either videos or images.

This script prepares a single image list from mixed inputs:
- image file
- image directory
- video file (sampled every N frames)

It is designed as a lightweight bridge for calibration tools that expect
image sequences.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def collect_images_from_dir(directory: Path) -> List[Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and is_image(p))
    if not files:
        raise ValueError(f"No images found in directory: {directory}")
    return files


def extract_frames_from_video(video_path: Path, output_dir: Path, step: int) -> List[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved_idx = 0
    saved_paths: List[Path] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            out = output_dir / f"{video_path.stem}_f{frame_idx:06d}.png"
            cv2.imwrite(str(out), frame)
            saved_paths.append(out)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    if not saved_paths:
        raise ValueError(f"No frames extracted from video: {video_path}")

    print(f"Extracted {saved_idx} frames from {video_path.name} -> {output_dir}")
    return saved_paths


def prepare_inputs(inputs: Iterable[Path], output_dir: Path, video_step: int) -> List[Path]:
    images: List[Path] = []

    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input does not exist: {p}")

        if p.is_dir():
            images.extend(collect_images_from_dir(p))
            continue

        if is_image(p):
            images.append(p)
            continue

        if is_video(p):
            video_output_dir = output_dir / p.stem
            images.extend(extract_frames_from_video(p, video_output_dir, video_step))
            continue

        raise ValueError(
            f"Unsupported input type: {p}. Supported: image file, image directory, video file"
        )

    return sorted(images)


def write_manifest(images: List[Path], output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as f:
        for p in images:
            f.write(str(p.resolve()) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare calibration inputs from image/video paths."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files, image folders, or video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_frames"),
        help="Where extracted video frames are written.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("prepared_frames/manifest.txt"),
        help="Output list file containing one image path per line.",
    )
    parser.add_argument(
        "--video-step",
        type=int,
        default=10,
        help="Sample every N frames from each video.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.video_step <= 0:
        raise ValueError("--video-step must be > 0")

    input_paths = [Path(x) for x in args.inputs]
    images = prepare_inputs(input_paths, args.output_dir, args.video_step)
    write_manifest(images, args.manifest)

    print(f"Prepared {len(images)} calibration images")
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
