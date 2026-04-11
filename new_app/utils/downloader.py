#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video from a URL and save it as an MP4 file."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="downloads",
        help="Directory to save the downloaded file into (default: downloads)",
    )
    parser.add_argument(
        "-n",
        "--filename",
        default="%(title)s.%(ext)s",
        help="Output filename template (default: %%(title)s.%%(ext)s)",
    )
    return parser


def build_ydl_options(output_dir: Path, filename_template: str) -> dict[str, Any]:
    return {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / filename_template),
        "noplaylist": True,
        "quiet": False,
    }


def download_video(url: str, output_dir: Path, filename_template: str) -> Path:
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = build_ydl_options(output_dir, filename_template)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        final_path = Path(ydl.prepare_filename(info))
        if final_path.suffix.lower() != ".mp4":
            candidate = final_path.with_suffix(".mp4")
            if candidate.exists():
                final_path = candidate
        return final_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    saved_path = download_video(
        url=args.url,
        output_dir=Path(args.output_dir),
        filename_template=args.filename,
    )
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()
