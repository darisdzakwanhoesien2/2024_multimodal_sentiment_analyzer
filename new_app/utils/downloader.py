#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


AUTH_REQUIRED_SNIPPETS = (
    "sign in to confirm you're not a bot",
    "use --cookies-from-browser or --cookies for the authentication",
)

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
    parser.add_argument(
        "--cookies-from-browser",
        help="Browser name to read cookies from, e.g. chrome, firefox, safari",
    )
    parser.add_argument(
        "--cookies",
        help="Path to a cookies.txt file exported for YouTube authentication",
    )
    return parser


def build_ydl_options(
    output_dir: Path,
    filename_template: str,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / filename_template),
        "noplaylist": True,
        "quiet": False,
    }
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        options["cookiefile"] = str(cookies_file)
    return options


def _raise_friendly_error(exc: Exception) -> None:
    message = str(exc)
    lowered = message.lower()
    if all(snippet in lowered for snippet in AUTH_REQUIRED_SNIPPETS):
        raise RuntimeError(
            "YouTube is asking for an authenticated session. Pick a browser in "
            "`Use browser cookies`, or provide an exported cookies.txt file, then try again."
        ) from exc
    raise exc


def download_video(
    url: str,
    output_dir: Path,
    filename_template: str,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
) -> Path:
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = build_ydl_options(
        output_dir=output_dir,
        filename_template=filename_template,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_path = Path(ydl.prepare_filename(info))
            if final_path.suffix.lower() != ".mp4":
                candidate = final_path.with_suffix(".mp4")
                if candidate.exists():
                    final_path = candidate
            return final_path
    except Exception as exc:
        _raise_friendly_error(exc)
        raise


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    saved_path = download_video(
        url=args.url,
        output_dir=Path(args.output_dir),
        filename_template=args.filename,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=Path(args.cookies).expanduser() if args.cookies else None,
    )
    print(f"Saved to: {saved_path}")


if __name__ == "__main__":
    main()
