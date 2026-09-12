import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# SQLite rather than a server-based DB: this is a single-operator tool on a
# disk-constrained VPS that already runs several other services — no reason
# to add a new database process for a small cache table.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"

CACHE_TTL_SECONDS = 6 * 3600


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_cache (
                channel_url TEXT PRIMARY KEY,
                channel_title TEXT NOT NULL,
                videos_json TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
        """)


def get_cached_channel(channel_url: str, max_age: float = CACHE_TTL_SECONDS) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT channel_title, videos_json, fetched_at FROM channel_cache WHERE channel_url = ?",
            (channel_url,),
        ).fetchone()
    if row is None:
        return None

    channel_title, videos_json, fetched_at = row
    age = time.time() - fetched_at
    if age > max_age:
        return None

    return {
        "channel_title": channel_title,
        "channel_url": channel_url,
        "videos": json.loads(videos_json),
        "cached": True,
        "fetched_at": fetched_at,
        "cache_age_seconds": round(age),
    }


def save_channel_cache(channel_url: str, channel_title: str, videos: list[dict]):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO channel_cache (channel_url, channel_title, videos_json, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel_url) DO UPDATE SET
                channel_title = excluded.channel_title,
                videos_json = excluded.videos_json,
                fetched_at = excluded.fetched_at
            """,
            (channel_url, channel_title, json.dumps(videos), time.time()),
        )


init_db()
