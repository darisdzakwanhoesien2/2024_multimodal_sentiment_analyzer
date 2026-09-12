import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# SQLite rather than a server-based DB: this is a single-operator tool on a
# disk-constrained VPS that already runs several other services — no reason
# to add a new database process for what's still a modest amount of data.
#
# This is a genuine persistent store, not a cache: channels and videos are
# real rows in real tables and are never deleted. REFRESH_INTERVAL_SECONDS
# only controls when /api/youtube/channel bothers re-asking YouTube for
# updates (view counts, new uploads) — it does not gate whether previously
# discovered data is still available. list_channels()/list_all_videos()
# always return everything ever seen, regardless of age.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "youtube.db"

REFRESH_INTERVAL_SECONDS = 6 * 3600


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
            CREATE TABLE IF NOT EXISTS channels (
                channel_url TEXT PRIMARY KEY,
                channel_title TEXT NOT NULL,
                first_fetched_at REAL NOT NULL,
                last_fetched_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                channel_url TEXT NOT NULL REFERENCES channels(channel_url),
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                duration REAL,
                thumbnail TEXT,
                view_count INTEGER,
                first_seen_at REAL NOT NULL,
                last_seen_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_url)")


def _row_to_video(row) -> dict:
    (video_id, channel_url, title, url, duration, thumbnail, view_count,
     first_seen_at, last_seen_at, channel_title) = row
    return {
        "id": video_id, "channel_url": channel_url, "channel_title": channel_title,
        "title": title, "url": url, "duration": duration, "thumbnail": thumbnail,
        "view_count": view_count, "first_seen_at": first_seen_at, "last_seen_at": last_seen_at,
    }


def get_channel(channel_url: str) -> dict | None:
    """Whatever is stored for this channel, however old — staleness is a
    separate question (see is_stale) from whether the data still exists."""
    with _connect() as conn:
        channel_row = conn.execute(
            "SELECT channel_title, first_fetched_at, last_fetched_at FROM channels WHERE channel_url = ?",
            (channel_url,),
        ).fetchone()
        if channel_row is None:
            return None
        video_rows = conn.execute(
            """
            SELECT v.video_id, v.channel_url, v.title, v.url, v.duration, v.thumbnail,
                   v.view_count, v.first_seen_at, v.last_seen_at, c.channel_title
            FROM videos v JOIN channels c ON c.channel_url = v.channel_url
            WHERE v.channel_url = ?
            ORDER BY v.last_seen_at DESC
            """,
            (channel_url,),
        ).fetchall()

    channel_title, first_fetched_at, last_fetched_at = channel_row
    age = time.time() - last_fetched_at
    return {
        "channel_title": channel_title,
        "channel_url": channel_url,
        "videos": [_row_to_video(r) for r in video_rows],
        "first_fetched_at": first_fetched_at,
        "last_fetched_at": last_fetched_at,
        "stale": age > REFRESH_INTERVAL_SECONDS,
        "age_seconds": round(age),
    }


def is_stale(channel_url: str, max_age: float = REFRESH_INTERVAL_SECONDS) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT last_fetched_at FROM channels WHERE channel_url = ?", (channel_url,),
        ).fetchone()
    return row is None or (time.time() - row[0]) > max_age


def save_channel(channel_url: str, channel_title: str, videos: list[dict]):
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO channels (channel_url, channel_title, first_fetched_at, last_fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel_url) DO UPDATE SET
                channel_title = excluded.channel_title,
                last_fetched_at = excluded.last_fetched_at
            """,
            (channel_url, channel_title, now, now),
        )
        for v in videos:
            conn.execute(
                """
                INSERT INTO videos (video_id, channel_url, title, url, duration, thumbnail,
                                     view_count, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title = excluded.title,
                    duration = excluded.duration,
                    thumbnail = excluded.thumbnail,
                    view_count = excluded.view_count,
                    last_seen_at = excluded.last_seen_at
                """,
                (v["id"], channel_url, v["title"], v["url"], v.get("duration"),
                 v.get("thumbnail"), v.get("view_count"), now, now),
            )


def list_channels() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT channel_url, channel_title, first_fetched_at, last_fetched_at "
            "FROM channels ORDER BY last_fetched_at DESC"
        ).fetchall()
    return [
        {"channel_url": r[0], "channel_title": r[1], "first_fetched_at": r[2], "last_fetched_at": r[3]}
        for r in rows
    ]


def list_all_videos() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT v.video_id, v.channel_url, v.title, v.url, v.duration, v.thumbnail,
                   v.view_count, v.first_seen_at, v.last_seen_at, c.channel_title
            FROM videos v JOIN channels c ON c.channel_url = v.channel_url
            ORDER BY v.last_seen_at DESC
            """
        ).fetchall()
    return [_row_to_video(r) for r in rows]


init_db()
