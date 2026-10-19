"""Local, request-exact checkpoints for interrupted Groq extractions."""
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from filelock import FileLock


def fingerprint(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def retry_after_seconds(headers, now):
    """Only interpret the explicit HTTP Retry-After, never guess a reset window."""
    value = headers.get("retry-after")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                return None
            origin = now
            if headers.get("date"):
                server_date = parsedate_to_datetime(headers["date"])
                if server_date.tzinfo is not None:
                    origin = server_date.astimezone(timezone.utc).timestamp()
            seconds = max(0, deadline.astimezone(timezone.utc).timestamp() - origin)
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def rate_limit_details(response, key, now):
    headers = {}
    response_headers = response.headers if isinstance(response.headers, Mapping) else {}
    for name, value in response_headers.items():
        name = str(name).lower()
        if name in {"retry-after", "date", "x-request-id"} or name.startswith("x-ratelimit-"):
            headers[name] = str(value).replace(key, "[REDACTED]") if key else str(value)
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    safe_error = {name: value.replace(key, "[REDACTED]") if key else value
                  for name in ("message", "type", "code")
                  if isinstance((value := error.get(name)), str)}
    delay = retry_after_seconds(headers, now)
    return {"http_status": 429, "headers": headers, "error": safe_error,
            "received_at": now, "retry_after_seconds": delay,
            "retry_not_before": now + delay if delay is not None else None}


class GroqResumeStore:
    VERSION = 1

    def __init__(self, directory=None):
        default = Path(__file__).resolve().parent.parent / "data" / "groq_resume"
        self.directory = Path(directory or os.getenv("GROQ_RESUME_DIR", str(default)))

    def path(self, checkpoint_id):
        if len(checkpoint_id) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_id):
            raise ValueError("Invalid checkpoint identity")
        return self.directory / (checkpoint_id + ".json")

    def lock(self, checkpoint_id):
        self.directory.mkdir(parents=True, exist_ok=True)
        # A second process must not pay for the same in-flight extraction.
        return FileLock(str(self.path(checkpoint_id)) + ".lock", timeout=0)

    def load(self, checkpoint_id):
        path = self.path(checkpoint_id)
        if not path.exists():
            return {"version": self.VERSION, "checkpoint_id": checkpoint_id,
                    "responses": {}, "blocked": None}
        state = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(state, dict) or state.get("version") != self.VERSION or state.get("checkpoint_id") != checkpoint_id
                or not isinstance(state.get("responses"), dict)):
            raise ValueError("Invalid Groq checkpoint")
        blocked = state.get("blocked")
        if blocked is not None:
            if not isinstance(blocked, dict):
                raise ValueError("Invalid Groq checkpoint quota state")
            deadline = blocked.get("retry_not_before")
            if deadline is not None and (not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
                raise ValueError("Invalid Groq checkpoint retry deadline")
        for request_id, row in state["responses"].items():
            if not isinstance(row, dict) or row.get("sha256") != fingerprint(row.get("payload")):
                raise ValueError("Corrupt Groq checkpoint response")
        return state

    def save(self, state):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path(state["checkpoint_id"])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=target.stem + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(state, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def complete(self, checkpoint_id):
        self.path(checkpoint_id).unlink(missing_ok=True)
