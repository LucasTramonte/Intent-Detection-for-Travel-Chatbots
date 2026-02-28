from __future__ import annotations


def format_run_message(run_id: str | None, message: str) -> str:
    if run_id is None:
        return message
    return f"[run_id={run_id}] {message}"
