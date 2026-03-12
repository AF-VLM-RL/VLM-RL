"""Shared utilities for experiments."""

import re


def sanitize_prompt_for_filename(prompt: str, max_length: int = 40) -> str:
    """Sanitize a text prompt for safe use in file names, logs, and wandb names.

    Replaces spaces with underscores and removes characters invalid in file paths.
    Truncates if longer than max_length.
    """
    if not prompt:
        return "default"
    # Replace invalid path chars and collapse multiple spaces
    s = re.sub(r'[\\/:*?"<>|]', "", prompt)
    s = re.sub(r"\s+", "_", s.strip())
    s = s.strip("_") or "default"
    return s[:max_length] if len(s) > max_length else s
