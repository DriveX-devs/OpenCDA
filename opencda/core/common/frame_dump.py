# -*- coding: utf-8 -*-
"""Helpers for saving runtime visualization frames."""

import os
import re


def _safe_path_part(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "frames"


def get_root():
    root = os.environ.get("OPENCDA_FRAME_DUMP_DIR")
    if not root:
        return None
    os.makedirs(root, exist_ok=True)
    return root


def make_path(parts, filename):
    root = get_root()
    if not root:
        return None

    folder = os.path.join(root, *[_safe_path_part(part) for part in parts])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, _safe_path_part(filename))
