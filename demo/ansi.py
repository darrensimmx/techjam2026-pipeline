"""Terminal rendering primitives. Standard library only.

The colour block and its ``isatty()`` guard follow the one prior art in this
repo, ``.claude/skills/run-sol/bench.py:42-44``: colours are real constants when
stdout is a terminal and empty strings otherwise, so every format string stays
identical and piped output is clean ASCII.

Windows needs two extra things and gets both here: VT processing enabled via
``ctypes`` (older consoles print escapes literally), and line buffering so our
prints interleave correctly with anything else writing to fd 1
(``bench.py:25``).
"""
from __future__ import annotations

import os
import shutil
import sys

# --------------------------------------------------------------------------
# Colour.
# --------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GREY = "\033[90m"
ON_RED = "\033[41m"
ON_YELLOW = "\033[43m"
ON_GREEN = "\033[42m"
BLACK = "\033[30m"

_NAMES = (
    "RESET", "BOLD", "DIM", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA",
    "CYAN", "GREY", "ON_RED", "ON_YELLOW", "ON_GREEN", "BLACK",
)

_ENABLED = True


def _enable_windows_vt() -> None:
    """Turn on ANSI escape processing on a legacy Windows console.

    Best-effort by design: every failure path here means "this console does not
    need it" or "this console cannot do it", and both are handled by the
    isatty/NO_COLOR checks in :func:`configure`.
    """
    if os.name != "nt":
        return
    try:  # pragma: no cover - exercised only on Windows
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # -11 is STD_OUTPUT_HANDLE; 0x0004 is ENABLE_VIRTUAL_TERMINAL_PROCESSING.
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def configure(no_color: bool = False, stream: object = None) -> bool:
    """Decide once whether colour is on, and blank every constant if not.

    Honours ``--no-color`` and the NO_COLOR convention (https://no-color.org).
    Returns the resulting enabled state.
    """
    global _ENABLED
    target = stream if stream is not None else sys.stdout

    try:
        target.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    _enable_windows_vt()

    try:
        is_tty = bool(target.isatty())  # type: ignore[attr-defined]
    except Exception:
        is_tty = False

    _ENABLED = bool(is_tty) and not no_color and not os.environ.get("NO_COLOR")

    if not _ENABLED:
        # Blank our OWN module globals, so every format string downstream stays
        # identical and piped output is clean ASCII. globals() rather than
        # setattr() because it is visibly self-scoped: nothing in demo/ may
        # rewrite another module's attributes except demo/tracer.py.
        for name in _NAMES:
            globals()[name] = ""
    return _ENABLED


def enabled() -> bool:
    return _ENABLED


def paint(text: str, *codes: str) -> str:
    """Wrap ``text`` in ``codes``, or return it untouched when colour is off."""
    prefix = "".join(codes)
    if not prefix:
        return text
    return prefix + text + RESET


# --------------------------------------------------------------------------
# Width.
# --------------------------------------------------------------------------

MIN_WIDTH = 60
NARROW_WIDTH = 90


def terminal_width(default: int = 100) -> int:
    """Usable columns, clamped to something a renderer can actually lay out.

    ``COLUMNS=0`` shows up under some job runners, and ``shutil`` happily
    reports it, so the floor is not decorative.
    """
    try:
        columns = shutil.get_terminal_size(fallback=(default, 24)).columns
    except Exception:
        columns = default
    if not isinstance(columns, int) or columns < MIN_WIDTH:
        return default if default >= MIN_WIDTH else MIN_WIDTH
    return min(columns, 200)


def visible_len(text: str) -> int:
    """Length of ``text`` ignoring SGR escapes."""
    out = 0
    i = 0
    while i < len(text):
        if text[i] == "\033":
            end = text.find("m", i)
            if end == -1:
                return out + len(text) - i
            i = end + 1
            continue
        out += 1
        i += 1
    return out


def truncate(text: str, width: int, ellipsis: str = "..") -> str:
    """Cut ``text`` to ``width`` visible columns. Assumes no escapes inside."""
    if width <= 0:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)] + ellipsis


def pad(text: str, width: int) -> str:
    """Left-justify to ``width`` visible columns, escape-aware."""
    gap = width - visible_len(text)
    return text + " " * gap if gap > 0 else text


def wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Greedy word wrap. Returns at least one line, even for empty input."""
    text = str(text).replace("\n", " ").strip()
    if width <= len(indent) + 1:
        return [indent + text]
    room = width - len(indent)
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= room:
            current += " " + word
        else:
            lines.append(indent + current)
            current = word
        while len(current) > room:  # a single word longer than the line
            lines.append(indent + current[:room])
            current = current[room:]
    lines.append(indent + current)
    return lines


# --------------------------------------------------------------------------
# Structure.
# --------------------------------------------------------------------------

def rule(width: int, char: str = "─") -> str:
    return char * max(0, width)


def titled_rule(title: str, width: int, char: str = "─") -> str:
    """``-- title --------`` filling to ``width``."""
    head = char * 2 + " " + title + " "
    tail = char * max(0, width - visible_len(head))
    return head + tail


def box(lines: list[str], width: int, title: str = "") -> list[str]:
    """A light box-drawing frame. Content is truncated, never wrapped."""
    inner = max(0, width - 4)
    top = "┌─"
    if title:
        top += " " + truncate(title, inner) + " "
    top += "─" * max(0, width - visible_len(top) - 1) + "┐"

    out = [top]
    for line in lines:
        body = truncate(line, inner) if visible_len(line) > inner else line
        out.append("│ " + pad(body, inner) + " │")
    out.append("└" + "─" * max(0, width - 2) + "┘")
    return out


def banner(text: str, width: int, *codes: str) -> str:
    """A full-width alarm line."""
    return paint(pad(" " + truncate(text, max(0, width - 2)), width), *codes)


def columns(row: list[str], widths: list[int], gap: str = "  ") -> str:
    """Fixed-width table row. Trailing column is not padded."""
    cells = []
    for i, (cell, w) in enumerate(zip(row, widths)):
        text = truncate(str(cell), w) if visible_len(str(cell)) > w else str(cell)
        cells.append(text if i == len(widths) - 1 else pad(text, w))
    return gap.join(cells).rstrip()
