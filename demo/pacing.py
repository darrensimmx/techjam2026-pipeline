"""Line-by-line output, so a turn unfolds instead of landing whole.

Both CLIs print through :func:`say` rather than ``print``. It writes one line,
pauses, writes the next -- which is the whole point: a turn that appears all at
once tells a viewer nothing about the agent doing work in stages.

Three properties earn their keep:

  - **It splits on newlines.** Several call sites hand over pre-joined blocks
    (``"\\n".join(ansi.box(...))``, the two-line status string). Splitting here
    means the renderers needed no restructuring -- ``print(`` became ``say(``
    and multi-line blocks paced correctly for free.

  - **It is inert when stdout is not a terminal.** Not cosmetic: the demo tests
    capture stdout, and a paced test run would take minutes instead of the
    current sub-second. Piping to a file or a pager is instant too.

  - **KeyboardInterrupt is not swallowed.** Ctrl-C during a long render must
    exit, and both CLIs catch it at the top level to do so without a traceback.

Standard library only, and 3.11-compatible.
"""
from __future__ import annotations

import sys
import time

# Per-CLI defaults live with each CLI, not here: the backend emits roughly seven
# times as many lines per turn as the frontend, so one shared number would put
# the two terminals badly out of step.
DEFAULT_LINE_DELAY = 0.03

_DELAY = 0.0
_ENABLED = False


def configure(line_delay: float = DEFAULT_LINE_DELAY, stream: object = None) -> bool:
    """Decide once whether to pace. Returns the resulting enabled state.

    Paces only when stdout is a real terminal AND a positive delay was asked
    for, so ``--line-delay 0`` and any redirect both give instant output.
    """
    global _DELAY, _ENABLED
    target = stream if stream is not None else sys.stdout

    try:
        is_tty = bool(target.isatty())
    except Exception:
        is_tty = False

    try:
        delay = float(line_delay)
    except (TypeError, ValueError):
        delay = 0.0

    _DELAY = max(0.0, delay)
    _ENABLED = is_tty and _DELAY > 0.0
    return _ENABLED


def enabled() -> bool:
    return _ENABLED


def delay() -> float:
    return _DELAY


def say(text: object = "") -> None:
    """Print ``text``, pausing between its lines when pacing is on.

    A drop-in for ``print`` with a single argument. An empty call prints a
    blank line, unpaced -- a blank line carries nothing to read, so waiting on
    it only makes the render feel sticky.
    """
    body = "" if text is None else str(text)

    if not _ENABLED:
        print(body)
        return

    for line in body.split("\n"):
        print(line)
        # Blank lines carry nothing to read; pausing on them only makes the
        # render feel sticky at every block boundary.
        if line.strip():
            time.sleep(_DELAY)


def block(text: object = "") -> None:
    """Print without pacing. For banners that explain why a run is aborting."""
    print("" if text is None else str(text))


def pause(seconds: float) -> None:
    """A deliberate beat, honoured only when pacing is on."""
    if _ENABLED and seconds > 0:
        time.sleep(seconds)
