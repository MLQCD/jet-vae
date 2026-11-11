"""Reader for the ``.dat`` jet files (one particle per row: ``pid px py pz E``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_events_from_dat"]


def load_events_from_dat(path: str | Path) -> list[np.ndarray]:
    """Load jets with rows ``pid px py pz E``.

    Production files delimit jets with ``<event_begin>:id`` / ``<event_end>:id``
    lines. Blank-line-separated files are accepted too (small studies, tests).
    Returns one float32 array ``[n_particles, 5]`` per non-empty jet.
    """
    events: list[np.ndarray] = []
    current: list[list[float]] = []
    saw_markers = False

    def finish_event() -> None:
        nonlocal current
        if current:
            events.append(np.asarray(current, dtype=np.float32))
            current = []

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                if not saw_markers:
                    finish_event()
                continue
            if stripped.startswith("<event_begin>"):
                saw_markers = True
                finish_event()  # tolerate a missing previous end marker
                continue
            if stripped.startswith("<event_end>"):
                saw_markers = True
                finish_event()
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ValueError(
                    f"Expected 5 columns in {path}:{line_number}, got: {stripped!r}"
                )
            try:
                pid = int(fields[0])
                px, py, pz, energy = (float(v) for v in fields[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid particle row in {path}:{line_number}: {stripped!r}"
                ) from exc
            current.append([pid, px, py, pz, energy])
    finish_event()
    if not events:
        raise ValueError(f"No non-empty events found in {path}")
    return events
