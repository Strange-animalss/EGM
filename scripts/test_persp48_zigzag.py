"""Smoke test for the persp48_zigzag scheme.

Asserts:
  1. There are exactly 48 face entries.
  2. Frame names follow the rule "frame_{NNN:03d}_yaw{Y}_pitch{+/-P}".
  3. The first 8 frames match the documented pattern:
        col 0 (yaw=0)   pitch +45 -> +15 -> -15 -> -45  (frames 0..3)
        col 1 (yaw=30)  pitch -45 -> -15 -> +15 -> +45  (frames 4..7)
  4. Across the whole sequence, adjacent frames differ in exactly one axis
     (yaw or pitch but not both).
  5. yaw=0, pitch=+45 actually points into the upper hemisphere of +X.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from erpgen.erp_to_persp import (  # noqa: E402
    PERSP48_ZIGZAG_FACES,
    persp48_zigzag_face_names,
    persp48_zigzag_yaw_pitch,
)


def _parse_name(name: str) -> tuple[int, int, int]:
    parts = name.split("_")
    assert parts[0] == "frame", name
    frame_idx = int(parts[1])
    yaw = int(parts[2].removeprefix("yaw"))
    pitch_str = parts[3].removeprefix("pitch")
    pitch = int(pitch_str)
    return frame_idx, yaw, pitch


def main() -> int:
    names = persp48_zigzag_face_names()
    assert len(names) == 48, f"expected 48 frames, got {len(names)}"

    expected_first_8 = [
        (0, 0, +45), (1, 0, +15), (2, 0, -15), (3, 0, -45),
        (4, 30, -45), (5, 30, -15), (6, 30, +15), (7, 30, +45),
    ]
    for i, (fi, y, p) in enumerate(expected_first_8):
        parsed = _parse_name(names[i])
        assert parsed == (fi, y, p), f"frame {i}: got {parsed}, expected {(fi, y, p)} (name={names[i]})"

    yp = persp48_zigzag_yaw_pitch()
    assert len(yp) == 48
    for i, (yaw, pitch) in enumerate(yp):
        _, ny, np_ = _parse_name(names[i])
        assert int(yaw) == ny and int(pitch) == np_, (i, yaw, pitch, ny, np_)

    for i in range(1, 48):
        prev_y, prev_p = yp[i - 1]
        cur_y, cur_p = yp[i]
        same_yaw = prev_y == cur_y
        same_pitch = prev_p == cur_p
        assert same_yaw ^ same_pitch, (
            f"adjacent frames {i-1}->{i} change both axes: "
            f"({prev_y},{prev_p}) -> ({cur_y},{cur_p})"
        )

    R = PERSP48_ZIGZAG_FACES[names[0]]
    forward_face = np.array([1.0, 0.0, 0.0])
    forward_world = R @ forward_face
    assert forward_world[0] > 0.5, f"yaw=0 pitch=+45 should mostly look +X, got {forward_world}"
    assert forward_world[2] > 0.5, f"yaw=0 pitch=+45 should look UP (+Z), got {forward_world}"

    R6 = PERSP48_ZIGZAG_FACES["frame_024_yaw180_pitch+45"]
    fwd6 = R6 @ forward_face
    assert fwd6[0] < -0.5, f"yaw=180 should look -X, got {fwd6}"

    print(f"OK: 48 frames, zigzag adjacency holds, sanity rotations correct")
    print(f"first 8: {names[:8]}")
    print(f"last 8 : {names[-8:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
