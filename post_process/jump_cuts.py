#!/usr/bin/env python3
"""Find scenes that must not be played one right after the other (OpenCV, no AI).

A source video often shows the same shot several times (same camera position, same
objects) with other shots in between. When two of those scenes are joined directly, the
picture stays almost the same and jumps a little: a jump cut, which looks like a glitch.
A clear sideways move of the same picture looks like a slide and is fine. So is a camera that
does not move at all (tripod) while something else is shown: the background stays put.

For every ordered pair (a, b) the last frame of scene a is compared with the first frame
of scene b. Details (SIFT) of one found in the other at about the same size, spread over
several rows of the frame (a caption that stays on screen is only one or two rows), tell how
far the whole picture moved. It is a jump cut when
  - the picture moved a little, but not clearly sideways, or
  - it did not move and the two are almost identical, or
  - the details do not match, but the two look the same as a whole (coarse colours in the
    same places, same colours).
Scenes that follow each other in the source (n, n+1) are left out: that is the source's own
edit, or one long shot cut into parts (which plays on without a jump). Parts of one long
shot in any other order, or with a part left out, are always a jump.

split_video.py writes the result to <meta_dir>/jump_cuts.json; gemini_reorder.py reads it.

Usage (only to look at the result):
    python jump_cuts.py <sources_dir>
"""

import sys
from pathlib import Path

import cv2
import numpy as np

FILE_NAME = "jump_cuts.json"

SIZE = 360                # short side of the compared frames, in pixels
SAME_COLOURS_MIN = 0.65   # colour similarity (0-1) below this: another shot, not looked at further

# Same details
FEATURES = 800
MATCH_RATIO = 0.75        # Lowe's ratio test
INLIERS_MIN = 20          # matched details that agree on one move/zoom
SCALE_RANGE = (0.75, 1.35)  # ... with the objects about the same size
MATCH_ROWS, MATCH_ROWS_MIN = 10, 4  # ... in at least this many of the frame's rows

# A clear sideways slide between two scenes of the same shot is fine
SLIDE_MIN = 0.15          # x width: moved sideways at least this much
SLIDE_FLAT = 2.0          # ... and this many times more than up/down

MOVE_MIN = 0.005          # x width: less than this = the camera did not move
IDENTICAL_MAX = 5.0       # did not move: mean Lab difference of the coarse pictures below this = jump

# Same look
LOOK_GRID = (6, 10)       # coarse picture, columns x rows
LOOK_MAX = 7.5            # mean Lab difference of the coarse pictures
LOOK_COLOURS_MIN = 0.75


def first_and_last(clip: Path) -> list[np.ndarray] | None:
    capture = cv2.VideoCapture(str(clip))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first = last = None
    index = 0
    while capture.grab():  # only the first and the last few frames are converted (much faster)
        if index == 0 or index >= count - 3:
            ok, frame = capture.retrieve()
            if ok:
                first = frame if first is None else first
                last = frame
        index += 1
    capture.release()
    if first is None:
        return None
    scale = SIZE / min(first.shape[:2])
    size = (round(first.shape[1] * scale), round(first.shape[0] * scale))
    return [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in (first, last)]


def describe(frame: np.ndarray, sift) -> dict:
    histogram = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [16, 8, 8],
                             [0, 180, 0, 256, 0, 256])
    points, details = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    return {
        "colours": cv2.normalize(histogram, None, 1, 0, cv2.NORM_L1),
        "look": cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), LOOK_GRID,
                           interpolation=cv2.INTER_AREA).astype(np.float32),
        "points": np.float32([p.pt for p in points]), "details": details, "height": frame.shape[0], "width": frame.shape[1],
    }


def moved(a: dict, b: dict, matcher) -> tuple[float, float] | None:
    """How far the picture moved from a to b (right, down; in frame widths), or None when the
    two do not show the same objects at about the same size."""
    if a["details"] is None or b["details"] is None or len(a["points"]) < INLIERS_MIN or len(b["points"]) < INLIERS_MIN:
        return None
    good = [m[0] for m in matcher.knnMatch(a["details"], b["details"], k=2)
            if len(m) == 2 and m[0].distance < MATCH_RATIO * m[1].distance]
    if len(good) < INLIERS_MIN:
        return None
    here = a["points"][[m.queryIdx for m in good]]
    there = b["points"][[m.trainIdx for m in good]]
    move, agree = cv2.estimateAffinePartial2D(here, there, ransacReprojThreshold=4)
    if move is None:
        return None
    agree = agree.ravel().astype(bool)
    if agree.sum() < INLIERS_MIN or not SCALE_RANGE[0] <= np.hypot(move[0, 0], move[1, 0]) <= SCALE_RANGE[1]:
        return None
    if len(set((here[agree][:, 1] * MATCH_ROWS / a["height"]).astype(int))) < MATCH_ROWS_MIN:
        return None
    right, down = np.median(there[agree] - here[agree], axis=0) / a["width"]
    return float(right), float(down)


def is_slide(right: float, down: float) -> bool:
    """A clear sideways move: looks intended, not like a glitch."""
    return abs(right) >= SLIDE_MIN and abs(right) >= SLIDE_FLAT * abs(down)


def is_jump(a: dict, b: dict, matcher) -> bool:
    """a: end of a scene, b: start of the scene after it."""
    colours = 1 - cv2.compareHist(a["colours"], b["colours"], cv2.HISTCMP_BHATTACHARYYA)
    if colours < SAME_COLOURS_MIN:
        return False
    move = moved(a, b, matcher)
    look = float(np.abs(a["look"] - b["look"]).mean())
    if move is None:
        return colours >= LOOK_COLOURS_MIN and look <= LOOK_MAX
    if np.hypot(*move) < MOVE_MIN:
        return look <= IDENTICAL_MAX
    return not is_slide(*move)


def find_jump_cuts(sources: Path, shots: list[list[int]] = ()) -> list[list[int]]:
    """[a, b] for every pair of scenes where b must not come right after a.

    shots: for every long shot that was cut into parts, the numbers of its scenes.
    """
    sift, matcher = cv2.SIFT_create(FEATURES), cv2.BFMatcher()
    starts, ends = {}, {}
    for clip in sorted((p for p in sources.glob("*.mp4") if p.stem.isdigit()), key=lambda p: int(p.stem)):
        frames = first_and_last(clip)
        if frames is not None:
            starts[int(clip.stem)], ends[int(clip.stem)] = (describe(f, sift) for f in frames)
    same_shot = {(a, b) for parts in shots for a in parts for b in parts}
    return [[a, b] for a in ends for b in starts if b not in (a, a + 1)
            and ((a, b) in same_shot or is_jump(ends[a], starts[b], matcher))]


if __name__ == "__main__":
    for pair in find_jump_cuts(Path(sys.argv[1])):
        print(f"{pair[0]} -> {pair[1]}")
