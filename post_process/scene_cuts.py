#!/usr/bin/env python3
"""Find the places where a video changes to another shot (OpenCV + numpy, no AI).

Used by split_video.py next to PySceneDetect, which only finds fast cuts at least 0.6 s
apart. This module looks at small frames and finds:
  cuts       one frame to the next: the difference is big and stands out sharply from
             the frames around it (fast motion is big too, but lasts several frames).
  dissolves  one shot fades into the next (also fades to/from black or white): the frame
             in the middle is a mix of the frame before and the frame after. A camera or
             an object that moves changes the picture as much, but is not such a mix.

combine() merges both with PySceneDetect's scenes. The result is never "the one with more
scenes": a detector that is too sensitive also has more scenes, by cutting one shot into
pieces. Every place that passes its own test is kept.
"""

import math
from pathlib import Path

import cv2
import numpy as np

SHORT_SIDE = 48           # analysis size of the short side, in pixels

# Cuts (differences are mean absolute differences of 8-bit BGR pixels)
CUT_DIFF = 20.0           # difference to the previous frame
CUT_SHARPNESS = 3.0       # ... and this many times the biggest one of the frames around it
CUT_NEIGHBOURS = 5        # frames looked at on each side

# Dissolves
WINDOWS_S = (0.2, 0.3, 0.4)  # compare a frame with those this far before and after it
DISSOLVE_DIFF = 25.0      # the frames before and after are this different
DISSOLVE_MIX = (0.25, 0.75)  # share of the frame before in the mix
DISSOLVE_RESIDUAL = 0.21  # what the mix does not explain, relative to the difference
                          # (real dissolves: up to 0.17; camera and object motion: 0.25 and more)
DISSOLVE_ALIKE = 0.85     # correlation of before and after above this: same picture, other light
REACH_S = 0.8             # a dissolve starts/ends at most this far from its middle
RAMP_SHARE = 0.5          # frames changing the mix at least this much (x typical) belong to it

MIN_SCENE_S = 0.5         # a cut never makes a scene shorter than this; a leftover next to a
                          # dissolve that is shorter is dropped


def read_frames(video: Path) -> tuple[np.ndarray, np.ndarray]:
    """(small BGR frames, their times in seconds)."""
    capture = cv2.VideoCapture(str(video))
    frames, times = [], []
    size = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if size is None:
            h, w = frame.shape[:2]
            scale = SHORT_SIDE / min(h, w)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
        times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    capture.release()
    return np.array(frames, np.uint8), np.array(times)


def find_cuts(frames: np.ndarray) -> list[int]:
    """Frame numbers (0-based) that start another shot."""
    n = len(frames)
    diff = np.zeros(n)
    for t in range(1, n):
        diff[t] = cv2.absdiff(frames[t], frames[t - 1]).mean()
    cuts = []
    for t in range(1, n):
        around = np.r_[diff[max(1, t - CUT_NEIGHBOURS):t], diff[t + 1:t + 1 + CUT_NEIGHBOURS]]
        if diff[t] >= CUT_DIFF and diff[t] >= CUT_SHARPNESS * max(1.0, around.max(initial=0.0)):
            cuts.append(t)
    return cuts


def mix_share(before: np.ndarray, after: np.ndarray, frame: np.ndarray) -> float:
    """Share of `before` in the best mix of before and after that looks like frame."""
    d = before - after
    return float((d * (frame - after)).sum() / max(1.0, (d * d).sum()))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(1.0, np.sqrt((a * a).sum() * (b * b).sum())))


def find_dissolves(frames: np.ndarray, fps: float, cuts: list[int]) -> list[tuple[int, int]]:
    """Dissolves as (first, last) mixed frame; none crosses a cut."""
    n = len(frames)
    edges = [0, *cuts, n]
    data = frames.reshape(n, -1)
    hits: dict[int, float] = {}
    for lo, hi in zip(edges, edges[1:]):  # frames lo..hi-1 have no cut
        for seconds in WINDOWS_S:
            w = max(2, round(seconds * fps))
            for t in range(lo + w, hi - w):
                before, after, frame = (data[i].astype(np.float32) for i in (t - w, t + w, t))
                diff = float(np.abs(before - after).mean())
                if diff < DISSOLVE_DIFF:
                    continue
                share = mix_share(before, after, frame)
                if not DISSOLVE_MIX[0] <= share <= DISSOLVE_MIX[1]:
                    continue
                residual = float(np.abs(frame - share * before - (1 - share) * after).mean()) / diff
                if residual <= DISSOLVE_RESIDUAL and correlation(before, after) <= DISSOLVE_ALIKE:
                    hits[t] = min(hits.get(t, 1.0), residual)

    groups: list[list[int]] = []
    for t in sorted(hits):
        if groups and t - groups[-1][-1] <= round(0.1 * fps):
            groups[-1].append(t)
        else:
            groups.append([t])

    dissolves = []
    for group in groups:
        middle = group[len(group) // 2]
        lo = max(e for e in edges if e <= middle)
        hi = min(e for e in edges if e > middle) - 1
        first, last = max(lo, middle - round(REACH_S * fps)), min(hi, middle + round(REACH_S * fps))
        before, after = data[first].astype(np.float32), data[last].astype(np.float32)
        share = [mix_share(before, after, data[t].astype(np.float32)) for t in range(first, last + 1)]
        ramp = -np.diff(share)  # ramp[i]: how much the mix moves from frame first+i to the next
        core = [ramp[t - first] for t in range(max(first, group[0] - 1), min(last, group[-1] + 1))]
        enough = RAMP_SHARE * float(np.median(core))
        a = b = min(max(middle - first, 0), len(ramp) - 1)
        while a > 0 and ramp[a - 1] >= enough:
            a -= 1
        while b < len(ramp) - 1 and ramp[b + 1] >= enough:
            b += 1
        # Frames first+a and first+b+1 are the last clean one before and the first clean one after.
        start, end = first + a + 1, first + b
        if dissolves and start <= dissolves[-1][1] + 1:
            dissolves[-1] = (dissolves[-1][0], max(end, dissolves[-1][1]))
        elif end >= start:
            dissolves.append((start, end))
    return dissolves


def combine(scenes: list[tuple[float, float]], cuts: list[float],
            dissolves: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[float]]:
    """Merge PySceneDetect's scenes (seconds) with OpenCV's cuts and dissolves.

    dissolves are (start, end): the mixed frames are in start <= t < end. They are left out of
    the scenes, so no scene shows a ghost of another one. Returns (scenes, cuts added).
    """
    begin, finish = scenes[0][0], scenes[-1][1]
    known = [s for s, _ in scenes[1:]]
    known = [c for c in known if not any(a <= c <= b for a, b in dissolves)]
    added = []
    fixed = sorted([begin, finish, *known, *(x for d in dissolves for x in d)])
    for cut in cuts:
        if all(abs(cut - x) >= MIN_SCENE_S - 1e-6 for x in fixed) and not any(a <= cut <= b for a, b in dissolves):
            added.append(cut)
            fixed = sorted([*fixed, cut])

    starts = sorted([begin, *known, *added, *(b for _, b in dissolves)])
    ends = sorted([finish, *known, *added, *(a for a, _ in dissolves)])
    fades = {x for d in dissolves for x in d}
    result = []
    for start, end in zip(starts, ends):
        if end - start < MIN_SCENE_S - 1e-6 and (start in fades or end in fades):
            continue
        result.append((start, end))
    return (result or scenes), added


def split_long(scenes: list[tuple[float, float]], times: np.ndarray, max_s: float, part_s: float,
               trim_s: float) -> tuple[list[tuple[float, float]], list[dict]]:
    """Cut scenes longer than max_s (one long shot, nothing to detect) into equal parts.

    As few parts as possible, each at most part_s, starting on a frame. Equal parts: no short
    leftover. Up to trim_s is dropped from the end of the shot when that saves a part. Two
    neighbouring parts never last more than 2 x part_s, also after rounding to frames (so a
    plan may play them one after the other, see gemini_reorder.py SOURCE_RUN_MAX_S).
    Returns (scenes, what was split). max_s 0 = leave them as they are.
    """
    if max_s <= 0:
        return scenes, []
    frame_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    if frame_s > 0:  # two parts = a whole number of frames, not more than 2 x part_s
        part_s = math.floor(2 * part_s / frame_s + 1e-6) * frame_s / 2

    def on_frame(seconds: float) -> float:
        return float(times[max(0, np.searchsorted(times, seconds + 1e-3, side="right") - 1)]) if len(times) else seconds

    result, done = [], []
    for start, end in scenes:
        length = end - start
        if length <= max_s + 1e-3:
            result.append((start, end))
            continue
        count = max(1, math.ceil((length - trim_s) / part_s - 1e-6))
        step = min(part_s, length / count)
        last = end if step * count >= length - 1e-3 else on_frame(start + step * count)
        edges = [start, *(on_frame(start + step * k) for k in range(1, count)), last]
        result += list(zip(edges, edges[1:]))
        done.append({"start": round(start, 3), "end": round(end, 3), "parts": count,
                     "trimmed": round(end - last, 3)})
    return result, done
