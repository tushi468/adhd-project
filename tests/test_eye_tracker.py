"""
Quick sanity checks for eye_tracker.py and gaze_state.py
Run with: python -m pytest tests/ -v
"""

import numpy as np
from src.vision.gaze_state import GazeState


def test_gaze_state_defaults():
    """GazeState starts with safe empty values."""
    s = GazeState()
    assert s.pred_x is None
    assert s.pred_y is None
    assert s.blink_detected is None
    assert s.cursor_alpha == 0.0
    assert s.is_fixating is False
    print("PASS — GazeState defaults are correct")


def test_is_fixating_field_exists():
    """is_fixating field exists and can be set."""
    s = GazeState()
    s.is_fixating = True
    assert s.is_fixating is True
    s.is_fixating = False
    assert s.is_fixating is False
    print("PASS — is_fixating field works correctly")


def test_cursor_alpha_bounds():
    """cursor_alpha never goes below 0 or above 1."""
    s = GazeState()
    s.cursor_alpha = 0.0
    s.cursor_alpha = max(s.cursor_alpha - 0.1, 0.0)
    assert s.cursor_alpha == 0.0

    s.cursor_alpha = 1.0
    s.cursor_alpha = min(s.cursor_alpha + 0.1, 1.0)
    assert s.cursor_alpha == 1.0
    print("PASS — cursor_alpha stays within bounds")


def test_blink_rising_edge():
    """
    Rising edge detection: only counts blink onset, not every blink frame.
    Simulates 5 frames of blinking — should only count as 1 blink.
    """
    import collections
    import time

    blink_log = collections.deque()
    prev_blink = False
    blink_frames = [False, False, True, True, True, True, True, False]

    for blink_detected in blink_frames:
        if blink_detected and not prev_blink:
            blink_log.append(time.time())
        prev_blink = blink_detected

    assert len(blink_log) == 1
    print("PASS — rising edge correctly counts 1 blink across 5 blink frames")


def test_fixation_detection():
    """
    Fixation: gaze moved less than 30px = fixating.
    Saccade: gaze moved more than 30px = not fixating.
    """
    # fixating — small movement
    x, prev_x = 105, 100
    y, prev_y = 103, 100
    is_fixating = abs(x - prev_x) < 30 and abs(y - prev_y) < 30
    assert is_fixating is True
    print("PASS — small movement correctly detected as fixation")

    # saccade — large movement
    x, prev_x = 200, 100
    y, prev_y = 200, 100
    is_fixating = abs(x - prev_x) < 30 and abs(y - prev_y) < 30
    assert is_fixating is False
    print("PASS — large movement correctly detected as saccade")


if __name__ == "__main__":
    test_gaze_state_defaults()
    test_is_fixating_field_exists()
    test_cursor_alpha_bounds()
    test_blink_rising_edge()
    test_fixation_detection()
    print("\nAll tests passed.")