import cv2
import pygame
import numpy as np
import time
import collections

from src.config import *
from eyetrax.calibration import run_9_point_calibration
from eyetrax.filters import make_kalman, KalmanEMASmoother
from eyetrax.gaze import GazeEstimator
from src.vision.gaze_state import GazeState

# Created once at module level — expensive to initialise per frame
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# Fixation: gaze stable within 25px = fixating
FIXATION_THRESHOLD_PX = 25
# Require this many consecutive stable frames before declaring fixation.
# Prevents the flag flickering on single noisy frames. 5 frames ≈ 167 ms.
FIXATION_MIN_FRAMES = 5
# Outlier rejection: skip jumps larger than this per frame.
# 280 was nearly full screen height — tightened to block blink artefacts.
MAX_JUMP_PX = 150


def _preprocess(frame):
    """
    Bilateral filter + CLAHE on luminance channel.
    Improves iris detection for glasses wearers and small/squinted eyes.
    - Bilateral filter removes noise without blurring the iris edge
    - CLAHE only on L (brightness) channel — does not distort colour
    Inspired by GazeTracking (github.com/antoinelame/GazeTracking)
    """
    frame = cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge([_CLAHE.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _detect_iris_center(frame, face_landmarks, eye_indices):
    """
    Iris center detection on isolated eye region.
    Finds darkest point (pupil center) for sub-landmark accuracy.
    Reduces residual jitter that smoothing alone cannot fix.
    Inspired by GazeTracking iris detection approach.
    NOTE: Foundation for Week 1 — not yet integrated into prediction pipeline.
    """
    h, w = frame.shape[:2]
    xs = [int(face_landmarks[i].x * w) for i in eye_indices]
    ys = [int(face_landmarks[i].y * h) for i in eye_indices]
    x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, w)
    y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, h)
    eye_roi = frame[y1:y2, x1:x2]
    if eye_roi.size == 0:
        return None
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, _, _, max_loc = cv2.minMaxLoc(cv2.bitwise_not(gray))
    return x1 + max_loc[0], y1 + max_loc[1]


class EyeTracker:
    def __init__(self):
        self.gaze_estimator = GazeEstimator()
        self.current_state = GazeState()

        # rolling blink log → BPM over last 60s
        self._blink_log = collections.deque()
        self._prev_blink = False

        # outlier rejection
        self._last_x = None
        self._last_y = None

        # fixation: previous smoothed position
        self._smooth_x = None
        self._smooth_y = None

        # consecutive stable frames — resets on any movement
        self._fixation_counter = 0

        self._set_smoother()

    # -----------------------------
    # MODEL
    # -----------------------------

    def create_model(self, path):
        try:
            run_9_point_calibration(self.gaze_estimator)
            self.gaze_estimator.save_model(path)
            self.gaze_estimator.load_model(path)
            print(f"[EyeTracker] Model created: {path}")
        except Exception as e:
            print(f"[EyeTracker] \033[91mError:\033[0m create_model failed: {e}")
            raise

    def load_model(self, path):
        try:
            self.gaze_estimator.load_model(path)
            print(f"[EyeTracker] Loaded model: {path}")
        except Exception as e:
            print(f"[EyeTracker] \033[91mError:\033[0m load_model failed: {e}")
            raise

    # -----------------------------
    # FILTER
    # -----------------------------

    def _set_smoother(self):
        # KalmanEMA reduces jitter better than plain Kalman at 30fps.
        # ema_alpha lowered to 0.20 — less reactive to frame noise,
        # ~1 frame extra latency which is acceptable for gaze interaction.
        self.smoother = KalmanEMASmoother(kf=make_kalman(), ema_alpha=0.20)

    # -----------------------------
    # RESET
    # -----------------------------

    def reset(self):
        """Clear per-trial state without losing the loaded model."""
        self.current_state = GazeState()
        self._blink_log.clear()
        self._prev_blink = False
        self._last_x = None
        self._last_y = None
        self._smooth_x = None
        self._smooth_y = None
        self._fixation_counter = 0
        self._set_smoother()

    # -----------------------------
    # UPDATE
    # -----------------------------

    def update(self, frame) -> GazeState:
        if frame is None:
            return self.current_state

        processed = _preprocess(frame)
        s = self.current_state

        try:
            features, blink_detected = self.gaze_estimator.extract_features(processed)
        except Exception:
            return s

        s.blink_detected = blink_detected

        # blink onset — rising edge only (1 count per blink, not per frame)
        if blink_detected and not self._prev_blink:
            self._blink_log.append(time.time())
        self._prev_blink = blink_detected

        # drop blinks older than 60s
        now = time.time()
        while self._blink_log and self._blink_log[0] < now - 60:
            self._blink_log.popleft()

        if features is not None and not blink_detected:
            try:
                gaze_point = self.gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)

                # outlier rejection — skip physically impossible jumps
                if self._last_x is not None:
                    dist = ((x - self._last_x)**2 + (y - self._last_y)**2) ** 0.5
                    if dist > MAX_JUMP_PX:
                        s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP * 0.5, 0.0)
                        return s

                self._last_x, self._last_y = x, y

                # smooth with KalmanEMA
                sx, sy = self.smoother.step(x, y)
                s.pred_x, s.pred_y = sx, sy
                s.cursor_alpha = min(s.cursor_alpha + CURSOR_STEP, 1.0)

                # fixation detection on SMOOTHED position
                # compares current smooth to previous smooth — stable, not flickery.
                # counter must reach FIXATION_MIN_FRAMES before is_fixating = True,
                # any movement resets it immediately.
                if self._smooth_x is not None:
                    moved = (
                        (sx - self._smooth_x) ** 2 +
                        (sy - self._smooth_y) ** 2
                    ) ** 0.5
                    if moved < FIXATION_THRESHOLD_PX:
                        self._fixation_counter = min(self._fixation_counter + 1, FIXATION_MIN_FRAMES)
                    else:
                        self._fixation_counter = 0
                    s.is_fixating = self._fixation_counter >= FIXATION_MIN_FRAMES
                else:
                    self._fixation_counter = 0
                    s.is_fixating = False

                self._smooth_x, self._smooth_y = sx, sy

            except Exception:
                s.pred_x = s.pred_y = None
                s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP, 0.0)
                s.is_fixating = False
                self._fixation_counter = 0

        else:
            # hold position during blink — less jarring visually
            if not blink_detected:
                s.pred_x = s.pred_y = None
            s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP * 0.5, 0.0)
            s.is_fixating = False
            self._fixation_counter = 0

        return s

    # -----------------------------
    # RENDER
    # -----------------------------

    def render(self, screen, frame, current_state, font):
        if frame is None:
            return

        frame = cv2.flip(frame, 1)

        if (
            current_state.pred_x is not None
            and current_state.pred_y is not None
            and current_state.cursor_alpha > 0
        ):
            cx = int(current_state.pred_x)
            cy = int(current_state.pred_y)
            radius = 15
            alpha_int = int(current_state.cursor_alpha * 255)

            # SRCALPHA so alpha fade actually renders
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                surf, (*COLORS[4][:3], alpha_int), (radius, radius), radius
            )

            # fixation ring: green = fixating, amber = scanning
            if current_state.is_fixating:
                ring_col = (60, 220, 120, alpha_int // 2)
            else:
                ring_col = (220, 160, 40, alpha_int // 3)

            pygame.draw.circle(surf, ring_col, (radius, radius), radius, 3)
            screen.blit(surf, (cx - radius, cy - radius))

        # camera thumbnail
        try:
            thumb = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            screen.blit(
                pygame.surfarray.make_surface(thumb),
                (screen.get_width() - CAM_WIDTH - MARGIN,
                 screen.get_height() - CAM_HEIGHT - MARGIN),
            )
        except Exception:
            pass

        # HUD: blink state + BPM
        bpm = len(self._blink_log)
        if current_state.blink_detected:
            blink_txt = f"Blinking   BPM: {bpm}"
            blink_clr = (80, 80, 240)
        else:
            blink_txt = f"Eyes open  BPM: {bpm}"
            blink_clr = (60, 210, 100)
        screen.blit(font.render(blink_txt, True, blink_clr), (50, 100))

        # fixation label — clearly visible
        try:
            small = pygame.font.SysFont("Arial", 18)
            if current_state.is_fixating:
                fix_txt, fix_col = "Fixating", (60, 220, 120)
            else:
                fix_txt, fix_col = "Scanning", (220, 160, 40)
            screen.blit(small.render(fix_txt, True, fix_col), (50, 140))
        except Exception:
            pass
