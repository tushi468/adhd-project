import cv2
import pygame
import numpy as np
import time
import collections
import os   

from src.config import *
from eyetrax.calibration import run_9_point_calibration
from eyetrax.filters import make_kalman, KalmanEMASmoother
from eyetrax.gaze import GazeEstimator
from src.vision.gaze_state import GazeState

# Created once at module level — expensive to initialise per frame
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# Fixation threshold: gaze stable within 70px = fixating (relaxed for noisy gaze)
FIXATION_THRESHOLD_PX = 70
# Outlier rejection: skip jumps larger than this
MAX_JUMP_PX = 280


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
    Refine gaze using actual iris center detection on the eye region.
    Finds the darkest point (pupil center) in the isolated eye region.
    Returns (cx, cy) in frame coordinates, or None if detection fails.
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
    cx = x1 + max_loc[0]
    cy = y1 + max_loc[1]
    return cx, cy


class EyeTracker:
    def __init__(self):
        self.gaze_estimator = GazeEstimator()
        self.current_state = GazeState()

        # rolling blink log → BPM over last 60 s
        self._blink_log = collections.deque()
        self._prev_blink = False

        # outlier rejection: track last accepted position
        self._last_x = None
        self._last_y = None

        # fixation: track smoothed position history
        self._smooth_x = None
        self._smooth_y = None

        # small window for stable fixation
        self._fix_window = collections.deque(maxlen=3)

        self._set_smoother()

    # -----------------------------
    # MODEL
    # -----------------------------

    def create_model(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

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
        # KalmanEMA: smoother than plain Kalman, less lag than heavy averaging
        # Lower ema_alpha → stronger smoothing, less jitter
        self.smoother = KalmanEMASmoother(kf=make_kalman(), ema_alpha=0.12)

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
        self._fix_window.clear()
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

        # blink onset — rising edge only
        if blink_detected and not self._prev_blink:
            self._blink_log.append(time.time())
        self._prev_blink = blink_detected

        # drop blinks older than 60 s
        now = time.time()
        while self._blink_log and self._blink_log[0] < now - 60:
            self._blink_log.popleft()

        if features is not None and not blink_detected:
            try:
                gaze_point = self.gaze_estimator.predict(np.array([features]))[0]
                x, y = float(gaze_point[0]), float(gaze_point[1])
                print("GAZE:", x, y)

                if np.isnan(x) or np.isnan(y):
                    return s

                # iris refinement 
                if hasattr(self.gaze_estimator, "face_landmarks"):
                    iris = _detect_iris_center(frame, self.gaze_estimator.face_landmarks, [33, 133])
                    if iris is not None:
                        ix, iy = iris
                        # Heavier weight on iris center for stability
                        x = 0.3 * x + 0.7 * ix
                        y = 0.3 * y + 0.7 * iy

                # outlier rejection
                if self._last_x is not None:
                    dist = ((x - self._last_x)**2 + (y - self._last_y)**2) ** 0.5
                    if dist > MAX_JUMP_PX:
                        s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP * 0.5, 0.0)
                        return s

                self._last_x, self._last_y = x, y

                sx, sy = self.smoother.step(x, y)
                s.pred_x, s.pred_y = sx, sy
                s.cursor_alpha = min(s.cursor_alpha + CURSOR_STEP, 1.0)

                # stable fixation using 3-frame window
                self._fix_window.append((sx, sy))
                if len(self._fix_window) == 3:
                    xs = [p[0] for p in self._fix_window]
                    ys = [p[1] for p in self._fix_window]
                    spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    s.is_fixating = spread < FIXATION_THRESHOLD_PX
                else:
                    s.is_fixating = False

                self._smooth_x, self._smooth_y = sx, sy

            except Exception:
                s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP, 0.0)
                s.is_fixating = False
        else:
            s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP * 0.5, 0.0)
            s.is_fixating = False

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

            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                surf, (*COLORS[4][:3], alpha_int), (radius, radius), radius
            )

            if current_state.is_fixating:
                ring_col = (60, 220, 120, alpha_int // 2)
            else:
                ring_col = (220, 160, 40, alpha_int // 3)

            pygame.draw.circle(surf, ring_col, (radius, radius), radius, 3)
            screen.blit(surf, (cx - radius, cy - radius))

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

        bpm = len(self._blink_log)
        if current_state.blink_detected:
            blink_txt = f"Blinking   BPM: {bpm}"
            blink_clr = (80, 80, 240)
        else:
            blink_txt = f"Eyes open  BPM: {bpm}"
            blink_clr = (60, 210, 100)

        screen.blit(font.render(blink_txt, True, blink_clr), (50, 100))

        try:
            small = pygame.font.SysFont("Arial", 18)
            if current_state.is_fixating:
                fix_txt, fix_col = "Fixating", (60, 220, 120)
            else:
                fix_txt, fix_col = "Scanning", (220, 160, 40)
            screen.blit(small.render(fix_txt, True, fix_col), (50, 140))
        except Exception:
            pass
