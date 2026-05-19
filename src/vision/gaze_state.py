from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class GazeState:
    """
    Stores current gaze estimation state.
    """
    pred_x: Optional[float] = None
    pred_y: Optional[float] = None
    blink_detected: Optional[bool] = None
    cursor_alpha: float = 0.0
    contours: List = field(default_factory=list)  # only for kde filter

    # True when gaze has been stable in one spot — used for fixation ring
    is_fixating: bool = False
