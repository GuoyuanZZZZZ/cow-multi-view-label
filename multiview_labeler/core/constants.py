"""Shared constants for multi-view labeling."""

KEYPOINTS = [
    "nose",
    "neck",
    "spine",
    "tail_base",
    "lf_leg",
    "rf_leg",
    "lb_leg",
    "rb_leg",
]

SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (1, 4),
    (1, 5),
    (2, 6),
    (2, 7),
]

LEFT_RIGHT_PAIRS = [(4, 5), (6, 7)]
VISIBILITY_STATES = ["visible", "occluded", "absent"]
VISIBILITY_COLORS = {
    "visible": (0, 255, 0),
    "occluded": (0, 165, 255),
    "absent": (120, 120, 120),
}
