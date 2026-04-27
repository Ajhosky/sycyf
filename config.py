"""
Project-wide constants for the Define-stage reference model.
"""

FRAME_SIZE = 8
NUM_CLASSES = 4
SEQUENCE_LENGTH = 6

FORWARD = 0
LEFT = 1
RIGHT = 2
BACK = 3

LABEL_NAMES = {
    FORWARD: "FORWARD",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    BACK: "BACK",
}

LABEL_TO_ID = {name: idx for idx, name in LABEL_NAMES.items()}

DEFAULT_SEED = 17
