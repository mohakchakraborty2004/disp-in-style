import numpy as np


def hand(ratio: float) -> np.ndarray:
    """Synthetic 21-landmark hand with palm size 1 and thumb-middle pinch ratio `ratio`."""
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[0] = (0.0, 0.0)  # wrist
    lm[9] = (0.0, -1.0)  # middle MCP -> palm size 1
    lm[4] = (0.0, -1.5)  # thumb tip
    lm[12] = (ratio, -1.5)  # middle tip
    return lm
