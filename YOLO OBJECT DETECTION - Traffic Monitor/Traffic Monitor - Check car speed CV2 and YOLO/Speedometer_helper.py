import numpy as np
from Cam2WorldMapper import *
from collections import defaultdict
MPS_TO_KPH = 3.6


class Speedometer:
    """Estimates speed of objects in the world coordinates."""

    def __init__(self, mapper, fps, unit= MPS_TO_KPH):
        self._mapper = mapper
        self._fps = fps
        self._unit = unit
        self._speeds= defaultdict(list)

    @property
    def speeds(self):
        return self._speeds

    def update_with_trace(self, idx, image_trace):
        if len(image_trace) > 1:
            world_trace = self._mapper(image_trace)
            # Median displacement in x and y directions.
            # This stabilises after around several frames.
            dx, dy = np.median(np.abs(np.diff(world_trace, axis=0)), axis=0)
            ds = np.linalg.norm((dx, dy))
            self._speeds[idx].append(int(ds * self._fps * self._unit))

    def get_current_speed(self, idx: int):
        return self._speeds[idx][-1] if self._speeds[idx] else 0