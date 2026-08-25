"""Moves the Windows cursor with One Euro smoothing and a hold/track clutch.

Absolute control with a clutch: while parked, the physical mouse works
normally and head movement is ignored; a clear, sustained head move engages
tracking, and the cursor then follows the absolute target position (where
the nose points). Stopping briefly parks it again.
"""

import ctypes
import math
from ctypes import wintypes


def _enable_dpi_awareness():
    # Without this, Windows display scaling virtualizes coordinates and the
    # reported screen size does not match real pixels.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class OneEuroFilter:
    """Adaptive low-pass filter: steady when still, responsive when moving fast."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t = None
        self._x = 0.0
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._t = None

    def __call__(self, x: float, t: float) -> float:
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        return self._x


class CursorController:
    MAX_STEP_PX = 150   # per-frame cap so a landmark glitch cannot teleport the cursor
    ENGAGE_FRAMES = 3   # consecutive frames beyond engage_px needed to start moving
    PARK_REF_LEAK_S = 6.0  # slow posture drift is absorbed instead of accumulating

    def __init__(self, dry_run: bool = False):
        _enable_dpi_awareness()
        user32 = ctypes.windll.user32
        self.screen_w = int(user32.GetSystemMetrics(0))
        self.screen_h = int(user32.GetSystemMetrics(1))
        self.dry_run = dry_run
        self._fx = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        self._fy = OneEuroFilter(min_cutoff=1.0, beta=0.01)
        self.engage_px = 60.0
        self.track_slack_px = 10.0
        self.park_speed = 30.0
        self.park_time_s = 0.30
        self.tracking = False
        self._pos: tuple[float, float] | None = None
        self._prev_target: tuple[float, float, float] | None = None
        self._park_ref: tuple[float, float] | None = None
        self._speed = 0.0
        self._engage_count = 0
        self._still_since: float | None = None

    def set_tuning(self, cfg):
        """cfg provides min_cutoff, beta, engage_px, track_slack_px, park_speed, park_time_s."""
        for f in (self._fx, self._fy):
            f.min_cutoff = cfg.min_cutoff
            f.beta = cfg.beta
        self.engage_px = cfg.engage_px
        self.track_slack_px = cfg.track_slack_px
        self.park_speed = cfg.park_speed
        self.park_time_s = cfg.park_time_s

    def reset(self):
        self._fx.reset()
        self._fy.reset()
        self.tracking = False
        self._pos = None
        self._prev_target = None
        self._park_ref = None
        self._speed = 0.0
        self._engage_count = 0
        self._still_since = None

    def _actual_cursor(self) -> tuple[float, float]:
        p = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(p))
        return float(p.x), float(p.y)

    def move_to(self, x: float, y: float, t: float) -> tuple[int, int]:
        sx = self._fx(x, t)
        sy = self._fy(y, t)

        dt = 0.0
        dtx = dty = 0.0
        if self._prev_target is not None:
            px_, py_, pt_ = self._prev_target
            dt = max(t - pt_, 1e-3)
            dtx, dty = sx - px_, sy - py_
            self._speed = 0.7 * self._speed + 0.3 * (math.hypot(dtx, dty) / dt)
        self._prev_target = (sx, sy, t)

        if self._pos is None:
            self._pos = (self.screen_w / 2.0, self.screen_h / 2.0) if self.dry_run \
                else self._actual_cursor()
        if self._park_ref is None:
            self._park_ref = (sx, sy)

        if not self.tracking:
            # Parked: leave the cursor alone (the physical mouse stays usable)
            # and resume from wherever it actually is.
            if not self.dry_run:
                self._pos = self._actual_cursor()
            rx, ry = self._park_ref
            if math.hypot(sx - rx, sy - ry) > self.engage_px:
                self._engage_count += 1
                if self._engage_count >= self.ENGAGE_FRAMES:
                    self.tracking = True
                    self._engage_count = 0
                    self._still_since = None
            else:
                self._engage_count = 0
                if dt > 0:
                    # Leak the reference toward the target so slow posture
                    # drift never accumulates into a spurious grab.
                    k = min(dt / self.PARK_REF_LEAK_S, 1.0)
                    self._park_ref = (rx + (sx - rx) * k, ry + (sy - ry) * k)
        else:
            # Follow the absolute target with a short rope so active motion
            # feels direct but sub-slack wiggle is ignored.
            dxp, dyp = sx - self._pos[0], sy - self._pos[1]
            dist = math.hypot(dxp, dyp)
            if dist > self.track_slack_px:
                pull = (dist - self.track_slack_px) / dist
                step_x = max(-self.MAX_STEP_PX, min(self.MAX_STEP_PX, dxp * pull))
                step_y = max(-self.MAX_STEP_PX, min(self.MAX_STEP_PX, dyp * pull))
                nx = max(0.0, min(self.screen_w - 1.0, self._pos[0] + step_x))
                ny = max(0.0, min(self.screen_h - 1.0, self._pos[1] + step_y))
                self._pos = (nx, ny)
                if not self.dry_run:
                    ctypes.windll.user32.SetCursorPos(int(nx), int(ny))
            # Park again once the target has been still long enough.
            if self._speed < self.park_speed:
                if self._still_since is None:
                    self._still_since = t
                elif t - self._still_since >= self.park_time_s:
                    self.tracking = False
                    self._park_ref = (sx, sy)
            else:
                self._still_since = None

        return int(self._pos[0]), int(self._pos[1])
