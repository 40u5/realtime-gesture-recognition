"""Gesture state machines: mouth-tap clicks and hold detection."""


class MouthTaps:
    """Counts quick mouth open+close taps, and detects a sustained hold.

    Cursor mode maps 1 tap = left click, 2 = right click; voice mode maps
    2 = backspace, 3 = back to cursor. 3 taps fires immediately; 1 and 2
    fire once the window after the last close passes with no new open (so
    a right click waits out the chance of a third tap). Keeping the mouth
    open past hold_s fires HOLD once (voice dictation toggle) and voids
    the tap, so releasing it does not also click.
    """

    HOLD = -1

    def __init__(self, cfg):
        self.open_ratio = cfg.mouth_open_ratio
        self.close_ratio = cfg.mouth_close_ratio
        self.window_s = cfg.tap_window_s
        self.hold_s = cfg.mouth_hold_s
        self._open = False
        self._taps = 0
        self._last_close = 0.0
        self._open_since = 0.0
        self._hold_fired = False

    def cancel(self):
        """Drop any taps in flight; if the mouth is mid-open, swallow the
        rest of that open. Resetting _open here instead would make the
        still-open mouth register as a fresh open, so the hold that just
        toggled voice mode would re-fire 1s later and toggle it right back.
        """
        self._taps = 0
        if self._open:
            self._hold_fired = True

    @property
    def mouth_open(self) -> bool:
        return self._open

    @property
    def engaged(self) -> bool:
        """True while a gesture is in flight; hold the cursor so a click
        lands where it was aimed."""
        return self._open or self._taps > 0

    def update(self, ratio: float, t: float) -> int:
        """Feed the mouth ratio each frame; returns the completed tap count
        (1..3), HOLD, or 0."""
        # Expire the window before reading this frame's ratio, so a stale
        # pending tap can never merge into a new gesture.
        if self._taps and not self._open and t - self._last_close > self.window_s:
            n, self._taps = self._taps, 0
            return n
        if not self._open and ratio > self.open_ratio:
            self._open = True
            self._open_since = t
            self._hold_fired = False
        elif self._open:
            if ratio < self.close_ratio:
                self._open = False
                if self._hold_fired:
                    return 0
                self._taps += 1
                self._last_close = t
                if self._taps >= 3:
                    self._taps = 0
                    return 3
            elif not self._hold_fired and t - self._open_since >= self.hold_s:
                self._hold_fired = True
                self._taps = 0  # a hold supersedes any taps in flight
                return self.HOLD
        return 0
