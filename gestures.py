"""Gesture state machines: mouth clicks and blink Morse typing."""

MORSE = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
    "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k", ".-..": "l",
    "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q", ".-.": "r",
    "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w", "-..-": "x",
    "-.--": "y", "--..": "z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}
# ---- is unassigned in international Morse (it's CH in some variants),
# so it is safe to claim for backspace.
BACKSPACE_CODE = "----"


class EyeState:
    """Two-state eye classifier (OPEN / CLOSED) with hysteresis.

    Until calibrate() provides measured open/closed references it falls back
    to a rolling open-eye baseline; calibration gives much cleaner separation
    because absolute eye-aspect ratios vary per face and camera angle.
    """

    def __init__(self):
        self.open_ref = None
        self.closed_ref = None
        self._baseline = None
        self._low = None
        self.is_open = True

    @property
    def calibrated(self) -> bool:
        return self.open_ref is not None

    def calibrate(self, open_ref: float, closed_ref: float):
        self.open_ref = open_ref
        self.closed_ref = closed_ref

    def update(self, ear: float) -> bool:
        if self.calibrated:
            band = max(self.open_ref - self.closed_ref, 1e-6)
            close_thr = self.closed_ref + 0.40 * band
            open_thr = self.closed_ref + 0.60 * band
        else:
            if self._baseline is None:
                self._baseline = ear
            elif ear > 0.75 * self._baseline:
                # Only open-ish samples feed the baseline, so blinks
                # don't drag it down.
                self._baseline += 0.05 * (ear - self._baseline)
            else:
                # Learn where this face's closures sit: some eyes/cameras
                # give a closed EAR far above a fixed fraction of open.
                if self._low is None:
                    self._low = ear
                else:
                    self._low += 0.15 * (ear - self._low)
            self._baseline = max(self._baseline, 0.10)
            if self._low is not None and self._low < 0.85 * self._baseline:
                band = self._baseline - self._low
                close_thr = self._low + 0.45 * band
                open_thr = self._low + 0.60 * band
            else:
                close_thr = 0.55 * self._baseline
                open_thr = 0.70 * self._baseline
        if self.is_open and ear < close_thr:
            self.is_open = False
        elif not self.is_open and ear > open_thr:
            self.is_open = True
        return self.is_open


class MouthTaps:
    """Counts quick mouth open+close taps, and detects a sustained hold.

    Cursor mode maps 1 tap = left click, 2 = right click; typing mode maps
    1 = space. 3 taps toggles typing mode from either side and fires
    immediately; 1 and 2 fire once the window after the last close passes
    with no new open (so a right click waits out the chance of a third tap).
    Keeping the mouth open past hold_s fires HOLD once (voice dictation
    toggle) and voids the tap, so releasing it does not also click.
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
        """Drop any gesture in flight."""
        self._open = False
        self._taps = 0
        self._hold_fired = False

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


class MorseTyper:
    """Deliberate blinks type Morse: short = dot, long = dash.

    A pause ends the letter and decodes it; BACKSPACE_CODE (four dashes,
    not a letter in international Morse) is backspace. Space is a mouth
    tap: the app calls flush() then types a space.
    """

    def __init__(self, cfg):
        self.min_blink_s = cfg.min_blink_s
        self.dot_max_s = cfg.dot_max_s
        self.letter_gap_s = cfg.letter_gap_s
        self.merge_gap_s = cfg.merge_gap_s
        self.code = ""          # dots/dashes of the letter in progress (for HUD)
        self.last_decoded = ""  # for HUD
        self._closed_since = None
        self._open_since = 0.0
        self._pending = None    # closure duration awaiting confirmation
        self._last_blink = None

    def reset(self):
        self.code = ""
        self.last_decoded = ""
        self._closed_since = None
        self._pending = None
        self._last_blink = None

    def _decode(self):
        code, self.code = self.code, ""
        if not code:
            return None
        if code == BACKSPACE_CODE:
            self.last_decoded = "BKSP"
            return ("backspace",)
        ch = MORSE.get(code)
        if ch is None:
            self.last_decoded = "?" + code
            return ("invalid", code)
        self.last_decoded = ch
        return ("char", ch)

    def flush(self):
        """End the letter in progress now (a space gesture interrupts it)."""
        self._closed_since = None
        if self._pending is not None:
            # A completed closure awaiting the merge window is a real
            # symbol; commit it rather than dropping it.
            dur, self._pending = self._pending, None
            if dur >= self.min_blink_s:
                self.code += "." if dur <= self.dot_max_s else "-"
        return self._decode()

    def update(self, eyes_open: bool, t: float):
        """Feed eye state each frame.

        Returns None, ("char", c), ("backspace",) or ("invalid", code).
        """
        if not eyes_open:
            if self._closed_since is None:
                if self._pending is not None and t - self._open_since <= self.merge_gap_s:
                    # The tracker flickered "open" for a frame mid-hold;
                    # resume the closure so a held blink stays one dash.
                    self._closed_since = t - self._pending
                else:
                    self._closed_since = t
                self._pending = None
            return None

        if self._closed_since is not None:
            self._pending = t - self._closed_since
            self._closed_since = None
            self._open_since = t
            return None

        if self._pending is not None:
            if t - self._open_since <= self.merge_gap_s:
                return None  # the closure may still resume
            dur, self._pending = self._pending, None
            if dur >= self.min_blink_s:
                self.code += "." if dur <= self.dot_max_s else "-"
                self._last_blink = t

        if self._last_blink is None:
            return None
        if self.code and t - self._last_blink > self.letter_gap_s:
            return self._decode()
        return None
