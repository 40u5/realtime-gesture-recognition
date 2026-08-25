"""Tuning knobs for head-tracking mouse control."""

from dataclasses import dataclass


@dataclass
class Config:
    gain_x: float = 10.5        # screen widths of cursor travel per frame-width of nose travel
    gain_y: float = 14.0
    min_cutoff: float = 0.6     # One Euro smoothing floor (lower = steadier but laggier)
    beta: float = 0.008
    engage_px: float = 60       # target pull needed to unpark the cursor
    track_slack_px: float = 10  # rope slack while actively moving
    park_speed: float = 30      # px/s below which the target counts as "still"
    park_time_s: float = 0.30   # how long it must stay still before parking again

    # Mouth taps (ratio = lip gap / nose-to-upper-lip distance)
    # 1 tap = left click / space, 2 = right click, 3 = typing-mode toggle
    mouth_open_ratio: float = 0.55   # above this counts as "open"
    mouth_close_ratio: float = 0.30  # must drop back below this to finish a tap
    tap_window_s: float = 0.6        # gap after a close that ends the tap count

    # Blink Morse typing
    min_blink_s: float = 0.06   # closures shorter than this are ignored
    dot_max_s: float = 0.35     # closure up to this = dot, longer = dash
    merge_gap_s: float = 0.12   # re-closure within this joins the previous closure
    letter_gap_s: float = 0.8   # eyes-open pause that ends a letter


CONFIG = Config()
