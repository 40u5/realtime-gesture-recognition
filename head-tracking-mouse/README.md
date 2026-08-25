# Head Tracking Mouse

Control the Windows mouse cursor with small head turns, using a webcam,
MediaPipe Face Mesh, and the Win32 cursor API. Built so a computer can be
driven entirely hands-free (e.g., by amputees): head points, mouth taps
click and switch modes, blinks type.

A pure eye-gaze mode was prototyped and removed: webcam iris tracking is too
noisy for cursor control (eye movement wobbles the whole face mesh). Head
pose is the signal that works.

## Setup

```
pip install -r requirements.txt
```

Python 3.10–3.12 recommended (MediaPipe wheel availability).

## Run

```
python main.py [--camera 0] [--dry-run]
```

`--dry-run` tracks and shows the preview without moving the real cursor.

The app starts **paused**. Face the screen comfortably and press SPACE —
that pose becomes your **neutral**, mapped to the screen center. Control is
absolute: your nose points at a screen position like a laser pointer.

## How to drive it

- **Hold your head still** → the cursor is parked. Your physical mouse works
  normally, and eye movement / blinks / small jitters do nothing.
- **Turn your head clearly** → after ~0.1 s the head "grabs" the cursor,
  which glides to where your nose points and follows it.
- **Stop moving** → after ~0.3 s it parks again and freezes.
- Press **C** any time to re-capture neutral (your current pose becomes
  screen center). The hollow yellow circle on the preview marks neutral.

Use the **Sens X %** and **Sens Y %** sliders on the preview window to set
how far a head turn moves the cursor on each axis (`[` / `]` move both
together).

## Clicking (mouth taps)

A "tap" is one quick open+close of the mouth. In cursor mode:

- **1 tap** → left click.
- **2 taps quickly** → right click (it fires after a short beat — the app
  waits to see whether a third tap is coming).
- **3 taps quickly** → toggle blink-Morse typing mode.
- The cursor freezes while a tap sequence is in flight so the click lands
  where you aimed. (Talking will trigger clicks — pause with SPACE first.)

## Typing (blink Morse code)

**Tap your mouth 3 times quickly** to toggle typing mode — fully
hands-free. (Keyboard fallbacks: **Right Ctrl** or **F7** from any window —
laptop F-keys may need Fn held — or **T** on the preview window.) It works
even while head control is paused. The HUD's `Type:` indicator confirms the
toggle. Entering typing mode runs a short eye calibration — after a brief
get-ready pause it measures your open and closed eyes. It is
audio-guided so you can follow it with your eyes shut: **high beep** =
keep eyes open, **low beep** = close your eyes, **rising double-beep** =
done, open them (a low buzz means it failed and adaptive detection is
used instead). Then deliberate blinks type Morse
code. Keystrokes go to whichever window has focus — click into a text box
first. The HUD echoes everything typed either way:

- **Short blink** (≤ ~0.35 s) = dot, **long blink** = dash.
- **Pause ~0.8 s** with eyes open → the letter is decoded and typed.
- **1 mouth tap** → space (any letter in progress is typed first).
  Mouth clicks only apply in cursor mode; **3 taps** exits typing mode.
- **Four dashes** (`----`) = backspace — deliberately not a letter in
  international Morse, so it cannot collide with real text.

The HUD shows the dots/dashes entered so far and the last decoded
character. Letters a–z and digits 0–9 are supported (standard Morse).
Space and backspace are invisible, so the HUD flashes `[ SPACE ]` /
`[ BACKSPACE ]` when they fire (and `[ ? code ]` for an unknown code),
and spaces appear as `_` in the typed echo.

## Controls

| Control | Action |
| --- | --- |
| SPACE | enable / pause head control; enabling captures neutral (window must have focus — Alt-Tab to it) |
| C | re-capture neutral at the current head pose |
| 3 mouth taps | toggle blink-Morse typing mode (starts with a short eye calibration) |
| Right Ctrl / F7 (global) or T (preview) | keyboard fallback for the typing toggle |
| mouth taps | cursor mode: 1 = left click, 2 = right click; typing mode: 1 = space |
| Sens X % / Sens Y % sliders | per-axis sensitivity ([ / ] move both) |
| Q / ESC | quit |

## How it works

- Face Mesh yields 468 landmarks; the signal is the raw nose position in the
  frame (an averaged 3-landmark cluster — the most stable part of the mesh).
  No yaw/pitch geometry: earlier proxies divided by noisy face-silhouette
  landmarks, which added jitter. Blinks freeze the cursor because they
  wobble the whole mesh.
- Nose offset from the captured neutral maps *absolutely* to a screen
  position (gain × sensitivity), smoothed by a One Euro filter and gated by
  a hold/track clutch: parked until the target pulls past a threshold for
  several consecutive frames (slow posture drift leaks away instead of
  accumulating), tracking while you move, parked again once you stop. While
  tracking the cursor follows the target on a short rope, with a per-frame
  step cap against landmark glitches.

## Notes / limits

- Primary monitor only.
- Tuning lives in `config.py` (gains, smoothing, clutch thresholds) and
  `face_tracker.py` (blink threshold).

- Clicks and keystrokes are injected with Win32 `SendInput`
  (`win_input.py`); gesture state machines live in `gestures.py`
  (mouth-tap counting with open/close hysteresis, blink-duration Morse
  decoding, two-state eye classifier with calibration + adaptive fallback).

## Ideas next

Dwell-to-click, scroll gestures, Enter/punctuation Morse prosigns.
