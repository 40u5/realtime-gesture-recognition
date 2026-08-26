# Head Tracking Mouse

Control the Windows mouse cursor with small head turns, using a webcam,
MediaPipe Face Mesh, and the Win32 cursor API. Built so a computer can be
driven entirely hands-free (e.g., by amputees): head points, mouth taps
click and switch modes, voice dictates text (blink Morse is the silent
fallback for users who can't speak or can't be overheard).

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
- **Hold mouth open ~1 s** → toggle voice dictation (see below).
- The cursor freezes while a tap sequence is in flight so the click lands
  where you aimed. (Talking will trigger clicks — pause with SPACE first
  or use voice dictation mode, which suppresses them.)

## Typing (voice dictation)

**Hold your mouth open for about a second** (rising double-beep confirms;
**V** on the preview window also works) to start dictating, then just talk.
Speech is recognized locally with [Vosk](https://alphacephei.com/vosk/) —
offline, no audio leaves the machine — and each finished phrase is typed
into whichever window has focus, followed by a space. The HUD shows the
phrase being recognized live (`hearing: ...`) and echoes everything typed.
The microphone is only captured while dictation is on. While dictating:

- **2 mouth taps** → backspace (same gesture as in Morse mode).
- **3 mouth taps** → back to cursor mode (hold ~1 s or **V** also exit,
  with a falling beep).
- Taps never click while dictating. Fast talking can register as taps —
  pause your speech for a beat before deliberate tap gestures.

The small English model (~40 MB) is downloaded to `models/` automatically
the first time; the HUD shows download/load progress. Output is lowercase
words without punctuation — Morse remains the way to type digits or
precise characters, and is the fallback when speaking isn't possible.

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
- **2 mouth taps** → backspace (mid-letter it scraps the dots/dashes
  entered so far instead — the HUD flashes `[ CODE CLEARED ]`).
- **Four dashes** (`----`) also = backspace — deliberately not a letter in
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
| hold mouth open ~1 s | toggle voice dictation (V on the preview window also works) |
| 3 mouth taps | toggle blink-Morse typing mode (starts with a short eye calibration) |
| Right Ctrl / F7 (global) or T (preview) | keyboard fallback for the Morse typing toggle |
| mouth taps | cursor mode: 1 = left click, 2 = right click; Morse mode: 1 = space, 2 = backspace; voice mode: 2 = backspace, 3 = back to cursor |
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
  (mouth-tap counting with open/close hysteresis + hold detection,
  blink-duration Morse decoding, two-state eye classifier with
  calibration + adaptive fallback).
- Voice dictation (`voice.py`) streams mic audio into Vosk on a
  background thread; the tracking loop just polls for finished phrases.

## Ideas next

Dwell-to-click, scroll gestures, Enter/punctuation Morse prosigns,
spoken punctuation/commands ("new line", "delete that") in voice mode.
