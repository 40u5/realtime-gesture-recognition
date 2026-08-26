# Head Tracking Mouse

Control the Windows mouse cursor with small head turns, using a webcam,
MediaPipe Face Mesh, and the Win32 cursor API. Built so a computer can be
driven entirely hands-free (e.g., by amputees): head points, mouth taps
click and switch modes, voice dictation types.

A pure eye-gaze mode was prototyped and removed: webcam iris tracking is too
noisy for cursor control (eye movement wobbles the whole face mesh). Head
pose is the signal that works.

## Demo

[▶ Watch the demo](demo.mp4) — head-pointing, mouth-tap clicks, and voice
dictation in action (GitHub plays it when clicked).

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

- **~2.5 s of silence** → dictation turns itself off automatically (falling
  beep; a phrase in flight is still typed). You get ~5 s to start speaking
  after the toggle. Tune with `voice_silence_s` / `voice_start_grace_s` in
  `config.py`.
- **2 mouth taps** → backspace.
- **3 mouth taps** → back to cursor mode (hold ~1 s or **V** also exit,
  with a falling beep).
- Taps never click while dictating. Fast talking can register as taps —
  pause your speech for a beat before deliberate tap gestures.

The small English model (~40 MB) is downloaded to `models/` automatically
the first time; the HUD shows download/load progress. Output is lowercase
words without punctuation.

## Controls

| Control | Action |
| --- | --- |
| SPACE | enable / pause head control; enabling captures neutral (window must have focus — Alt-Tab to it) |
| C | re-capture neutral at the current head pose |
| hold mouth open ~1 s | toggle voice dictation (V on the preview window also works) |
| mouth taps | cursor mode: 1 = left click, 2 = right click; voice mode: 2 = backspace, 3 = back to cursor |
| Sens X % / Sens Y % sliders | per-axis sensitivity ([ / ] move both) |
| Q / ESC | quit |

## How it works

- Face Mesh yields 468 landmarks; the signal is the raw nose position in the
  frame (an averaged 3-landmark cluster — the most stable part of the mesh).
  No yaw/pitch geometry: earlier proxies divided by noisy face-silhouette
  landmarks, which added jitter.
- Nose offset from the captured neutral maps *absolutely* to a screen
  position (gain × sensitivity), smoothed by a One Euro filter and gated by
  a hold/track clutch: parked until the target pulls past a threshold for
  several consecutive frames (slow posture drift leaks away instead of
  accumulating), tracking while you move, parked again once you stop. While
  tracking the cursor follows the target on a short rope, with a per-frame
  step cap against landmark glitches.

## Notes / limits

- Primary monitor only.
- Tuning lives in `config.py` (gains, smoothing, clutch thresholds, voice
  silence timeout).
- Clicks and keystrokes are injected with Win32 `SendInput`
  (`win_input.py`); the mouth-tap state machine (open/close hysteresis +
  hold detection) lives in `gestures.py`.
- Voice dictation (`voice.py`) streams mic audio into Vosk on a
  background thread; the tracking loop just polls for finished phrases.

## Ideas next

Dwell-to-click, scroll gestures, spoken punctuation/commands ("new line",
"delete that") in voice mode.
