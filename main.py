"""Head Tracking Mouse

Moves the Windows cursor with small head turns, using a webcam and
MediaPipe Face Mesh. Mouth gestures click; voice dictation types
(blink Morse code is the silent fallback).
"""

import argparse
import ctypes
import sys
import threading
import time
import winsound
from statistics import median

import cv2

import win_input
from config import CONFIG
from cursor_controller import CursorController
from face_tracker import FaceTracker
from gestures import EyeState, MorseTyper, MouthTaps
from voice import VoiceTyper

WINDOW_NAME = "Head Tracking Mouse"
SENS_X_TRACKBAR = "Sens X %"
SENS_Y_TRACKBAR = "Sens Y %"
FACE_LOST_RESET_S = 0.5
VK_F7 = 0x76        # may not reach us on laptops where F-keys default to media functions
VK_RCONTROL = 0xA3  # always a real key, never lands in typed text


class GlobalKey:
    """Focus-independent key press detection via GetAsyncKeyState."""

    def __init__(self, vk: int):
        self.vk = vk
        self._down = False

    def pressed(self) -> bool:
        down = bool(ctypes.windll.user32.GetAsyncKeyState(self.vk) & 0x8000)
        edge = down and not self._down
        self._down = down
        return edge


def parse_args():
    parser = argparse.ArgumentParser(description="Control the mouse with your head.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="track and preview without moving the cursor")
    return parser.parse_args()


def beep(*notes):
    """Play (freq_hz, dur_ms) notes without blocking the tracking loop.

    Calibration is audio-guided because the user cannot watch the HUD
    countdown with their eyes closed.
    """
    def _play():
        for freq, dur in notes:
            winsound.Beep(freq, dur)
    threading.Thread(target=_play, daemon=True).start()


def put_text(frame, text, pos, color=(255, 255, 255), scale=0.55):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main() -> int:
    args = parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    tracker = FaceTracker()
    controller = CursorController(dry_run=args.dry_run)
    controller.set_tuning(CONFIG)
    tapper = MouthTaps(CONFIG)
    typer = MorseTyper(CONFIG)
    voice = VoiceTyper()
    eye_state = EyeState()

    active = False
    typing_mode = False
    voice_mode = False
    typed = ""  # everything typed this session, echoed on the HUD
    flash_text = ""   # transient HUD notice for invisible characters
    flash_until = 0.0
    last_ear_log = 0.0
    cal_phase = None  # None | "open" | "closed" (eye calibration on entering typing)
    cal_until = 0.0
    cal_samples = []
    cal_open = 0.0
    neutral = None  # nose position (normalized) that maps to screen center
    last_face_time = 0.0
    fps = 0.0
    prev_t = time.monotonic()

    print(f"Screen: {controller.screen_w}x{controller.screen_h}"
          + (" (dry run: cursor will not move)" if args.dry_run else ""))
    print("SPACE start/pause | C re-center | 3 mouth taps (or Right-Ctrl/F7) Morse mode | Q quit")
    print("Mouth taps: 1 = left click, 2 = right click, 3 = Morse typing mode")
    print("Voice: hold mouth open ~1s (or V) to dictate - speech is typed; "
          "2 taps = backspace, 3 taps = back to cursor")
    print("Morse mode: short blink = dot, long = dash, pause = end letter, "
          "1 tap = space, 2 taps (or ----) = backspace, 3 taps = exit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.createTrackbar(SENS_X_TRACKBAR, WINDOW_NAME, 100, 400, lambda v: None)
    cv2.createTrackbar(SENS_Y_TRACKBAR, WINDOW_NAME, 100, 400, lambda v: None)
    typing_keys = (GlobalKey(VK_F7), GlobalKey(VK_RCONTROL))

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed", file=sys.stderr)
                return 1
            # Mirror so cursor motion matches the user's own sense of direction.
            frame = cv2.flip(frame, 1)

            now = time.monotonic()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            sens_x = max(cv2.getTrackbarPos(SENS_X_TRACKBAR, WINDOW_NAME), 5) / 100.0
            sens_y = max(cv2.getTrackbarPos(SENS_Y_TRACKBAR, WINDOW_NAME), 5) / 100.0
            obs = tracker.process(frame)
            pointer = None
            status, status_color = "PAUSED - press SPACE", (0, 200, 255)

            if obs is not None:
                last_face_time = now
            eyes_open = eye_state.update(obs.min_ear) if obs is not None else True

            # The tap counter runs in every mode so 3 taps can always toggle
            # typing; what 1 and 2 taps mean depends on the mode below.
            taps = tapper.update(obs.mouth_ratio, now) if obs is not None else 0

            if active and not typing_mode and not voice_mode and obs is not None:
                if taps == 1:
                    print("[dry run] left click") if args.dry_run else win_input.left_click()
                elif taps == 2:
                    print("[dry run] right click") if args.dry_run else win_input.right_click()

            # Drain finished phrases every frame (not only in voice mode) so
            # the last phrase still lands after dictation is toggled off.
            for text in voice.poll():
                typed += text + " "
                if args.dry_run:
                    print(f"[dry run] type {text + ' '!r}")
                else:
                    win_input.type_text(text + " ")

            if voice_mode:
                if taps == 2:
                    typed = typed[:-1]
                    flash_text, flash_until = "[ BACKSPACE ]", now + 1.2
                    print("[dry run] backspace") if args.dry_run \
                        else win_input.backspace()
                st = voice.state
                if st == "listening":
                    status, status_color = "VOICE - listening", (0, 255, 0)
                elif st.startswith("error"):
                    status, status_color = "VOICE " + st, (0, 0, 255)
                else:
                    status, status_color = f"VOICE - {st}", (255, 200, 0)
            elif typing_mode:
                # Typing works even while head control is paused. Keystrokes
                # land in whichever window has focus; the HUD echoes them.
                if obs is None:
                    status, status_color = "TYPING - NO FACE", (0, 0, 255)
                elif cal_phase == "wait":
                    # Give the user a beat to settle after the toggle gesture
                    # before measuring anything.
                    status, status_color = "EYE CAL: get ready...", (255, 200, 0)
                    if now >= cal_until:
                        cal_phase, cal_until, cal_samples = "open", now + 1.2, []
                        beep((880, 120))  # high beep: eyes-open phase
                elif cal_phase is not None:
                    remain = cal_until - now
                    # Skip each phase's early frames: reacting to the beep
                    # takes a beat, and early samples blend the previous eye
                    # state into the median (seen as cal refs landing between
                    # true open and closed).
                    if (0.7 if cal_phase == "open" else 1.0) >= remain > 0:
                        cal_samples.append(obs.min_ear)
                    if cal_phase == "open":
                        status = f"EYE CAL 1/2: look normal, eyes OPEN ({max(remain, 0):.1f}s)"
                        status_color = (255, 200, 0)
                        if remain <= 0:
                            if len(cal_samples) >= 5:
                                cal_open = median(cal_samples)
                                cal_phase, cal_until, cal_samples = "closed", now + 1.8, []
                                beep((500, 150))  # low beep: close your eyes now
                            else:
                                cal_until, cal_samples = now + 1.2, []
                    else:
                        status = f"EYE CAL 2/2: CLOSE both eyes ({max(remain, 0):.1f}s)"
                        status_color = (0, 200, 255)
                        if remain <= 0:
                            if len(cal_samples) >= 5:
                                cal_closed = median(cal_samples)
                                if cal_closed > 0.8 * cal_open:
                                    # Bad refs make the classifier flicker and
                                    # type garbage; the fallback is safer.
                                    beep((250, 350))  # buzz: calibration failed
                                    print(f"eye cal FAILED (open={cal_open:.3f} "
                                          f"closed={cal_closed:.3f} - eyes never "
                                          "closed?); using adaptive fallback",
                                          flush=True)
                                else:
                                    beep((880, 90), (1175, 130))  # done, open eyes
                                    eye_state.calibrate(cal_open, cal_closed)
                                    print(f"eye cal: open={cal_open:.3f} "
                                          f"closed={cal_closed:.3f}", flush=True)
                                cal_phase = None
                                typer.reset()
                            else:
                                cal_until, cal_samples = now + 1.8, []
                else:
                    if now - last_ear_log > 2.0:
                        print(f"[typing] ear={obs.min_ear:.3f} open={eyes_open} "
                              f"code={typer.code!r}", flush=True)
                        last_ear_log = now
                    events = []
                    if taps == 1:
                        # Mouth tap = space; finish the letter in progress
                        # first so "a" + tap comes out as "a ".
                        fl = typer.flush()
                        if fl is not None:
                            events.append(fl)
                        events.append(("space",))
                    elif taps == 2:
                        if typer.code:
                            # Mid-letter, backspace means "scrap this letter",
                            # not "delete already-typed text".
                            typer.code = ""
                            flash_text, flash_until = "[ CODE CLEARED ]", now + 1.2
                        else:
                            events.append(("backspace",))
                    elif not tapper.mouth_open:
                        # While the mouth is open the mesh distorts; don't
                        # count blinks until it closes again.
                        ev = typer.update(eyes_open, now)
                        if ev is not None:
                            events.append(ev)
                    for ev in events:
                        kind = ev[0]
                        if kind == "char":
                            typed += ev[1]
                            print(f"[dry run] type {ev[1]!r}") if args.dry_run \
                                else win_input.type_char(ev[1])
                        elif kind == "space":
                            typed += " "
                            flash_text, flash_until = "[ SPACE ]", now + 1.2
                            print("[dry run] type ' '") if args.dry_run \
                                else win_input.type_char(" ")
                        elif kind == "backspace":
                            typed = typed[:-1]
                            flash_text, flash_until = "[ BACKSPACE ]", now + 1.2
                            print("[dry run] backspace") if args.dry_run \
                                else win_input.backspace()
                        elif kind == "invalid":
                            flash_text, flash_until = f"[ ? {ev[1]} ]", now + 1.2
                    status, status_color = "TYPING - blink Morse (T exits)", (255, 200, 0)
            elif active:
                if obs is None:
                    status, status_color = "NO FACE", (0, 0, 255)
                    if now - last_face_time > FACE_LOST_RESET_S:
                        controller.reset()
                elif not eyes_open:
                    # Blinks wobble the whole mesh; hold position.
                    status, status_color = "BLINK HOLD", (0, 200, 255)
                elif tapper.engaged:
                    # Freeze the cursor so the click lands where it was aimed.
                    status, status_color = "MOUTH TAPS", (255, 0, 255)
                else:
                    tx = controller.screen_w / 2 + (obs.nose_x - neutral[0]) \
                        * CONFIG.gain_x * sens_x * controller.screen_w
                    ty = controller.screen_h / 2 + (obs.nose_y - neutral[1]) \
                        * CONFIG.gain_y * sens_y * controller.screen_h
                    pointer = controller.move_to(tx, ty, now)
                    if controller.tracking:
                        status = "TRACKING (dry run)" if args.dry_run else "TRACKING"
                        status_color = (0, 255, 0)
                    else:
                        status, status_color = "HOLDING - move head clearly to grab", (0, 255, 255)

            if neutral is not None:
                fh, fw = frame.shape[:2]
                cv2.circle(frame, (int(neutral[0] * fw), int(neutral[1] * fh)),
                           7, (255, 255, 0), 1)
            if obs is not None:
                for name, (mx, my) in obs.markers.items():
                    color = (255, 0, 255) if name == "nose" else (0, 255, 255)
                    cv2.circle(frame, (mx, my), 3, color, -1)
            if pointer is not None:
                fh, fw = frame.shape[:2]
                cx = int(pointer[0] / controller.screen_w * fw)
                cy = int(pointer[1] / controller.screen_h * fh)
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

            put_text(frame, status, (10, 25), status_color, 0.65)
            face_txt = (f"   Mouth: {obs.mouth_ratio:.2f}"
                        f"   Eye: {obs.min_ear:.2f} {'OPEN' if eyes_open else 'SHUT'}") \
                if obs is not None else ""
            type_label = "MORSE" if typing_mode else "VOICE" if voice_mode else "OFF"
            put_text(frame, f"Type: {type_label}   "
                     f"Sens X: {sens_x:.2f}x  Y: {sens_y:.2f}x   FPS: {fps:.0f}"
                     + face_txt, (10, 50))
            if voice_mode:
                put_text(frame, "Typed: " + typed[-40:].replace(" ", "_"),
                         (10, 75), (255, 200, 0))
                if voice.partial:
                    put_text(frame, "hearing: " + voice.partial[-45:],
                             (10, 100), (0, 255, 255))
                if now < flash_until:
                    put_text(frame, flash_text, (10, 130), (0, 255, 255), 0.8)
                fh = frame.shape[0]
                put_text(frame, "speak normally - each finished phrase is typed"
                         " into the focused window",
                         (10, fh - 30), (200, 200, 200), 0.45)
                put_text(frame, "2 mouth taps = BACKSPACE   3 taps (or hold / V) = back to cursor",
                         (10, fh - 12), (200, 200, 200), 0.45)
            if typing_mode:
                put_text(frame, f"Morse: {typer.code or '.'}   last: {typer.last_decoded}",
                         (10, 75), (255, 200, 0))
                # Spaces render as _ so they can be seen and counted.
                put_text(frame, "Typed: " + typed[-40:].replace(" ", "_"),
                         (10, 100), (255, 200, 0))
                if now < flash_until:
                    put_text(frame, flash_text, (10, 130), (0, 255, 255), 0.8)
                # Cheat sheet: these gestures are not standard Morse.
                fh = frame.shape[0]
                put_text(frame, "short blink = dot   long blink = dash   pause = type letter",
                         (10, fh - 30), (200, 200, 200), 0.45)
                put_text(frame, "1 mouth tap = SPACE   2 taps (or ----) = BACKSPACE   3 taps = exit",
                         (10, fh - 12), (200, 200, 200), 0.45)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if ord('A') <= key <= ord('Z'):
                key += 32  # accept keys regardless of Shift / Caps Lock
            if key in (27, ord('q')):
                break
            elif key == ord(' '):
                if active:
                    active = False
                elif obs is not None:
                    neutral = (obs.nose_x, obs.nose_y)
                    controller.reset()
                    active = True
                else:
                    print("No face visible - cannot start")
            elif key == ord('c') and obs is not None:
                neutral = (obs.nose_x, obs.nose_y)
                controller.reset()
            elif key in (ord('['), ord(']')):
                step = -20 if key == ord('[') else 20
                for bar in (SENS_X_TRACKBAR, SENS_Y_TRACKBAR):
                    pos = cv2.getTrackbarPos(bar, WINDOW_NAME)
                    cv2.setTrackbarPos(bar, WINDOW_NAME,
                                       max(0, min(400, pos + step)))
            # In voice mode 3 taps means "back to cursor", not "Morse mode",
            # so it is routed to the voice toggle below instead.
            if any([k.pressed() for k in typing_keys]) or key == ord('t') \
                    or (taps == 3 and not voice_mode):
                typing_mode = not typing_mode
                typer.reset()
                tapper.cancel()
                typed = ""
                if typing_mode:
                    if voice_mode:
                        voice_mode = False
                        voice.stop()
                    # Re-calibrate eye open/closed references on every entry.
                    cal_phase, cal_until, cal_samples = "wait", now + 0.8, []
                else:
                    cal_phase = None
                print("Morse typing mode " + ("ON" if typing_mode else "OFF"), flush=True)
            elif key == ord('v') or taps == MouthTaps.HOLD \
                    or (voice_mode and taps == 3):
                voice_mode = not voice_mode
                tapper.cancel()
                if voice_mode:
                    typing_mode = False
                    cal_phase = None
                    typed = ""
                    voice.start()
                    beep((660, 90), (988, 130))  # rising: dictation on
                else:
                    voice.stop()
                    beep((988, 90), (660, 130))  # falling: dictation off
                print("Voice dictation " + ("ON" if voice_mode else "OFF"), flush=True)
    finally:
        voice.stop()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
