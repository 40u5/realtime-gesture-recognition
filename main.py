"""Head Tracking Mouse

Moves the Windows cursor with small head turns, using a webcam and
MediaPipe Face Mesh. Mouth gestures click; voice dictation types.
"""

import argparse
import sys
import threading
import time
import winsound

import cv2

import win_input
from config import CONFIG
from cursor_controller import CursorController
from face_tracker import FaceTracker
from gestures import MouthTaps
from voice import VoiceTyper

WINDOW_NAME = "Head Tracking Mouse"
SENS_X_TRACKBAR = "Sens X %"
SENS_Y_TRACKBAR = "Sens Y %"
FACE_LOST_RESET_S = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Control the mouse with your head.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="track and preview without moving the cursor")
    return parser.parse_args()


def log(msg):
    print(msg, flush=True)
    try:
        with open("voice_log.txt", "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def beep(*notes):
    """Play (freq_hz, dur_ms) notes without blocking the tracking loop."""
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
    voice = VoiceTyper()

    active = False
    voice_mode = False
    typed = ""  # everything typed this session, echoed on the HUD
    flash_text = ""   # transient HUD notice for invisible characters
    flash_until = 0.0
    neutral = None  # nose position (normalized) that maps to screen center
    last_face_time = 0.0
    fps = 0.0
    prev_t = time.monotonic()

    print(f"Screen: {controller.screen_w}x{controller.screen_h}"
          + (" (dry run: cursor will not move)" if args.dry_run else ""))
    print("SPACE start/pause | C re-center | V or hold mouth open ~1s: voice dictation | Q quit")
    print("Mouth taps: 1 = left click, 2 = right click")
    print("Voice: speak and each finished phrase is typed; 2 taps = backspace, "
          f"3 taps = back to cursor; turns itself off after {CONFIG.voice_silence_s:g}s of silence")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.createTrackbar(SENS_X_TRACKBAR, WINDOW_NAME, 100, 400, lambda v: None)
    cv2.createTrackbar(SENS_Y_TRACKBAR, WINDOW_NAME, 100, 400, lambda v: None)

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

            # The tap counter runs in every mode so voice-mode gestures
            # (2 taps = backspace, 3 = exit) still work while dictating.
            taps = tapper.update(obs.mouth_ratio, now) if obs is not None else 0

            if active and not voice_mode and obs is not None:
                if taps == 1:
                    print("[dry run] left click") if args.dry_run else win_input.left_click()
                elif taps == 2:
                    print("[dry run] right click") if args.dry_run else win_input.right_click()

            # Drain finished phrases every frame (not only in voice mode) so
            # the last phrase still lands after dictation is toggled off.
            for text in voice.poll():
                typed += text + " "
                if args.dry_run:
                    print(f"[dry run] paste {text + ' '!r}")
                else:
                    log(f"[type] paste {text + ' '!r}")
                    win_input.paste_text(text + " ")

            if voice_mode:
                # Fast talking registers as mouth taps, so gestures only
                # count while the recognizer hears silence (no partial in
                # flight) - deliberate taps happen in a speech pause.
                if taps == 2 and not voice.partial:
                    typed = typed[:-1]
                    flash_text, flash_until = "[ BACKSPACE ]", now + 1.2
                    if args.dry_run:
                        print("[dry run] backspace")
                    else:
                        log("[type] backspace")
                        win_input.backspace()
                st = voice.state
                if st == "listening":
                    status, status_color = "VOICE - listening", (0, 255, 0)
                elif st.startswith("error"):
                    status, status_color = "VOICE " + st, (0, 0, 255)
                else:
                    status, status_color = f"VOICE - {st}", (255, 200, 0)
            elif active:
                if obs is None:
                    status, status_color = "NO FACE", (0, 0, 255)
                    if now - last_face_time > FACE_LOST_RESET_S:
                        controller.reset()
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
            face_txt = f"   Mouth: {obs.mouth_ratio:.2f}" if obs is not None else ""
            put_text(frame, f"Voice: {'ON' if voice_mode else 'OFF'}   "
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
                put_text(frame, "speak normally - each finished phrase is typed into"
                         f" the focused window; {CONFIG.voice_silence_s:g}s of silence exits",
                         (10, fh - 30), (200, 200, 200), 0.45)
                put_text(frame, "2 mouth taps = BACKSPACE   3 taps (or hold / V) = back to cursor",
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
            if key == ord('v') or taps == MouthTaps.HOLD \
                    or (voice_mode and taps == 3 and not voice.partial):
                voice_mode = not voice_mode
                tapper.cancel()
                if voice_mode:
                    typed = ""
                    voice.start()
                    beep((660, 90), (988, 130))  # rising: dictation on
                else:
                    voice.stop()
                    beep((988, 90), (660, 130))  # falling: dictation off
                print("Voice dictation " + ("ON" if voice_mode else "OFF"), flush=True)
            elif voice_mode and voice.idle_s() > (
                    CONFIG.voice_silence_s if voice.heard_speech
                    else CONFIG.voice_start_grace_s):
                # The worker's FinalResult still lands via poll() next frame,
                # so a phrase in flight when the timeout hits is not lost.
                voice_mode = False
                voice.stop()
                beep((988, 90), (660, 130))  # falling: dictation off
                print("Voice dictation OFF (silence)", flush=True)
    finally:
        voice.stop()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
