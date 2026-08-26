"""Offline voice dictation via Vosk.

The microphone is only captured while dictation is active, and all
recognition runs locally - no audio leaves the machine. The model
(~40 MB) is downloaded automatically on first use.
"""

import json
import os
import queue
import threading
import time
import urllib.request
import zipfile

import numpy as np

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SAMPLE_RATE = 16000


class VoiceTyper:
    """Streams mic audio into Vosk on a background thread.

    The tracking loop stays real-time: start()/stop() only flip a flag and
    spawn/join the worker; poll() drains finished phrases without blocking.
    """

    def __init__(self):
        self._model = None
        self._finals = queue.Queue()
        self._thread = None
        self._stop = threading.Event()
        self.partial = ""   # phrase in progress, for the HUD
        self.state = "off"  # "off" | "listening" | progress text | "error: ..."
        self._last_speech = None  # monotonic time speech was last heard
        self._noise = None  # adaptive noise-floor RMS estimate
        self.heard_speech = False  # any speech heard since start()

    def idle_s(self) -> float:
        """Seconds since speech was last heard (0 unless actively listening)."""
        if self.state != "listening" or self._last_speech is None:
            return 0.0
        return time.monotonic() - self._last_speech

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            # A previous session is still winding down; let it finish so two
            # workers never hold the mic at once.
            self._stop.set()
            self._thread.join(timeout=5)
        self._stop = threading.Event()
        self.partial = ""
        self.heard_speech = False
        self._noise = None
        self.state = "starting"
        self._thread = threading.Thread(target=self._run, args=(self._stop,),
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def poll(self):
        """Return the list of finished phrases since the last call."""
        out = []
        while True:
            try:
                out.append(self._finals.get_nowait())
            except queue.Empty:
                return out

    def _ensure_model(self, stop):
        path = os.path.join(MODELS_DIR, MODEL_NAME)
        if os.path.isdir(path):
            return path
        os.makedirs(MODELS_DIR, exist_ok=True)
        zip_path = path + ".zip"
        try:
            with urllib.request.urlopen(MODEL_URL) as resp, open(zip_path, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                self.state = "downloading model (~40 MB)"
                while True:
                    if stop.is_set():
                        return None
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        self.state = f"downloading model {done * 100 // total}%"
            self.state = "unpacking model"
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(MODELS_DIR)
            return path
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def _run(self, stop):
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as e:
            self.state = f"error: pip install vosk sounddevice ({e.name} missing)"
            return
        try:
            if self._model is None:
                path = self._ensure_model(stop)
                if path is None:  # toggled off mid-download
                    self.state = "off"
                    return
                SetLogLevel(-1)
                self.state = "loading model"
                self._model = Model(path)
            rec = KaldiRecognizer(self._model, SAMPLE_RATE)
            audio = queue.Queue()

            def on_audio(indata, frames, timestamp, status):
                # PortAudio thread: just hand the bytes off, never recognize
                # here or the callback overruns and drops audio.
                audio.put(bytes(indata))

            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000,
                                   dtype="int16", channels=1,
                                   callback=on_audio):
                # Set before flipping the state so idle_s() never reads a
                # stale timestamp from a previous session.
                self._last_speech = time.monotonic()
                self.state = "listening"
                while not stop.is_set():
                    try:
                        data = audio.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if rec.AcceptWaveform(data):
                        self.partial = ""
                        text = json.loads(rec.Result()).get("text", "")
                        if text:
                            self._finals.put(text)
                            self.heard_speech = True
                    else:
                        self.partial = json.loads(rec.PartialResult()).get("partial", "")

                    # Silence is judged by ear (chunk RMS vs an adaptive
                    # noise floor), not by recognition progress: Vosk's
                    # partial text can freeze for seconds mid-speech, which
                    # cut dictation off while the user was still talking.
                    samples = np.frombuffer(data, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                    if self._noise is None:
                        self._noise = rms
                    else:
                        # Falls fast so a loud first chunk can't poison the
                        # floor; rises slowly so speech can't become "noise".
                        k = 0.25 if rms < self._noise else 0.01
                        self._noise += k * (rms - self._noise)
                    if rms > max(3.0 * self._noise, 150.0):
                        self._last_speech = time.monotonic()
                        self.heard_speech = True
                    elif self.partial:
                        # The recognizer is still mid-phrase; don't call it
                        # silence until the phrase is finalized.
                        self._last_speech = time.monotonic()
            # Whatever was being said when dictation ended still counts.
            text = json.loads(rec.FinalResult()).get("text", "")
            if text:
                self._finals.put(text)
            self.partial = ""
            self.state = "off"
        except Exception as e:
            self.partial = ""
            self.state = f"error: {e}"
