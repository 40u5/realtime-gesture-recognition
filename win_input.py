"""Win32 SendInput wrappers: mouse clicks and keystrokes."""

import ctypes
import time
from ctypes import wintypes

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_BACK = 0x08
VK_CONTROL = 0x11
VK_V = 0x56
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
# Default ctypes restype is a 32-bit int; handles/pointers get truncated
# on 64-bit Windows without these.
_kernel32.GlobalAlloc.restype = ctypes.c_void_p
_kernel32.GlobalLock.restype = ctypes.c_void_p
_kernel32.GlobalLock.argtypes = (ctypes.c_void_p,)
_kernel32.GlobalUnlock.argtypes = (ctypes.c_void_p,)
_kernel32.GlobalFree.argtypes = (ctypes.c_void_p,)
_user32.SetClipboardData.restype = ctypes.c_void_p
_user32.SetClipboardData.argtypes = (wintypes.UINT, ctypes.c_void_p)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


def _send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    ctypes.windll.user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT))


def _mouse(flags):
    return INPUT(type=INPUT_MOUSE, u=_INPUTUNION(mi=MOUSEINPUT(0, 0, 0, flags, 0, 0)))


def _key(vk=0, scan=0, flags=0):
    return INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=KEYBDINPUT(vk, scan, flags, 0, 0)))


def left_click():
    _send(_mouse(MOUSEEVENTF_LEFTDOWN), _mouse(MOUSEEVENTF_LEFTUP))


def right_click():
    _send(_mouse(MOUSEEVENTF_RIGHTDOWN), _mouse(MOUSEEVENTF_RIGHTUP))


def set_clipboard(text: str) -> bool:
    data = text.encode("utf-16-le") + b"\x00\x00"
    handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not handle:
        return False
    ptr = _kernel32.GlobalLock(handle)
    ctypes.memmove(ptr, data, len(data))
    _kernel32.GlobalUnlock(handle)
    if not _user32.OpenClipboard(None):
        _kernel32.GlobalFree(handle)
        return False
    _user32.EmptyClipboard()
    ok = bool(_user32.SetClipboardData(CF_UNICODETEXT, handle))
    if not ok:  # on success the clipboard owns the handle
        _kernel32.GlobalFree(handle)
    _user32.CloseClipboard()
    return ok


def paste_text(text: str) -> bool:
    """Put text on the clipboard and send Ctrl+V.

    Injecting long phrases as raw unicode key events proved unreliable
    (apps dropped or mangled parts of the burst); a paste delivers the
    exact string no matter its length.
    """
    if not set_clipboard(text):
        return False
    time.sleep(0.02)  # let the clipboard update settle before the paste
    _send(_key(vk=VK_CONTROL), _key(vk=VK_V),
          _key(vk=VK_V, flags=KEYEVENTF_KEYUP),
          _key(vk=VK_CONTROL, flags=KEYEVENTF_KEYUP))
    return True


def backspace():
    _send(_key(vk=VK_BACK), _key(vk=VK_BACK, flags=KEYEVENTF_KEYUP))
