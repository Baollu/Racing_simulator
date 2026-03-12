"""
input_manager.py
-----------------
Non-blocking keyboard input handler using pynput.
Runs a listener in a background thread and exposes get_action()
which the main loop calls every simulation step.
"""

import threading
from pynput import keyboard


class InputManager:
    """
    Captures arrow key state in real time using a background pynput listener.

    Usage:
        im = InputManager()
        im.start()
        # in your loop:
        steering, acceleration = im.get_action()
        # on exit:
        im.stop()
    """

    def __init__(self) -> None:
        self._pressed_keys: set = set()
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background keyboard listener thread."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        print("[InputManager] Keyboard listener started.")
        print("[InputManager] Controls: LEFT/RIGHT = steer | UP = accelerate | DOWN = brake")

    def stop(self) -> None:
        """Stop the background keyboard listener thread."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        print("[InputManager] Keyboard listener stopped.")

    # ------------------------------------------------------------------
    # pynput callbacks (called from background thread)
    # ------------------------------------------------------------------

    def _on_press(self, key) -> None:
        key_name = self._key_name(key)
        if key_name is not None:
            with self._lock:
                self._pressed_keys.add(key_name)

    def _on_release(self, key) -> None:
        key_name = self._key_name(key)
        if key_name is not None:
            with self._lock:
                self._pressed_keys.discard(key_name)

    @staticmethod
    def _key_name(key) -> str | None:
        """Convert a pynput key to a simple string name, or None if not relevant."""
        try:
            if key == keyboard.Key.left:
                return "left"
            if key == keyboard.Key.right:
                return "right"
            if key == keyboard.Key.up:
                return "up"
            if key == keyboard.Key.down:
                return "down"
        except AttributeError:
            pass
        return None

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def get_action(self) -> tuple[float, float]:
        """
        Read current key state and return (steering, acceleration).

        Mapping:
          LEFT  arrow -> steering      = -1.0
          RIGHT arrow -> steering      = +1.0
          UP    arrow -> acceleration  = +1.0
          DOWN  arrow -> acceleration  = -1.0
          No key      ->  0.0 for that axis

        Returns:
            steering:     float in [-1.0, 1.0]
            acceleration: float in [-1.0, 1.0]
        """
        # Acquire lock, copy set, release immediately
        with self._lock:
            keys = set(self._pressed_keys)

        steering = 0.0
        if "left" in keys:
            steering -= 1.0
        if "right" in keys:
            steering += 1.0

        acceleration = 0.0
        if "up" in keys:
            acceleration += 1.0
        if "down" in keys:
            acceleration -= 1.0

        return steering, acceleration
