#!/usr/bin/env python3
"""Yaw the RViz Orbit camera via synthetic left/right drags (recording helper).

RViz Orbit does not expose a ROS API for yaw. We hold a left-drag and slowly
move so the camera sweeps a yaw span and reverses (back-and-forth).

Uses python-xlib (XTEST) so we do not depend on xdotool.
"""
from __future__ import annotations

import argparse
import os
import signal
import time

from Xlib import X, display
from Xlib.ext import xtest

_STOP = False


def _request_stop(_signum, _frame):
    global _STOP
    _STOP = True


def _window_title(win) -> str:
    try:
        name = win.get_wm_name()
        if name:
            return str(name)
    except Exception:
        pass
    try:
        atom = win.display.intern_atom("_NET_WM_NAME")
        prop = win.get_full_property(atom, 0)
        if prop and prop.value:
            val = prop.value
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore")
            return str(val)
    except Exception:
        pass
    return ""


def find_rviz_window(dpy: display.Display):
    """Return the largest top-level window whose title contains 'RViz'."""
    root = dpy.screen().root
    atom = dpy.intern_atom("_NET_CLIENT_LIST")
    prop = root.get_full_property(atom, X.AnyPropertyType)
    candidates = []
    if prop and prop.value:
        for wid in prop.value:
            try:
                win = dpy.create_resource_object("window", wid)
                title = _window_title(win)
                if "rviz" not in title.lower():
                    continue
                geom = win.get_geometry()
                area = int(geom.width) * int(geom.height)
                candidates.append((area, win, title))
            except Exception:
                continue
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, win, title = candidates[0]
    return win, title


def absolute_xy(win) -> tuple[int, int, int, int]:
    """Window absolute origin and size."""
    geom = win.get_geometry()
    x = geom.x
    y = geom.y
    parent = win.query_tree().parent
    while parent is not None:
        try:
            pg = parent.get_geometry()
            x += pg.x
            y += pg.y
            parent = parent.query_tree().parent
        except Exception:
            break
    return int(x), int(y), int(geom.width), int(geom.height)


def _move(dpy: display.Display, x: float, y: float) -> None:
    xtest.fake_input(dpy, X.MotionNotify, x=int(x), y=int(y))
    dpy.sync()


def _button(dpy: display.Display, press: bool) -> None:
    xtest.fake_input(dpy, X.ButtonPress if press else X.ButtonRelease, 1)
    dpy.sync()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    p.add_argument(
        "--period",
        type=float,
        default=60.0,
        help="Total seconds to keep sweeping (usually bag duration)",
    )
    p.add_argument(
        "--one-way",
        type=float,
        default=None,
        help="Seconds for one 180-deg pass (default: period/2 → one out-and-back)",
    )
    p.add_argument(
        "--span-deg",
        type=float,
        default=180.0,
        help="Yaw span in degrees for each one-way pass (default: 180)",
    )
    p.add_argument(
        "--pixels-per-turn",
        type=float,
        default=1100.0,
        help="Horizontal drag pixels for ~360 deg yaw (scale for span-deg)",
    )
    p.add_argument("--rate", type=float, default=20.0)
    args = p.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _request_stop)

    dpy = display.Display(args.display)
    win = None
    title = None
    for _ in range(60):
        if _STOP:
            print("[orbit-drag] stopped while waiting for RViz")
            return 130
        win, title = find_rviz_window(dpy)
        if win is not None:
            break
        time.sleep(0.5)
    if win is None:
        print("[orbit-drag] no RViz window found; exiting")
        return 1

    ox, oy, ww, hh = absolute_xy(win)
    cx = ox + max(ww // 2, 1)
    cy = oy + max(hh // 2, 1)

    try:
        win.set_input_focus(X.RevertToParent, X.CurrentTime)
        dpy.sync()
    except Exception:
        pass
    _move(dpy, cx, cy)
    time.sleep(0.05)
    _button(dpy, True)
    time.sleep(0.02)
    _button(dpy, False)

    period = max(1.0, float(args.period))
    span_deg = max(1.0, min(360.0, float(args.span_deg)))
    one_way = float(args.one_way) if args.one_way is not None else max(period / 2.0, 1.0)
    one_way = max(1.0, one_way)
    rate = max(5.0, float(args.rate))
    dt = 1.0 / rate
    span_pixels = float(args.pixels_per_turn) * (span_deg / 360.0)
    pixels_per_step = span_pixels / (one_way * rate)

    print(
        f"[orbit-drag] display={args.display} title={title!r} "
        f"duration={period:.1f}s one-way={one_way:.1f}s "
        f"span={span_deg:.0f}deg ({span_pixels:.0f}px) "
        f"step={pixels_per_step:.3f}px at=({cx},{cy}) (back-and-forth)"
    )

    direction = -1
    traveled = 0.0
    accum = 0.0
    cur_x = float(cx)
    cur_y = float(cy)
    dragging = False
    t0 = time.time()
    try:
        _move(dpy, cur_x, cur_y)
        _button(dpy, True)
        dragging = True
        while not _STOP and time.time() - t0 < period + 1.0:
            accum += pixels_per_step
            step = int(accum)
            if step >= 1:
                accum -= step
                remaining = span_pixels - traveled
                if step > remaining:
                    step = max(1, int(remaining)) if remaining >= 1 else 0
                if step >= 1:
                    cur_x += direction * step
                    traveled += step
                    if cur_x < ox + ww * 0.2 or cur_x > ox + ww * 0.8:
                        if dragging:
                            _button(dpy, False)
                            dragging = False
                        cur_x = cx
                        _move(dpy, cur_x, cur_y)
                        _button(dpy, True)
                        dragging = True
                    else:
                        _move(dpy, cur_x, cur_y)
                if traveled >= span_pixels - 0.5:
                    direction *= -1
                    traveled = 0.0
                    accum = 0.0
            time.sleep(dt)
    finally:
        if dragging:
            try:
                _button(dpy, False)
            except Exception:
                pass

    print("[orbit-drag] done")
    return 130 if _STOP else 0


if __name__ == "__main__":
    raise SystemExit(main())
