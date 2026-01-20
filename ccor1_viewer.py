import os
import re
import sys
import json
import time
import math
import queue
import threading
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple
from tkcalendar import DateEntry

import requests
import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ----------------------------
# Config: bucket + prefix
# ----------------------------
BUCKET = "noaa-nesdis-swfo-ccor-1-pds"
S3_BASE = f"https://{BUCKET}.s3.amazonaws.com"

# You can change this if NOAA moves folders. This is the only thing you may need to edit.
# Start with the most likely L1B layout (time-based folders).
DEFAULT_PREFIX = "SWFO/GOES-19/CCOR-1/ccor1-l1b/"

TS_RE = re.compile(r"CCOR1_(?:1B|L1B)_(\d{8}T\d{6}).*\.fits$", re.IGNORECASE)

@dataclass
@dataclass
class FitsItem:
    key: str
    url: str
    obs_dt: dt.datetime
    size_bytes: int


def parse_obs_dt_from_key(key: str) -> Optional[dt.datetime]:
    m = TS_RE.search(os.path.basename(key))
    if not m:
        return None
    s = m.group(1)  # YYYYMMDDThhmmss
    try:
        return dt.datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def utc_parse(s: str) -> dt.datetime:
    """
    Accepts:
      2026-01-15 06:45
      2026-01-15T06:45
      2026-01-15T06:45:25
    Assumes UTC.
    """
    s = s.strip().replace(" ", "T")
    fmts = ["%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"]
    for f in fmts:
        try:
            return dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime: {s}")


def s3_list_objects_with_sizes(prefix: str) -> List[Tuple[str, int]]:
    """
    Public S3 ListObjectsV2 (XML) without credentials.
    Returns list of (key, size_bytes).
    """
    out: List[Tuple[str, int]] = []
    token = None
    session = requests.Session()

    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token

        r = session.get(S3_BASE + "/", params=params, timeout=60)
        r.raise_for_status()
        xml = r.text

        # Parse each <Contents>...</Contents> block
        contents = re.findall(r"<Contents>(.*?)</Contents>", xml, flags=re.DOTALL)
        for block in contents:
            km = re.search(r"<Key>(.*?)</Key>", block)
            sm = re.search(r"<Size>(\d+)</Size>", block)
            if not km:
                continue
            key = km.group(1)
            size = int(sm.group(1)) if sm else 0
            out.append((key, size))

        is_truncated = "<IsTruncated>true</IsTruncated>" in xml
        if not is_truncated:
            break

        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    return out


def daterange_days(start_utc, end_utc):
    """
    Yield datetime.date objects for each UTC day in [start_utc, end_utc].
    """
    cur = start_utc.date()
    end = end_utc.date()
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def find_fits_in_range(prefix: str, start_utc: dt.datetime, end_utc: dt.datetime) -> List[FitsItem]:
    items: List[FitsItem] = []

    for day in daterange_days(start_utc, end_utc):
        day_prefix = f"{prefix}{day.year:04d}/{day.month:02d}/{day.day:02d}/"
        day_objs = s3_list_objects_with_sizes(day_prefix)

        for key, size in day_objs:
            if not key.lower().endswith(".fits"):
                continue
            obs = parse_obs_dt_from_key(key)
            if obs is None:
                continue
            if start_utc <= obs <= end_utc:
                items.append(FitsItem(
                    key=key,
                    url=f"{S3_BASE}/{key}",
                    obs_dt=obs,
                    size_bytes=size
                ))

    items.sort(key=lambda x: x.obs_dt)
    return items


def download_file(url: str, out_path: str, timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = out_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, out_path)


def read_ccor1_l1b_image(fits_path: str) -> Tuple[np.ndarray, dt.datetime]:
    """
    Returns float32 image (H,W) and DATE-OBS as UTC datetime.
    CCOR-1 L1B images are in HDU 1 CompImageHDU (per your header).
    """
    with fits.open(fits_path) as hdul:
        img = hdul[1].data
        if img is None:
            raise RuntimeError("No image data in HDU 1.")
        img = np.asarray(img, dtype=np.float32)

        hdr = hdul[1].header
        date_obs = hdr.get("DATE-OBS")
        if not date_obs:
            # fallback: parse from filename
            obs = parse_obs_dt_from_key(os.path.basename(fits_path))
            if obs is None:
                obs = dt.datetime.fromtimestamp(os.path.getmtime(fits_path), tz=dt.timezone.utc)
        else:
            # date_obs like: 2026-01-15T06:45:25.773
            # strip fractional seconds if needed
            ds = str(date_obs).split(".")[0]
            obs = dt.datetime.strptime(ds, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)

    return img, obs

# ----------------------------
# Processing modes (comet-friendly options)
# ----------------------------
def enhance_stack_temporal_median(stack: np.ndarray, idx: int, window: int = 2) -> np.ndarray:
    """
    Your original idea: temporal median background subtraction (good for CMEs; mixed for comets).
    Returns float image >=0
    """
    start = max(0, idx - window)
    end = min(stack.shape[0], idx + window + 1)
    bg = np.median(stack[start:end], axis=0)
    diff = stack[idx] - bg
    diff = np.clip(diff, 0, None)
    return diff


def enhance_spatial_highpass(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Better for point-like faint objects: subtract a heavily blurred background.
    """
    # OpenCV expects float32
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = img - blur
    # Keep both signs? For comet hunting, positive usually enough; but keep a little symmetric by shifting:
    # We'll keep positives but you can change this.
    hp = np.clip(hp, 0, None)
    return hp


def stretch_to_uint8(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, gamma: float = 0.6) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    # Replace NaN/Inf with finite numbers
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, [p_low, p_high])

    # Guard against flat frames
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)

    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    # Gamma (guard)
    if gamma and gamma > 0:
        x = np.power(x, gamma)

    # Final safety
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def postprocess_uint8(u8: np.ndarray, use_clahe: bool = True, sharpen: bool = True) -> np.ndarray:
    """
    CLAHE + mild unsharp mask
    """
    out = u8
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if sharpen:
        blurred = cv2.GaussianBlur(out, (3, 3), 0)
        out = cv2.addWeighted(out, 1.2, blurred, -0.2, 0)

    return out


def process_to_frames(
    fits_paths: List[str],
    out_frames_dir: str,
    mode: str = "spatial",
    fps_hint: int = 10,
) -> List[dict]:
    """
    Creates PNG frames and returns manifest entries [{png, date_obs, fits}]
    """
    os.makedirs(out_frames_dir, exist_ok=True)

    # Load all images for temporal mode
    imgs = []
    obs_times = []

    for p in fits_paths:
        img, obs = read_ccor1_l1b_image(p)
        imgs.append(img)
        obs_times.append(obs)

    stack = np.stack(imgs).astype(np.float32)

    manifest = []
    for i, (img, obs, fp) in enumerate(zip(imgs, obs_times, fits_paths)):
        if mode == "temporal":
            enhanced = enhance_stack_temporal_median(stack, i, window=2)
        elif mode == "runningdiff":
            if i == 0:
                enhanced = np.zeros_like(img)
            else:
                enhanced = img - imgs[i - 1]
                enhanced = np.clip(enhanced, 0, None)
        elif mode == "spatial":
            enhanced = enhance_spatial_highpass(img, sigma=50.0)
        else:  # "none"
            enhanced = np.clip(img, 0, None)

        u8 = stretch_to_uint8(enhanced, p_low=1, p_high=99, gamma=0.6)
        out = postprocess_uint8(u8, use_clahe=True, sharpen=True)

        fname = obs.strftime("%Y%m%d_%H%M%S") + "_ccor1.png"
        out_path = os.path.join(out_frames_dir, fname)
        Image.fromarray(out).save(out_path)

        manifest.append({
            "png": out_path,
            "fits": fp,
            "date_obs_utc": obs.isoformat(),
            "mode": mode,
        })

    # Save manifest
    mf_path = os.path.join(out_frames_dir, "manifest.json")
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump({"fps_hint": fps_hint, "frames": manifest}, f, indent=2)

    return manifest


# ----------------------------
# Tkinter Viewer
# ----------------------------
class Viewer(tk.Toplevel):
    def __init__(self, master, frames: List[dict], fps_default: int = 10):
        super().__init__(master)
        self.title("CCOR-1 Viewer (PNG cache)")
        self.geometry("1100x800")

        self.frames = frames
        self.idx = 0
        self.playing = False
        self.fps = tk.IntVar(value=fps_default)

        # UI
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text="<<", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(top, text="<", command=lambda: self.step(-1)).pack(side=tk.LEFT, padx=4)
        self.play_btn = ttk.Button(top, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text=">", command=lambda: self.step(+1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text=">>", command=self.next_frame).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="FPS:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Spinbox(top, from_=1, to=60, textvariable=self.fps, width=5).pack(side=tk.LEFT)

        self.info = ttk.Label(top, text="")
        self.info.pack(side=tk.LEFT, padx=16)

        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._tk_img = None
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(+1))
        self.bind("<space>", lambda e: self.toggle_play())

        self.update_idletasks()
        self.show_frame(0)

    def show_frame(self, idx: int):
        idx = max(0, min(idx, len(self.frames) - 1))
        self.idx = idx
        item = self.frames[self.idx]

        img = Image.open(item["png"])
        # Fit-to-window preserving aspect ratio
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            cw, ch = 1000, 700

        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img2 = img.resize((nw, nh), Image.BILINEAR)

        self._tk_img = ImageTk.PhotoImage(img2, master=self)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img)

        ts = item["date_obs_utc"]
        self.info.config(text=f"{self.idx+1}/{len(self.frames)}   {ts}")

    def step(self, delta: int):
        self.show_frame(self.idx + delta)

    def prev_frame(self):
        self.show_frame(0)

    def next_frame(self):
        self.show_frame(len(self.frames) - 1)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self.after(0, self._tick)

    def _tick(self):
        if not self.playing:
            return
        self.show_frame(self.idx + 1 if self.idx + 1 < len(self.frames) else 0)
        delay_ms = int(1000 / max(1, int(self.fps.get())))
        self.after(delay_ms, self._tick)

# ----------------------------
# CLI entry
# ----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="CCOR-1 L1B date-range downloader + processor + viewer")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"S3 prefix (default: {DEFAULT_PREFIX})")
    ap.add_argument("--start", required=True, help="UTC start datetime (e.g. '2026-01-15 06:00')")
    ap.add_argument("--end", required=True, help="UTC end datetime (e.g. '2026-01-15 12:00')")
    ap.add_argument("--out", default="ccor1_session", help="Output folder")
    ap.add_argument("--mode", default="spatial", choices=["spatial", "temporal", "runningdiff", "none"],
                    help="Processing mode (spatial is best starting point for comet hunting)")
    ap.add_argument("--skip-download", action="store_true", help="Use already-downloaded FITS if present")
    args = ap.parse_args()

    start = utc_parse(args.start)
    end = utc_parse(args.end)
    if end < start:
        raise SystemExit("end must be >= start")

    session_dir = os.path.abspath(args.out)
    raw_dir = os.path.join(session_dir, "raw")
    frames_dir = os.path.join(session_dir, "frames_" + args.mode)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Listing objects under prefix: {args.prefix}")
    items = find_fits_in_range(args.prefix, start, end)
    print(f"Found {len(items)} FITS in range [{start.isoformat()} .. {end.isoformat()}].")
    if not items:
        print("\nNo items found.")
        print("If you’re sure the time range is valid, the most likely issue is the prefix.")
        print("Try changing --prefix to match the bucket’s actual L1B folder layout.")
        return

    fits_paths = []
    for it in items:
        local_name = os.path.basename(it.key)
        out_path = os.path.join(raw_dir, local_name)
        fits_paths.append(out_path)
        if args.skip_download and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
        print(f"Downloading {local_name} ...")
        download_file(it.url, out_path)

    print(f"Processing {len(fits_paths)} frames -> {frames_dir} (mode={args.mode})")
    manifest = process_to_frames(fits_paths, frames_dir, mode=args.mode, fps_hint=10)
    print("Launching viewer...")
    self.viewer = Viewer(self, payload["manifest"], fps_default=int(self.fps.get()))


class JobGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CCOR-1 Downloader + Processor")
        self.geometry("780x420")

        self.prefix = tk.StringVar(value=DEFAULT_PREFIX)
        self.out_dir = tk.StringVar(value=os.path.abspath("ccor1_session"))
        self.mode = tk.StringVar(value="spatial")
        self.start_s = tk.StringVar(value="2026-01-15 06:00")
        self.end_s = tk.StringVar(value="2026-01-15 09:00")
        self.fps = tk.IntVar(value=10)

        self._confirm_var = tk.BooleanVar(value=False)
        self._confirm_ready = threading.Event()

        self._worker = None
        self._queue = queue.Queue()

        self._build_ui()
        self.after(100, self._poll_queue)

    def ask_proceed(self, count: int, total_mb: float) -> bool:
        msg = f"Found {count} files totaling {total_mb:.1f} MB.\n\nProceed with download?"
        return messagebox.askyesno("Confirm download", msg)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, **pad)

        row = 0
        # ---- Start ----
        ttk.Label(frm, text="Start (UTC)").grid(row=row, column=0, sticky="w", **pad)

        self.start_date = DateEntry(frm, width=12, date_pattern="yyyy-mm-dd")
        self.start_date.grid(row=row, column=1, sticky="w", **pad)

        self.start_h = tk.IntVar(value=6)
        self.start_m = tk.IntVar(value=0)

        ttk.Spinbox(frm, from_=0, to=23, textvariable=self.start_h, width=3, format="%02.0f").grid(row=row, column=1,
                                                                                                   sticky="w",
                                                                                                   padx=(150, 2),
                                                                                                   pady=6)
        ttk.Label(frm, text=":").grid(row=row, column=1, sticky="w", padx=(185, 2), pady=6)
        ttk.Spinbox(frm, from_=0, to=59, textvariable=self.start_m, width=3, format="%02.0f").grid(row=row, column=1,
                                                                                                   sticky="w",
                                                                                                   padx=(195, 2),
                                                                                                   pady=6)

        row += 1

        # ---- End ----
        ttk.Label(frm, text="End (UTC)").grid(row=row, column=0, sticky="w", **pad)

        self.end_date = DateEntry(frm, width=12, date_pattern="yyyy-mm-dd")
        self.end_date.grid(row=row, column=1, sticky="w", **pad)

        self.end_h = tk.IntVar(value=9)
        self.end_m = tk.IntVar(value=0)

        ttk.Spinbox(frm, from_=0, to=23, textvariable=self.end_h, width=3, format="%02.0f").grid(row=row, column=1,
                                                                                                 sticky="w",
                                                                                                 padx=(150, 2), pady=6)
        ttk.Label(frm, text=":").grid(row=row, column=1, sticky="w", padx=(185, 2), pady=6)
        ttk.Spinbox(frm, from_=0, to=59, textvariable=self.end_m, width=3, format="%02.0f").grid(row=row, column=1,
                                                                                              sticky="w",
                                                                                                padx=(195, 2), pady=6)

        row += 1
        ttk.Label(frm, text="Mode").grid(row=row, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.mode, values=["spatial", "temporal", "runningdiff", "none"],
                     state="readonly", width=26).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(frm, text="FPS (viewer)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(frm, from_=1, to=60, textvariable=self.fps, width=7).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(frm, text="S3 Prefix").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.prefix, width=58).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(frm, text="Output Folder").grid(row=row, column=0, sticky="w", **pad)
        out_entry = ttk.Entry(frm, textvariable=self.out_dir, width=58)
        out_entry.grid(row=row, column=1, sticky="w", **pad)
        ttk.Button(frm, text="Browse...", command=self._choose_out).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        self.start_btn = ttk.Button(frm, text="Start", command=self._start_job)
        self.start_btn.grid(row=row, column=0, sticky="w", **pad)
        self.pb = ttk.Progressbar(frm, mode="indeterminate")
        self.pb.grid(row=row, column=1, sticky="we", **pad)

        row += 1
        ttk.Label(frm, text="Status").grid(row=row, column=0, sticky="nw", **pad)
        self.log = tk.Text(frm, height=12, width=80)
        self.log.grid(row=row, column=1, columnspan=2, sticky="nsew", **pad)

        frm.grid_columnconfigure(1, weight=1)
        frm.grid_rowconfigure(row, weight=1)

        self._log("Ready. Enter a UTC start/end and click Start.")

    def _choose_out(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.out_dir.set(d)

    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()

                if kind == "log":
                    self._log(payload)

                elif kind == "confirm":
                    count, total_mb = payload
                    ok = self.ask_proceed(count, total_mb)
                    self._confirm_var.set(ok)
                    self._confirm_ready.set()

                elif kind == "done":
                    self.pb.stop()
                    self.start_btn.config(state="normal")

                    # Canceled
                    if payload.get("canceled"):
                        self._log("Canceled. No download started.")
                        continue

                    # Error
                    if payload.get("error"):
                        messagebox.showerror("Error", payload["error"])
                        continue

                    manifest = payload.get("manifest") or []
                    if not manifest:
                        self._log("No frames to display.")
                        continue

                    self._log("Launching viewer...")
                    self.viewer = Viewer(self, manifest, fps_default=int(self.fps.get()))

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    def _start_job(self):
        try:
            sd = self.start_date.get_date()  # datetime.date
            ed = self.end_date.get_date()

            start = dt.datetime(sd.year, sd.month, sd.day, int(self.start_h.get()), int(self.start_m.get()),
                                tzinfo=dt.timezone.utc)
            end = dt.datetime(ed.year, ed.month, ed.day, int(self.end_h.get()), int(self.end_m.get()),
                              tzinfo=dt.timezone.utc)

            if end < start:
                messagebox.showerror("Date error", "End must be after Start.")
                return
        except Exception as e:
            messagebox.showerror("Date error", str(e))
            return

        self.start_btn.config(state="disabled")
        self.pb.start(10)

        args = {
            "prefix": self.prefix.get().strip(),
            "out_dir": os.path.abspath(self.out_dir.get().strip()),
            "mode": self.mode.get().strip(),
            "start": start,
            "end": end,
            "fps": int(self.fps.get()),
        }

        self._worker = threading.Thread(target=self._run_job, args=(args,), daemon=True)
        self._worker.start()

    def _run_job(self, args):
        try:
            session_dir = args["out_dir"]
            raw_dir = os.path.join(session_dir, "raw")
            frames_dir = os.path.join(session_dir, "frames_" + args["mode"])
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)

            self._queue.put(("log", f"Listing: prefix={args['prefix']}"))
            items = find_fits_in_range(args["prefix"], args["start"], args["end"])
            if not items:
                self._queue.put(("log", "Found 0 FITS in range. Nothing to download."))
                self._queue.put(("done", {"canceled": True}))
                return

            total_bytes = sum(it.size_bytes for it in items)
            total_mb = total_bytes / (1024 * 1024)

            self._queue.put(("log", f"Found {len(items)} FITS; total download {total_mb:.1f} MB"))
            self._confirm_ready.clear()
            self._queue.put(("confirm", (len(items), total_mb)))

            # Wait for user response
            self._confirm_ready.wait()
            if not bool(self._confirm_var.get()):
                self._queue.put(("done", {"error": None, "manifest": []}))
                return

            self._queue.put(("log", f"Found {len(items)} FITS in range."))

            if not items:
                raise RuntimeError(
                    "Found 0 FITS. Most likely the S3 prefix is wrong for L1B.\n"
                    "Try a different prefix under SWFO/GOES-19/CCOR-1/ that contains CCOR1_1B_*.fits files."
                )

            fits_paths = []
            for n, it in enumerate(items, start=1):
                local_name = os.path.basename(it.key)
                out_path = os.path.join(raw_dir, local_name)
                fits_paths.append(out_path)
                self._queue.put(("log", f"Downloading {n}/{len(items)}: {local_name}"))
                download_file(it.url, out_path)

            self._queue.put(("log", f"Processing -> {frames_dir}  (mode={args['mode']})"))
            manifest = process_to_frames(fits_paths, frames_dir, mode=args["mode"], fps_hint=args["fps"])
            self._queue.put(("log", f"Saved {len(manifest)} PNG frames."))

            self._queue.put(("done", {"manifest": manifest}))
        except Exception as e:
            self._queue.put(("done", {"error": str(e)}))

if __name__ == "__main__":
    JobGUI().mainloop()
