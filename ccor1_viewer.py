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
import inspect
import requests
import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk, ImageOps
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
DEFAULT_PREFIX = "SWFO/GOES-19/CCOR-1/ccor1-l3/"

TS_RE = re.compile(
    r"CCOR1_(?:[1-9]B?)_(\d{8}T\d{6})_.*\.fits$",
    re.IGNORECASE
)

MODES = [
    "none",
    "spatial",
    "temporal",
    "runningdiff",
]

import re
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional
import requests

SWPC_CCOR1_FITS_BASE = "https://services.swpc.noaa.gov/products/ccor1/fits"  # no trailing slash needed

@dataclass
class FitsItem:
    key: str
    url: str
    obs_dt: Optional[dt.datetime]
    size_bytes: int

_ccor1_ts_re = re.compile(r"CCOR1_1B_(\d{8})T(\d{6})_V\d+_NC\.fits$", re.IGNORECASE)


def _parse_dt_from_ccor1_name(fname: str) -> Optional[dt.datetime]:
    m = _ccor1_ts_re.search(fname)
    if not m:
        return None
    ymd = m.group(1)
    hms = m.group(2)
    # filename timestamps are UTC
    return dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def list_swpc_ccor1_fits_in_range(start_utc: dt.datetime, end_utc: dt.datetime) -> List[FitsItem]:
    """
    Scrape the SWPC directory index once and filter by timestamp in filename.
    Returns FitsItem list sorted by obs_dt.
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start_utc/end_utc must be timezone-aware (UTC).")

    url = SWPC_CCOR1_FITS_BASE + "/"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    html = r.text

    # Apache index links look like: <a href="CCOR1_1B_20251215T000025_V00_NC.fits">...
    # Sizes are shown in a separate column; we'll *best-effort* parse size like "8.6M"
    link_re = re.compile(r'href="(CCOR1_1B_[^"]+?\.fits)"', re.IGNORECASE)
    files = sorted(set(link_re.findall(html)))

    def size_guess_from_row(fname: str) -> int:
        # Very light heuristic: find the row containing the filename and parse "8.6M" / "123K" etc if present.
        # If it fails, return 0 and your downloader still works.
        # Example row (from the index): "CCOR1_...fits</a> 2025-12-15 00:20  8.6M"
        row_re = re.compile(re.escape(fname) + r'.{0,120}?(\d+(?:\.\d+)?)\s*([KMG])', re.IGNORECASE)
        m = row_re.search(html)
        if not m:
            return 0
        val = float(m.group(1))
        unit = m.group(2).upper()
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
        return int(val * mult)

    items: List[FitsItem] = []
    for fname in files:
        obs = _parse_dt_from_ccor1_name(fname)
        if obs is None:
            continue
        if start_utc <= obs <= end_utc:
            items.append(FitsItem(
                key=fname,
                url=f"{SWPC_CCOR1_FITS_BASE}/{fname}",
                obs_dt=obs,
                size_bytes=size_guess_from_row(fname),
            ))

    items.sort(key=lambda x: x.obs_dt or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    return items


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

def parse_obs_dt_from_header(hdr):
    s = hdr.get("DATE-OBS") or hdr.get("DATE")
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def read_ccor1_l1b_image(path: str):
    with fits.open(path) as hdul:
        img = hdul[1].data.astype(np.float32)
        hdr = hdul[1].header

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # yawflip: record/log only (no pixel ops here)
    img, yawflip = apply_yawflip_if_needed(img, hdr)

    obs = parse_obs_dt_from_header(hdr)
    return img, obs, hdr

# ----------------------------
# Processing modes (comet-friendly options)
# ----------------------------


def wcs_make_north_up_vertical(img: np.ndarray, hdr) -> np.ndarray:
    """
    Ensure +image-Y points toward +SOLAR-Y (north).
    If not, flip vertically. (Keeps it simple: just fixes the upside-down case.)
    """
    # Build a CD matrix from PC + CDELT (common FITS pattern)
    cdelt1 = float(hdr.get("CDELT1", 1.0))
    cdelt2 = float(hdr.get("CDELT2", 1.0))

    pc11 = float(hdr.get("PC1_1", 1.0))
    pc12 = float(hdr.get("PC1_2", 0.0))
    pc21 = float(hdr.get("PC2_1", 0.0))
    pc22 = float(hdr.get("PC2_2", 1.0))

    # Pixel -> world linear part (approx; enough to determine axis direction)
    # Column 2 corresponds to a +1 step in pixel Y.
    d_world_dy_x = pc12 * cdelt2
    d_world_dy_y = pc22 * cdelt2   # this corresponds to SOLAR-Y / HPLT direction

    # If moving "down" in the image increases solar-y, the image is upside down.
    # (Depending on convention, this sign check may be inverted; this is the test.)
    if d_world_dy_y < 0:
        return np.flipud(img)

    return img


def apply_yawflip_if_needed(img: np.ndarray, hdr):
    yawflip = int(hdr.get("YAWFLIP", 0))

    # Otherwise apply based on state
    if yawflip == 2:  # INVERTED
        img = np.flipud(np.fliplr(img))  # 180° rotation
    # yawflip 0 upright: no-op
    # yawflip 1 neither: no-op + log (optional)
    print(f"YAWFLIP={yawflip} ")
    return img, yawflip


def enhance_stack_temporal_median(stack: np.ndarray, idx: int, window: int = 2) -> np.ndarray:
    """
    Temporal median background subtraction + band-pass for small slightly-extended blobs.
    Returns float32 image >= 0.
    """
    start = max(0, idx - window)
    end = min(stack.shape[0], idx + window + 1)

    cur = stack[idx].astype(np.float32)
    bg = np.median(stack[start:end], axis=0).astype(np.float32)

    resid = cur - bg
    resid = np.clip(resid, 0, None)

    # --- Band-pass (Difference of Gaussians) ---
    # small blur: suppress pixel noise
    g1 = cv2.GaussianBlur(resid, (0, 0), 1.2)
    # larger blur: suppress broader structures + star halos
    g2 = cv2.GaussianBlur(resid, (0, 0), 3.0)

    dog = g1 - g2
    dog = np.clip(dog, 0, None)

    # Optional: a touch more smoothing to stabilize thresholding
    dog = cv2.GaussianBlur(dog, (0, 0), 0.8)

    return dog


def enhance_spatial_highpass(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    x = img.astype(np.float32)

    pre = cv2.GaussianBlur(x, (0, 0), 1.2)
    bg  = cv2.GaussianBlur(pre, (0, 0), sigma)
    hp  = pre - bg
    hp  = np.clip(hp, 0, None)

    # NEW: remove broad residual structures (often rings/arcs)
    resid = cv2.GaussianBlur(hp, (0, 0), 120.0)  # try 100–140
    hp2 = hp - resid
    hp2 = np.clip(hp2, 0, None)

    return hp2


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
        img, obs, hdr = read_ccor1_l1b_image(p)

        # Apply orientation ONCE here
        img = wcs_make_north_up_vertical(img, hdr)

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
    def __init__(self, master, frames: List[dict], fps_default: int = 10, reg_var=None):
        super().__init__(master)
        self.title("CCOR-1 Viewer (PNG cache)")
        self.geometry("1100x800")

        self.frames = frames
        self.idx = 0
        self.playing = False
        self.fps = tk.IntVar(value=fps_default)
        self._disp_w, self._disp_h = 0, 0
        self._orig_w, self._orig_h = 0, 0
        self._offset_x = 0
        self._offset_y = 0
        self._scale = 1.0
        self.picks = []  # list of dicts
        self.reg_var = reg_var

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
        ttk.Button(top, text="Track →", command=self.track_forward_prompt).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(top, text="Export CSV", command=self.export_picks_csv).pack(side=tk.LEFT, padx=4)
        self.mouse_info = ttk.Label(top, text="x=—  y=—")
        self.mouse_info.pack(side=tk.LEFT, padx=8)

        self.info = ttk.Label(top, text="")
        self.info.pack(side=tk.LEFT, padx=16)

        self.canvas = tk.Canvas(self, bg="navy")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._tk_img = None
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(+1))
        self.bind("<space>", lambda e: self.toggle_play())

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)

        self._last_pick = None

        self.update_idletasks()
        self.show_frame(0)

    def load_png_gray(self, frame_idx: int) -> object:
        path = self.frames[frame_idx]["png"]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load PNG: {path}")
        return img

    def find_centroid_near(self, img_gray: np.ndarray, x: int, y: int, search: int = 30, half: int = 20):
        """
        Search around (x,y) within +-search pixels. For each candidate position, we
        compute a centroid in a small ROI (half) and score it by mask area (m00).
        Returns (best_x, best_y, ok).
        """
        h, w = img_gray.shape
        best = None  # (score, cx, cy)

        # Limit search window
        x_min = max(0, x - search)
        x_max = min(w - 1, x + search)
        y_min = max(0, y - search)
        y_max = min(h - 1, y + search)

        # Strategy: evaluate a coarse grid first (speed), then refine around best
        step = 3  # coarse step; adjust if needed

        def centroid_and_score_at(px, py):
            x0 = max(0, px - half);
            x1 = min(w, px + half + 1)
            y0 = max(0, py - half);
            y1 = min(h, py + half + 1)
            roi = img_gray[y0:y1, x0:x1]
            if roi.size == 0:
                return None

            roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
            _, mask = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m = cv2.moments(mask)
            if m["m00"] == 0:
                return None

            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            # Score: area (m00) — larger blob wins; you can change to sum of intensities if preferred
            score = m["m00"]
            return score, (x0 + cx), (y0 + cy)

        # Coarse scan
        for yy in range(y_min, y_max + 1, step):
            for xx in range(x_min, x_max + 1, step):
                res = centroid_and_score_at(xx, yy)
                if res is None:
                    continue
                score, cx, cy = res
                if best is None or score > best[0]:
                    best = (score, cx, cy)

        if best is None:
            return x, y, False

        # Refine scan around best centroid with step=1 in a small neighborhood
        _, bx, by = best
        refine = 6
        best2 = best
        for yy in range(max(0, by - refine), min(h - 1, by + refine) + 1):
            for xx in range(max(0, bx - refine), min(w - 1, bx + refine) + 1):
                res = centroid_and_score_at(xx, yy)
                if res is None:
                    continue
                score, cx, cy = res
                if score > best2[0]:
                    best2 = (score, cx, cy)

        return best2[1], best2[2], True

    def track_forward_prompt(self):
        if not hasattr(self, "_last_pick") or self._last_pick is None:
            messagebox.showinfo("Track", "Click on the object first to set the starting position.")
            return

        # Simple prompt via small popup
        win = tk.Toplevel(self)
        win.title("Track forward")
        win.resizable(False, False)

        ttk.Label(win, text="Frames to track:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        nvar = tk.IntVar(value=30)
        ttk.Spinbox(win, from_=1, to=500, textvariable=nvar, width=7).grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(win, text="Search radius (px/frame):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        svar = tk.IntVar(value=30)
        ttk.Spinbox(win, from_=5, to=200, textvariable=svar, width=7).grid(row=1, column=1, padx=10, pady=10)

        def go():
            win.destroy()
            self.track_forward(n_frames=int(nvar.get()), search=int(svar.get()))

        ttk.Button(win, text="Start", command=go).grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10))

    def track_forward(self, n_frames: int = 30, search: int = 30):
        x, y = self._last_pick
        start_idx = self.idx
        end_idx = min(len(self.frames) - 1, start_idx + n_frames)

        self._log_pick(start_idx, x, y, method="seed")

        for j in range(start_idx + 1, end_idx + 1):
            img = self.load_png_gray(j)
            nx, ny, ok = self.find_centroid_near(img, x, y, search=search, half=20)

            ts = self.frames[j].get("date_obs_utc", "")
            if not ok:
                self.info.config(text=f"{j + 1}/{len(self.frames)}  {ts}  TRACK LOST near ({x},{y})")
                break

            x, y = nx, ny
            self._log_pick(j, x, y, method="track")

        # Update last pick and show last tracked frame
        self._last_pick = (x, y)
        self.show_frame(min(end_idx, j))

    def _log_pick(self, frame_idx: int, x: int, y: int, method: str):
        ts = self.frames[frame_idx].get("date_obs_utc", "")
        rec = {"frame": frame_idx, "timestamp": ts, "x": int(x), "y": int(y), "method": method}
        self.picks.append(rec)
        #print(rec)  # optional

    def export_picks_csv(self):
        if not self.picks:
            messagebox.showinfo("Export", "No picks to export yet.")
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save picks CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "timestamp", "x", "y", "method"])
            w.writeheader()
            w.writerows(self.picks)

        messagebox.showinfo("Export", f"Saved: {path}")

    def _prep_for_reg(self, img_u8: np.ndarray) -> np.ndarray:
        x = img_u8.astype(np.float32)
        bg = cv2.GaussianBlur(x, (0, 0), 20)
        hp = x - bg
        hp = np.clip(hp, 0, None)
        hp = cv2.normalize(hp, None, 0, 1, cv2.NORM_MINMAX)
        return hp.astype(np.float32)

    def _align_to_prev(self, prev_u8: np.ndarray, cur_u8: np.ndarray):
        prev = self._prep_for_reg(prev_u8)
        cur = self._prep_for_reg(cur_u8)

        hann = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(prev * hann, cur * hann)

        # Guard rails: phaseCorrelate can return nonsense (e.g. W/2,H/2) when it fails.
        h, w = cur_u8.shape[:2]

        # "Too big to be real" threshold; tune as needed
        max_shift = min(w, h) * 0.10  # 10% of smallest dimension (~192 px here)

        if (response < 0.2) or (abs(dx) > max_shift) or (abs(dy) > max_shift):
            # Skip alignment; return original
            return cur_u8, (dx, dy), response

        # Sign: phaseCorrelate(prev, cur) gives the shift to apply to cur to match prev.
        M = np.array([[1, 0, -dx],
                      [0, 1, -dy]], dtype=np.float32)

        aligned = cv2.warpAffine(
            cur_u8, M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT
        )
        return aligned, (dx, dy), response

    def _running_diff_display(self, prev_u8, cur_aligned_u8):
        prev2 = cv2.GaussianBlur(prev_u8, (3, 3), 0)
        cur2 = cv2.GaussianBlur(cur_aligned_u8, (3, 3), 0)

        d = np.abs(cur2.astype(np.float32) - prev2.astype(np.float32))

        p1, p99 = np.percentile(d, [1, 99.5])
        if p99 <= p1:
            return cur_aligned_u8

        z = np.clip((d - p1) / (p99 - p1), 0, 1)
        z = np.power(z, 0.5)
        return (z * 255).astype(np.uint8)

    def show_frame(self, idx: int):
        idx = max(0, min(idx, len(self.frames) - 1))
        self.idx = idx
        item = self.frames[self.idx]

        use_reg = bool(self.reg_var and self.reg_var.get())
        print("DEBUG use_reg =", use_reg, "idx =", idx)

        if use_reg and idx > 0:
            prev = self.load_png_gray(idx - 1)
            cur = self.load_png_gray(idx)

            cur_aligned, (dx, dy), resp = self._align_to_prev(prev, cur)

            h, w = cur.shape
            max_shift = min(w, h) * 0.10

            if (resp < 0.2) or (abs(dx) > max_shift) or (abs(dy) > max_shift):
                disp_u8 = cur
            else:
                disp_u8 = self._running_diff_display(prev, cur_aligned)

            print(f"DEBUG shift dx={dx:.2f} dy={dy:.2f} resp={resp:.3f}")
            img = Image.fromarray(disp_u8)
        else:
            img = Image.open(item["png"])

        # If you move orientation fixes into process_to_frames(), remove this:
        img = ImageOps.flip(img)

        # Fit-to-window preserving aspect ratio
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 10 or ch < 10:
            cw, ch = 1000, 700

        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        # Save geometry for coordinate mapping
        self._orig_w, self._orig_h = iw, ih  # original PNG size
        self._disp_w, self._disp_h = nw, nh  # displayed size on canvas
        self._scale = scale  # displayed/original
        self._offset_x = (cw - nw) // 2  # top-left of displayed image on canvas
        self._offset_y = (ch - nh) // 2

        img2 = img.resize((nw, nh), Image.BILINEAR)

        self._tk_img = ImageTk.PhotoImage(img2, master=self)
        self.canvas.delete("all")
        self.canvas.create_image(self._offset_x, self._offset_y, anchor="nw", image=self._tk_img)
        self.draw_grid(step_pct=0.15)

        ts = item["date_obs_utc"]
        self.info.config(text=f"{self.idx + 1}/{len(self.frames)}   {ts}")

    def canvas_to_image_xy(self, cx: int, cy: int):
        """Map canvas coords to original PNG pixel coords (x,y). Returns None if outside image."""
        if not hasattr(self, "_offset_x"):
            return None

        # Inside the displayed image?
        if not (self._offset_x <= cx < self._offset_x + self._disp_w and
                self._offset_y <= cy < self._offset_y + self._disp_h):
            return None

        dx = cx - self._offset_x
        dy = cy - self._offset_y

        # displayed -> original (top-left origin)
        x = int(dx / self._scale)
        y = int(dy / self._scale)

        x = max(0, min(x, self._orig_w - 1))
        y = max(0, min(y, self._orig_h - 1))
        return x, y

    def refine_centroid_from_png(self, x: int, y: int, half: int = 20):
        """
        Shape-based refinement: centroid of a bright blob near (x,y) in processed PNG.
        Returns (rx, ry, ok).
        """
        #print("DEBUG: refine_centroid_from_png returning 3 values")
        path = self.frames[self.idx]["png"]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return x, y, False

        h, w = img.shape
        x0 = max(0, x - half);
        x1 = min(w, x + half + 1)
        y0 = max(0, y - half);
        y1 = min(h, y + half + 1)

        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return x, y, False

        roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
        _, mask = cv2.threshold(
            roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        m = cv2.moments(mask)
        if m["m00"] == 0:
            return x, y, False

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return x0 + cx, y0 + cy, True

    def draw_pick_overlay(self, x: int, y: int, half: int = 20):
        """
        Draw a crosshair at (x,y) in original coords and a ROI box of size (2*half+1).
        """
        # Convert original coords -> displayed coords
        dx = int(x * self._scale)
        dy = int(y * self._scale)

        cx = self._offset_x + dx
        cy = self._offset_y + dy

        # ROI box in displayed coords
        r = int(half * self._scale)
        x0 = cx - r;
        y0 = cy - r
        x1 = cx + r;
        y1 = cy + r

        self.canvas.delete("pick")
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="yellow", width=2, tags="pick")
        self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill="yellow", width=2, tags="pick")
        self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill="yellow", width=2, tags="pick")

    def on_click(self, event):
        mapped = self.canvas_to_image_xy(event.x, event.y)
        if mapped is None:
            return

        x, y = mapped

        print("DEBUG refine fn:", inspect.getsourcefile(self.refine_centroid_from_png),
              inspect.getsourcelines(self.refine_centroid_from_png)[1])

        res = self.refine_centroid_from_png(x, y, half=20)
        if isinstance(res, tuple) and len(res) == 3:
            rx, ry, ok = res
        elif isinstance(res, tuple) and len(res) == 2:
            rx, ry = res
            ok = True  # or False if you want to indicate "unverified centroid"
        else:
            raise RuntimeError(f"Unexpected refine_centroid_from_png return: {res!r}")

        self._last_pick = (rx, ry) if ok else (x, y)

        ts = self.frames[self.idx].get("date_obs_utc", "")
        if ok:
            self.info.config(text=f"{self.idx + 1}/{len(self.frames)}  {ts}  click=({x},{y})  centroid=({rx},{ry})")
        else:
            self.info.config(text=f"{self.idx + 1}/{len(self.frames)}  {ts}  click=({x},{y})  centroid=(n/a)")

        # Draw overlay at refined location (or click if no centroid)
        self.draw_pick_overlay(rx if ok else x, ry if ok else y, half=20)

    def on_motion(self, event):
        mapped = self.canvas_to_image_xy(event.x, event.y)
        if mapped is None:
            self.mouse_info.config(text="x=—  y=—")
            return
        x, y = mapped
        self.mouse_info.config(text=f"x={x}  y={y}")

    def _draw_marker(self, dx, dy):
        # dx,dy are displayed-image coords (not canvas)
        cx = self._offset_x + dx
        cy = self._offset_y + dy
        r = 6
        self.canvas.delete("marker")
        self.canvas.create_line(cx - r, cy, cx + r, cy, fill="yellow", tags="marker")
        self.canvas.create_line(cx, cy - r, cx, cy + r, fill="yellow", tags="marker")

    def draw_grid(self, step_pct: float = 0.15):
        """
        Draw major grid lines every step_pct of original image width/height.
        Grid is drawn in canvas coords, scaled/offset to match the displayed image.
        """
        if self._scale <= 0:
            return

        self.canvas.delete("grid")

        # Percent positions: 15%, 30%, ... 90%
        n = int(1.0 / step_pct)
        pcts = [step_pct * i for i in range(1, n)]  # exclude 0% and 100%

        # Draw vertical lines
        for p in pcts:
            x_img = p * (self._orig_w - 1)  # in original pixel coords
            x_can = self._offset_x + int(x_img * self._scale)
            self.canvas.create_line(
                x_can, self._offset_y,
                x_can, self._offset_y + self._disp_h,
                fill="lightcyan", width=1, tags="grid"
            )

        # Draw horizontal lines
        for p in pcts:
            y_img = p * (self._orig_h - 1)
            y_can = self._offset_y + int(y_img * self._scale)
            self.canvas.create_line(
                self._offset_x, y_can,
                self._offset_x + self._disp_w, y_can,
                fill="lightcyan", width=1, tags="grid"
            )

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
# def main():
#     import argparse
#     ap = argparse.ArgumentParser(description="CCOR-1 L1B date-range downloader + processor + viewer")
#     ap.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"S3 prefix (default: {DEFAULT_PREFIX})")
#     ap.add_argument("--start", required=True, help="UTC start datetime (e.g. '2026-01-15 00:00')")
#     ap.add_argument("--end", required=True, help="UTC end datetime (e.g. '2026-01-15 23:45')")
#     ap.add_argument("--out", default="ccor1_session", help="Output folder")
#     ap.add_argument("--mode", default="none", choices=["spatial", "temporal", "runningdiff", "none"],
#                     help="Processing mode (spatial is best starting point for comet hunting)")
#     ap.add_argument("--skip-download", action="store_true", help="Use already-downloaded FITS if present")
#     args = ap.parse_args()
#
#     start = utc_parse(args.start)
#     end = utc_parse(args.end)
#     if end < start:
#         raise SystemExit("end must be >= start")
#
#     session_dir = os.path.abspath(args.out)
#     raw_dir = os.path.join(session_dir, "raw")
#     frames_dir = os.path.join(session_dir, "frames_" + args.mode)
#
#     os.makedirs(raw_dir, exist_ok=True)
#     os.makedirs(frames_dir, exist_ok=True)
#
#     print(f"Listing objects under prefix: {args.prefix}")
#     items = find_fits_in_range(args.prefix, start, end)
#     print(f"Found {len(items)} FITS in range [{start.isoformat()} .. {end.isoformat()}].")
#     if not items:
#         print("\nNo items found.")
#         print("If you’re sure the time range is valid, the most likely issue is the prefix.")
#         print("Try changing --prefix to match the bucket’s actual L1B folder layout.")
#         return
#
#     fits_paths = []
#     for it in items:
#         local_name = os.path.basename(it.key)
#         out_path = os.path.join(raw_dir, local_name)
#         fits_paths.append(out_path)
#         if args.skip_download and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
#             continue
#         print(f"Downloading {local_name} ...")
#         download_file(it.url, out_path)
#
#     print(f"Processing {len(fits_paths)} frames -> {frames_dir} (mode={args.mode})")
#     manifest = process_to_frames(fits_paths, frames_dir, mode=args.mode, fps_hint=10)
#     print("Launching viewer...")
#     self.viewer = Viewer(self, payload["manifest"], fps_default=int(self.fps.get()))
#

class JobGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CCOR-1 Downloader + Processor")
        self.geometry("780x420")

        self.prefix = tk.StringVar(value=DEFAULT_PREFIX)
        self.out_dir = tk.StringVar(value=os.path.abspath("ccor1_session"))
        self.mode = tk.StringVar(value="none")
        self.start_s = tk.StringVar(value="2026-01-15 00:00")
        self.end_s = tk.StringVar(value="2026-01-15 23:45")
        self.fps = tk.IntVar(value=5)
        self.register_on_stars = tk.BooleanVar(value=False)
        self.source = tk.StringVar(value="s3")
        self._confirm_var = tk.BooleanVar(value=False)
        self._confirm_ready = threading.Event()

        self._worker = None
        self._queue = queue.Queue()

        self._build_ui()
        self.after(100, self._poll_queue)

    def _on_source_change(self):
        src = self.source.get().lower()
        if src == "swpc":
            self.prefix_entry.configure(state="disabled")
        else:
            self.prefix_entry.configure(state="normal")

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
        ttk.Label(frm, text="Source").grid(row=row, column=0, sticky="w", **pad)

        self.source_cb = ttk.Combobox(
            frm,
            textvariable=self.source,
            values=["s3", "swpc"],
            width=12,
            state="readonly"
        )
        self.source_cb.grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(frm, text="Mode").grid(row=row, column=0, sticky="w", **pad)
        ttk.Combobox(frm, textvariable=self.mode, values=MODES, state="readonly", width=12) \
            .grid(row=row, column=1, sticky="w", **pad)

        ttk.Checkbutton(
            frm,
            text="Register (stars)",
            variable=self.register_on_stars
        ).grid(row=row, column=2, sticky="w", padx=(12, 0), pady=6)

        row += 1
        ttk.Label(frm, text="FPS (viewer)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(frm, from_=1, to=60, textvariable=self.fps, width=7).grid(row=row, column=1, sticky="w", **pad)

        row += 1

        ttk.Label(frm, text="S3 Prefix").grid(row=row, column=0, sticky="w", **pad)

        self.prefix_entry = ttk.Entry(frm, textvariable=self.prefix, width=58)
        self.prefix_entry.grid(row=row, column=1, sticky="w", **pad)

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
       # self._on_source_change()

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
                    self.viewer = Viewer(
                        self,
                        manifest,
                        fps_default=int(self.fps.get()),
                        reg_var=self.register_on_stars
                    )

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

        source = self.source.get().strip().lower()

        args = {
            "prefix": self.prefix.get().strip(),
            "out_dir": os.path.abspath(self.out_dir.get().strip()),
            "mode": self.mode.get().strip(),
            "start": start,
            "end": end,
            "fps": int(self.fps.get()),
            "source": source,
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
            start_utc = args["start"]
            end_utc = args["end"]
            source = args["source"].lower()

            if source == "swpc":
                items = list_swpc_ccor1_fits_in_range(start_utc, end_utc)
            else:
                items = find_fits_in_range(prefix=args["prefix"], start_utc=start_utc, end_utc=end_utc)

            self._queue.put(("log", f"Listing: from source={source}"))

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
