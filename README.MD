# CCOR-1 FITS Downloader & Viewer

A Python/Tkinter GUI application for browsing, downloading, processing, and
interactively inspecting **NOAA CCOR-1 coronagraph images** (GOES-19) for
human-assisted discovery of faint moving objects such as comets.

This tool is designed for **visual analysis**, frame-to-frame inspection,
and pixel-level coordinate extraction — not automated detection.

---

## Features

### Data access
- Downloads **CCOR-1 Level-1B FITS files** from NOAA’s public S3 bucket
- Date-based folder traversal (`YYYY/MM/DD`)
- Calendar picker for start/end date selection (UTC)
- Pre-download confirmation dialog showing:
  - number of files
  - total download size (MB)

### Image processing
- FITS → PNG conversion and caching
- Multiple enhancement modes:
  - spatial (high-pass / local contrast)
  - temporal median subtraction
  - running difference
  - none (raw stretch)
- Contrast stretch + gamma
- CLAHE + mild sharpening (optimized for faint structures)

### Viewer
- Frame stepping and playback (adjustable FPS)
- Blue canvas background for reduced glare
- Keyboard controls:
  - Left / Right arrows → step frames
  - Space → play / pause
- Click-to-measure pixel coordinates:
  - origin at **top-left (0,0)**
  - coordinates reported in original image pixels

### Object analysis
- Shape-based centroid refinement around click location
- Interactive object tracking:
  - user-defined number of frames
  - configurable search radius (e.g. 10–30 px/frame)
- Frame-to-frame `(x, y)` coordinate logging
- CSV export of tracked positions

---

## Typical Workflow

1. Launch the application
2. Select **start** and **end** date/time (UTC)
3. Click **Start**
4. Review download size → confirm
5. Inspect frames visually
6. Click on a comet-like feature to get pixel coordinates
7. Use **Track →** to follow it across subsequent frames
8. Export tracked coordinates to CSV

---

## Data Source

Public NOAA S3 bucket:

s3://noaa-nesdis-swfo-ccor-1-pds/
SWFO/GOES-19/CCOR-1/ccor1-l1b/YYYY/MM/DD/


Files are named like:

CCOR1_3_20260120T074525_V00_NC.fits
CCOR1_3B_20260120T074525_V00_NC.fits


Timestamps are parsed directly from filenames.

---

## Installation

### Requirements
- Python 3.9+
- Tkinter (included with standard Python on Windows/macOS)

Install dependencies:

```bash
pip install -r requirements.txt

requirements.txt

numpy
requests
astropy
pillow
opencv-python
tkcalendar
```
---

##Running the Application

python ccor1_viewer.py
![CCOR-1 Viewer GUI](screenshots/ccor1_main.png)

### No command-line arguments are required — all configuration is done via the GUI.

Coordinate System

    Pixel coordinates are reported in PNG image space

    Origin: top-left corner (0,0)

    X increases to the right

    Y increases downward

    For CCOR-1 frames:

        x ∈ [0 … 2047]

        y ∈ [0 … 1919]

### Output
Cached data

    Downloaded FITS files

    Processed PNG frames (for fast viewing)

### CSV export

Tracked objects are saved as:

frame,timestamp,x,y,method
12,2026-01-20T07:45:25Z,1034,812,seed
13,2026-01-20T08:00:25Z,1041,809,track
...

## Limitations / Notes

    This is a human-in-the-loop analysis tool, not an automated detector

    Tracking assumes smooth motion (≈10–30 px/frame)

    Processed PNGs are not photometrically linear

    CCOR-1 data is currently considered preliminary per NOAA metadata

## License

MIT License

##Acknowledgements

    NOAA / NESDIS / SWPC for CCOR-1 data access

    Astropy, OpenCV, and the Python scientific ecosystem