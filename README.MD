# CCOR-1 Viewer & Analysis Tool

A Python GUI application for browsing, downloading, processing, and interactively
analyzing **NOAA CCOR-1 coronagraph images** from GOES-19.

This tool is designed for **human-assisted inspection** of CCOR-1 data, including
frame-by-frame viewing, pixel-accurate coordinate measurement, and short-range
object tracking (e.g. comet candidates).

---

## Key Features

### Data Access
- Downloads CCOR-1 Level-1B FITS files from NOAA’s public S3 archive
- Date-range selection using a calendar picker (UTC)
- Automatic traversal of `YYYY/MM/DD` directory structure
- Pre-download confirmation dialog showing:
  - number of files
  - estimated total download size

### Image Processing
- FITS → PNG conversion with local caching
- Multiple enhancement modes:
  - spatial contrast enhancement
  - temporal median subtraction
  - running difference
  - raw (minimal stretch)
- Contrast stretch, gamma adjustment, CLAHE, and mild sharpening
- Processing optimized for faint coronal structures

### Viewer
- Interactive Tkinter-based viewer
- Adjustable playback FPS
- Frame stepping and keyboard controls:
  - Left / Right arrows → step frames
  - Space → play / pause
- Blue canvas background to reduce glare
- Scaled grid overlay (major lines every 15% of image width/height)
- Live mouse tracking of pixel coordinates

### Coordinate Measurement
- Click-to-measure pixel coordinates
- Coordinates reported in **original image pixel space**
  - origin at top-left `(0,0)`
- Shape-based centroid refinement around click location

### Object Tracking
- User-initiated tracking across subsequent frames
- Configurable:
  - number of frames to track
  - search radius (typ. 10–30 px/frame)
- Frame-to-frame centroid tracking using processed PNGs
- Tracking stops automatically if object is lost

### Output
- CSV export of tracked positions:
  - frame index
  - timestamp (UTC)
  - x, y pixel coordinates
  - method (`seed` or `track`)

---

## Typical Workflow

1. Launch the application
2. Select start and end dates (UTC)
3. Confirm download size
4. Inspect frames visually
5. Click on a feature of interest to measure pixel coordinates
6. Use **Track →** to follow the object across frames
7. Export tracked coordinates to CSV for further analysis

---
## Mode-by-mode guidance
### mode = none + registration ON

Yes — this makes sense.

What you get:

Raw signal preserved

Stars stop drifting

Comets move smoothly across a stable star field

Use this when:

You want to visually track a faint dot

You want accurate centroiding / tracking

You’re inspecting shape (slightly extended blobs)

This is your baseline “science” view.

### ️ mode = spatial + registration ON

Usually unnecessary, sometimes harmful.

Why:

Spatial high-pass already suppresses smooth background

Registration adds interpolation noise

Star residuals can get “ringy”

When it might help:

Very slow star drift

Strong coronal gradient overwhelming faint objects

Recommendation:

Try without registration first

If stars smear frame-to-frame, then enable registration

### mode = temporal + registration ON

Usually redundant.

Why:

Temporal median subtraction already removes static features (stars)

Registration + temporal can over-subtract or create ghosts

Typical symptom:

Increased noise

Patchy background

Comet partially suppressed

Recommendation:

Use temporal WITHOUT registration

Or use none + registration instead

### mode = runningdiff + registration ON

This one is subtle.

What happens:

Registration stabilizes stars

Running difference then removes them almost completely

Moving objects pop nicely

BUT:

Any registration error → high-frequency noise

Phase correlation noise gets amplified

You may see “sparkle” or grain

When it does make sense:

You have small, smooth drift

You’re hunting very faint moving dots

You accept a noisier image for detectability

When it does not:

When images are already clean

When registration response (resp) is low (< ~0.3)

When you want accurate centroiding

---
## Recommended workflow (this is the key)

Think of modes as different passes, not one perfect view.

### Step 1: Detect

Use:

runningdiff + registration ON

or temporal + registration OFF

Goal:

Is there something moving at all?

### Step 2: Inspect

Switch to:

none + registration ON

Goal:

Confirm it’s real

Look at shape

Click and track it

### Step 3: Measure

Stay in:

none + registration ON

Goal:

Centroids

Frame-to-frame motion

CSV export

## Data Source

Public NOAA S3 bucket:
s3://noaa-nesdis-swfo-ccor-1-pds/
SWFO/GOES-19/CCOR-1/ccor1-l2/YYYY/MM/DD/

Timestamps are parsed directly from filenames.

---

## Coordinate System

- Pixel coordinates refer to the **processed PNG image**
- Origin `(0,0)` is the **top-left corner**
- X increases to the right
- Y increases downward
- Typical CCOR-1 frame size:
  - `2048 × 1920` pixels

Viewer resizing does **not** affect reported coordinates.

---

## Installation

### Requirements
- Python 3.9+
- Tkinter (included with standard Python distributions)

Install dependencies:

```bash
pip install -r requirements.txt
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

All configuration is handled via the GUI; no command-line arguments are required.


---


## Notes & Limitations

This is a visual analysis and measurement tool, not an automated detection system

Tracking assumes smooth motion between frames

Processed PNGs are not photometrically calibrated

CCOR-1 data is currently considered preliminary per NOAA metadata

---


## License


---


## Acknowledgements

NOAA / NESDIS / SWPC for CCOR-1 data access

The Python scientific ecosystem (Astropy, NumPy, OpenCV, Pillow)