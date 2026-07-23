"""
GRAPHENE FLAKE FINDER
======================
Trackpad-friendly. No mouse required.

VIEWER CONTROLS:
  Trackpad scroll      — pan up/down
  Trackpad horiz scroll— pan left/right
  +  /  -              — zoom in / out (centered on view)
  W A S D              — pan (arrow keys also work)
  F                    — place FLAKE marker at crosshair
  B                    — place suBstrate marker at crosshair
  U                    — undo last marker
  Z                    — zoom to fit
  ENTER                — confirm and continue
  Q                    — quit / skip

A crosshair is always shown at the center — navigate so it sits
on the target, then press F or B to mark it.

MODES:
  python flake_finder.py calibrate
  python flake_finder.py scan
  python flake_finder.py both
"""

import cv2
import numpy as np
import pandas as pd
import os, sys

CAL_CSV        = "calibration_data.csv"
SAMPLE_RADIUS  = 8
SUBSTRATE_GRID = 25
MIN_AREA_PX    = 300
MAX_AREA_PX    = 800000
WINDOW_W       = 1400
WINDOW_H       = 900

# ── color utilities ──────────────────────────────────────────────────────────

def sample_patch(img, x, y, r=SAMPLE_RADIUS):
    h, w = img.shape[:2]
    return img[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)].mean(axis=(0,1))

def auto_substrate(img):
    h, w = img.shape[:2]
    pts = []
    for gy in np.linspace(0.05, 0.95, SUBSTRATE_GRID):
        for gx in np.linspace(0.05, 0.95, SUBSTRATE_GRID):
            y, x = int(gy*h), int(gx*w)
            pts.append(img[max(0,y-4):y+4, max(0,x-4):x+4].mean(axis=(0,1)))
    return np.median(pts, axis=0)

def contrast(flake_bgr, sub_bgr):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sub_bgr > 1, (sub_bgr - flake_bgr) / sub_bgr, 0.0)

# ── zoomable viewer ──────────────────────────────────────────────────────────

class ZoomViewer:
    """
    Pan/zoom viewer controlled entirely by keyboard + trackpad scroll.
    A crosshair sits at the centre of the window; press F/B to mark that point.
    """

    def __init__(self, img, title="Viewer", marker_labels=None, max_markers=None):
        self.img          = img
        self.title        = title
        self.marker_labels = marker_labels or []
        self.max_markers  = max_markers
        self.ih, self.iw  = img.shape[:2]

        # viewport state: (cx,cy) = image coord at screen centre; zoom = img-px per screen-px
        self.cx   = self.iw / 2.0
        self.cy   = self.ih / 2.0
        self.zoom = max(self.iw / WINDOW_W, self.ih / WINDOW_H)

        self.markers = []   # list of (x, y) in image coords
        self.dirty   = True

        # pan speed (image pixels per keypress) scales with zoom
        self.PAN_STEP = 80   # screen pixels equivalent

    # ── coordinate helpers ───────────────────────────────────────────────────

    def centre_img_coords(self):
        """Image coordinates of the crosshair (screen centre)."""
        return int(self.cx), int(self.cy)

    def img_to_screen(self, ix, iy):
        sx = (ix - self.cx) / self.zoom + WINDOW_W / 2
        sy = (iy - self.cy) / self.zoom + WINDOW_H / 2
        return int(sx), int(sy)

    # ── rendering ────────────────────────────────────────────────────────────

    def render(self):
        half_w = WINDOW_W / 2 * self.zoom
        half_h = WINDOW_H / 2 * self.zoom

        x0 = int(max(0, self.cx - half_w))
        y0 = int(max(0, self.cy - half_h))
        x1 = int(min(self.iw, self.cx + half_w))
        y1 = int(min(self.ih, self.cy + half_h))

        crop = self.img[y0:y1, x0:x1]
        tw   = max(1, int((x1 - x0) / self.zoom))
        th   = max(1, int((y1 - y0) / self.zoom))
        view = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        off_x  = max(0, int((x0 - (self.cx - half_w)) / self.zoom))
        off_y  = max(0, int((y0 - (self.cy - half_h)) / self.zoom))
        eh = min(th, WINDOW_H - off_y)
        ew = min(tw, WINDOW_W - off_x)
        if eh > 0 and ew > 0:
            canvas[off_y:off_y+eh, off_x:off_x+ew] = view[:eh, :ew]

        # crosshair at screen centre
        cx_s, cy_s = WINDOW_W // 2, WINDOW_H // 2
        cv2.line(canvas, (cx_s-20, cy_s), (cx_s+20, cy_s), (0,255,255), 1)
        cv2.line(canvas, (cx_s, cy_s-20), (cx_s, cy_s+20), (0,255,255), 1)
        cv2.circle(canvas, (cx_s, cy_s), 6, (0,255,255), 1)

        # placed markers
        colors = [(0,255,0), (0,165,255), (255,100,0), (0,100,255)]
        for i, (mx, my) in enumerate(self.markers):
            sx, sy = self.img_to_screen(mx, my)
            if 0 <= sx < WINDOW_W and 0 <= sy < WINDOW_H:
                col = colors[i % len(colors)]
                cv2.circle(canvas, (sx, sy), 9, col, 2)
                cv2.circle(canvas, (sx, sy), 2, col, -1)
                lab = self.marker_labels[i] if i < len(self.marker_labels) else str(i+1)
                cv2.putText(canvas, lab, (sx+11, sy+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # banner
        n  = len(self.markers)
        mx = self.max_markers
        if mx and n >= mx:
            msg = "All markers placed — press ENTER to confirm, U to undo"
        else:
            nxt = self.marker_labels[n] if (self.marker_labels and n < len(self.marker_labels)) else "point"
            if nxt == "FLAKE":
                msg = f"Navigate to FLAKE, press F to mark  |  +/- zoom  |  WASD/scroll pan  |  U undo"
            elif nxt == "SUBSTRATE":
                msg = f"Navigate to bare SUBSTRATE, press B to mark  |  ENTER when done"
            else:
                msg = f"Navigate to {nxt}, press ENTER to mark  |  +/- zoom  |  WASD pan"

        cv2.rectangle(canvas, (0, 0), (WINDOW_W, 50), (20,20,20), -1)
        cv2.putText(canvas, msg, (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # zoom indicator + image coords of crosshair
        mag = 1.0 / self.zoom * max(self.iw/WINDOW_W, self.ih/WINDOW_H)
        ix, iy = self.centre_img_coords()
        info = f"zoom {mag:.1f}x  |  img pos ({ix}, {iy})"
        cv2.putText(canvas, info, (WINDOW_W-340, WINDOW_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

        cv2.imshow(self.title, canvas)

    # ── mouse (trackpad scroll) ───────────────────────────────────────────────

    def on_mouse(self, event, sx, sy, flags, _):
        # trackpad two-finger scroll → pan
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self.cy -= delta * self.zoom * 40
            self.dirty = True
        elif event == cv2.EVENT_MOUSEHWHEEL:
            delta = 1 if flags > 0 else -1
            self.cx += delta * self.zoom * 40
            self.dirty = True
        # left-click drag still works if user has a mouse
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._drag = True; self._dx, self._dy = sx, sy
            self._cx0, self._cy0 = self.cx, self.cy
        elif event == cv2.EVENT_MOUSEMOVE and getattr(self,'_drag',False):
            self.cx = self._cx0 - (sx - self._dx)*self.zoom
            self.cy = self._cy0 - (sy - self._dy)*self.zoom
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, WINDOW_W, WINDOW_H)
        cv2.setMouseCallback(self.title, self.on_mouse)
        self._drag = False
        self.render()

        pan = self.zoom * self.PAN_STEP   # will be updated each keypress

        while True:
            if self.dirty:
                self.render()
                self.dirty = False
            k = cv2.waitKey(20) & 0xFF

            pan = self.zoom * self.PAN_STEP

            # zoom
            if k == ord('+') or k == ord('='):
                self.zoom = max(0.15, self.zoom / 1.3)
                self.dirty = True
            elif k == ord('-') or k == ord('_'):
                self.zoom = min(self.zoom * 1.3, max(self.iw, self.ih))
                self.dirty = True

            # pan — WASD + arrow keys
            elif k in (ord('w'), 82):   # W or up arrow
                self.cy -= pan; self.dirty = True
            elif k in (ord('s'), 84):   # S or down arrow
                self.cy += pan; self.dirty = True
            elif k in (ord('a'), 81):   # A or left arrow
                self.cx -= pan; self.dirty = True
            elif k in (ord('d'), 83):   # D or right arrow
                self.cx += pan; self.dirty = True

            # zoom to fit
            elif k == ord('z'):
                self.zoom = max(self.iw/WINDOW_W, self.ih/WINDOW_H)
                self.cx, self.cy = self.iw/2, self.ih/2
                self.dirty = True

            # place markers
            elif k == ord('f'):   # mark FLAKE
                if self.max_markers is None or len(self.markers) < self.max_markers:
                    # only place if next expected label is FLAKE (or no labels)
                    n = len(self.markers)
                    expected = self.marker_labels[n] if n < len(self.marker_labels) else None
                    if expected in (None, "FLAKE"):
                        self.markers.append(self.centre_img_coords())
                        self.dirty = True
                    else:
                        print(f"  Expected {expected} next — press B for substrate")

            elif k == ord('b'):   # mark suBstrate
                if self.max_markers is None or len(self.markers) < self.max_markers:
                    n = len(self.markers)
                    expected = self.marker_labels[n] if n < len(self.marker_labels) else None
                    if expected in (None, "SUBSTRATE"):
                        self.markers.append(self.centre_img_coords())
                        self.dirty = True
                    else:
                        print(f"  Expected {expected} next — press F for flake")

            # undo
            elif k == ord('u') and self.markers:
                self.markers.pop(); self.dirty = True

            # confirm / quit
            elif k in (13, ord('\r'), ord('\n')):
                break
            elif k == ord('q'):
                self.markers = []
                break

        cv2.destroyAllWindows()
        return self.markers

# ── calibration ──────────────────────────────────────────────────────────────

def run_calibrate(image_path=None):
    print("\n── CALIBRATION MODE ─────────────────────────────────────────")
    print("Controls: +/- zoom  |  WASD or arrow keys pan  |  trackpad scroll pans")
    print("          F = mark flake  |  B = mark substrate  |  U = undo  |  ENTER = confirm\n")

    if image_path is None:
        image_path = input("Path to chip mosaic (with known AFM flakes): ").strip().strip('"')
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}"); return

    print("Loading image...")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Could not read image."); return
    h, w = img.shape[:2]
    print(f"  Loaded: {w}×{h} px")

    records = []
    if os.path.exists(CAL_CSV):
        df_ex = pd.read_csv(CAL_CSV)
        records = df_ex.to_dict("records")
        print(f"  Loaded {len(records)} existing records.\n")

    img_name = os.path.basename(image_path)

    while True:
        t_str = input("AFM thickness for next flake (nm) — blank to stop: ").strip()
        if not t_str: break
        try:
            t = float(t_str)
        except ValueError:
            print("  Not a number."); continue

        print(f"\n  → Navigate to the {t} nm flake")
        print(f"     Press F when crosshair is on the FLAKE")
        print(f"     Press B when crosshair is on bare SUBSTRATE nearby")
        print(f"     Press ENTER to confirm, U to undo, Q to skip\n")

        viewer = ZoomViewer(img,
                            title=f"Thickness {t} nm — F=flake  B=substrate  ENTER=confirm",
                            marker_labels=["FLAKE", "SUBSTRATE"],
                            max_markers=2)
        pts = viewer.run()

        if len(pts) < 2:
            print("  Skipped.\n"); continue

        fx, fy = pts[0]
        sx, sy = pts[1]
        f_bgr  = sample_patch(img, fx, fy)
        s_bgr  = sample_patch(img, sx, sy)
        c      = contrast(f_bgr, s_bgr)

        rec = {
            "filename":     img_name,
            "thickness_nm": t,
            "flake_R": round(f_bgr[2],1), "flake_G": round(f_bgr[1],1), "flake_B": round(f_bgr[0],1),
            "sub_R":   round(s_bgr[2],1), "sub_G":   round(s_bgr[1],1), "sub_B":   round(s_bgr[0],1),
            "contrast_R": round(c[2],4),
            "contrast_G": round(c[1],4),
            "contrast_B": round(c[0],4),
        }
        records.append(rec)
        pd.DataFrame(records).to_csv(CAL_CSV, index=False)
        print(f"  ✓ flake RGB=({f_bgr[2]:.0f},{f_bgr[1]:.0f},{f_bgr[0]:.0f})  "
              f"sub RGB=({s_bgr[2]:.0f},{s_bgr[1]:.0f},{s_bgr[0]:.0f})  "
              f"contrast RGB=({c[2]:.3f},{c[1]:.3f},{c[0]:.3f})  "
              f"[{len(records)} total]\n")

    if records:
        df = pd.DataFrame(records)
        print(f"\nCalibration summary ({len(df)} flakes):")
        print(df[["filename","thickness_nm","contrast_R","contrast_G","contrast_B"]].to_string(index=False))

    return image_path

# ── scan ─────────────────────────────────────────────────────────────────────

def load_calibration(tmin=20, tmax=45):
    if not os.path.exists(CAL_CSV):
        print(f"ERROR: '{CAL_CSV}' not found."); sys.exit(1)
    df     = pd.read_csv(CAL_CSV)
    target = df[(df["thickness_nm"] >= tmin) & (df["thickness_nm"] <= tmax)]
    if target.empty:
        print("  No flakes in range, using all."); target = df
    print(f"  Using {len(target)} calibration flakes ({tmin}-{tmax} nm)")
    mean_c = np.array([target["contrast_B"].mean(),
                       target["contrast_G"].mean(),
                       target["contrast_R"].mean()])
    std_c  = np.array([target["contrast_B"].std(),
                       target["contrast_G"].std(),
                       target["contrast_R"].std()])
    print(f"  Target contrast BGR: {mean_c.round(3)} ± {std_c.round(3)}")
    return mean_c, std_c

def build_mask(img, sub_bgr, target_c, tolerance, n_strips=20):
    """
    Process image in horizontal strips to keep memory under control.
    An 18000x16000 image needs ~4GB naively; strips keep each chunk ~200MB.
    Overlap strips by a few rows so morphological ops don't leave seam artifacts.
    """
    h, w   = img.shape[:2]
    sub_f  = sub_bgr.astype(np.float32)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = np.zeros((h, w), dtype=np.uint8)

    strip_h = max(1, h // n_strips)
    overlap = 10   # rows of overlap between strips to avoid seam artifacts

    for i in range(n_strips):
        y0 = i * strip_h
        y1 = min(h, y0 + strip_h + overlap)
        if y0 >= h:
            break

        strip  = img[y0:y1].astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            c_strip = np.where(sub_f > 1, (sub_f - strip) / sub_f, 0.0)

        dist        = np.linalg.norm(c_strip - target_c, axis=2)
        strip_mask  = (dist < tolerance).astype(np.uint8) * 255

        # morphological cleanup per strip
        strip_mask = cv2.morphologyEx(strip_mask, cv2.MORPH_OPEN,  k, iterations=1)
        strip_mask = cv2.morphologyEx(strip_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # write only the non-overlap rows (except last strip)
        write_end = y1 - overlap if (i < n_strips-1 and y1 < h) else y1
        mask[y0:write_end] = strip_mask[:write_end-y0]

        del strip, c_strip, dist, strip_mask

        pct = min(100, int((i+1)/n_strips*100))
        print(f"\r  Processing... {pct}%", end="", flush=True)

    print()  # newline after progress
    return mask

def score_region(region_bgr, sub_bgr, target_c, std_c):
    c    = contrast(region_bgr, sub_bgr)
    diff = np.abs(c - target_c)
    norm = diff / (std_c + 0.05)
    return max(0.0, round(100 - np.linalg.norm(norm)*35, 1))

def run_scan(image_path=None, tmin=None, tmax=None):
    print("\n── SCAN MODE ────────────────────────────────────────────────")

    if image_path is None:
        image_path = input("Path to chip mosaic to scan: ").strip().strip('"')
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}"); return

    if tmin is None:
        t_str = input("Thickness range (min max nm, default 20 45): ").strip()
        if t_str:
            parts = t_str.split(); tmin, tmax = float(parts[0]), float(parts[1])
        else:
            tmin, tmax = 20, 45

    target_c, std_c = load_calibration(tmin, tmax)

    print("Loading mosaic...")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Cannot read image."); return
    h, w = img.shape[:2]
    print(f"  {w}×{h} px")

    print("Detecting substrate color...")
    sub_bgr = auto_substrate(img)
    print(f"  Substrate BGR: {sub_bgr.round(1)}")

    tol_str    = input("Tolerance (default 0.15, raise to 0.20+ if too few results): ").strip()
    tolerance  = float(tol_str) if tol_str else 0.15
    min_px_str = input("Min flake size in pixels (default 300): ").strip()
    min_area   = int(min_px_str) if min_px_str.isdigit() else MIN_AREA_PX

    print("\nBuilding detection mask (may take ~30s for large images)...")
    mask = build_mask(img, sub_bgr, target_c, tolerance)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  {len(contours)} raw regions, filtering...")

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > MAX_AREA_PX: continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = 3
        roi  = img[max(0,y-pad):min(h,y+bh+pad), max(0,x-pad):min(w,x+bw+pad)]
        mean_bgr = roi.mean(axis=(0,1))
        sc = score_region(mean_bgr, sub_bgr, target_c, std_c)
        candidates.append({
            "rank":0, "score":sc, "area_px":int(area),
            "width_px":bw, "height_px":bh,
            "centre_x":x+bw//2, "centre_y":y+bh//2,
            "bbox_x1":x,"bbox_y1":y,"bbox_x2":x+bw,"bbox_y2":y+bh,
            "mean_R":round(mean_bgr[2],1),
            "mean_G":round(mean_bgr[1],1),
            "mean_B":round(mean_bgr[0],1),
        })

    candidates.sort(key=lambda c:(c["score"],c["area_px"]), reverse=True)
    for i, c in enumerate(candidates): c["rank"] = i+1
    print(f"  {len(candidates)} candidates kept")

    # annotate output image
    out_img = img.copy()
    fs = max(0.6, min(w,h)/10000)
    for c in candidates:
        x1,y1,x2,y2 = c["bbox_x1"],c["bbox_y1"],c["bbox_x2"],c["bbox_y2"]
        color = (0,220,0) if c["score"]>=70 else (0,200,255) if c["score"]>=45 else (0,80,255)
        lw    = max(3, int(fs*8))
        cv2.rectangle(out_img,(x1,y1),(x2,y2),color,lw)
        label = f"#{c['rank']} {c['score']:.0f}"
        cv2.putText(out_img,label,(x1+2,max(y1-8,30)),
                    cv2.FONT_HERSHEY_SIMPLEX,fs*2.5,(0,0,0),max(2,int(fs*6)))
        cv2.putText(out_img,label,(x1,  max(y1-10,28)),
                    cv2.FONT_HERSHEY_SIMPLEX,fs*2.5,color,max(2,int(fs*6)))

    base         = os.path.splitext(os.path.basename(image_path))[0]
    out_img_path = f"{base}_candidates.jpg"
    out_csv_path = "candidates.csv"
    cv2.imwrite(out_img_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    pd.DataFrame(candidates).to_csv(out_csv_path, index=False)

    print(f"\n✓ Annotated image → '{out_img_path}'")
    print(f"✓ Candidate list  → '{out_csv_path}'")
    if candidates:
        df = pd.DataFrame(candidates)
        print(f"\nTop 10:")
        print(df[["rank","score","area_px","centre_x","centre_y"]].head(10).to_string(index=False))

    # zoomable results viewer
    view = input("\nOpen zoomable results viewer? (y/n): ").strip().lower()
    if view == 'y':
        print("  +/- zoom  |  WASD/scroll pan  |  ENTER or Q to quit")
        viewer = ZoomViewer(out_img, title="Results — +/- zoom  WASD pan  Q quit")
        viewer.run()

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode not in ("calibrate","scan","both"):
        print(__doc__)
        mode = input("Mode (calibrate / scan / both): ").strip().lower()

    if mode == "calibrate":
        run_calibrate()
    elif mode == "scan":
        run_scan()
    elif mode == "both":
        mosaic = input("Path to chip mosaic: ").strip().strip('"')
        run_calibrate(image_path=mosaic)
        t_str = input("\nThickness range to scan (min max, default 20 45): ").strip()
        if t_str:
            parts = t_str.split(); tmin, tmax = float(parts[0]), float(parts[1])
        else:
            tmin, tmax = 20, 45
        run_scan(image_path=mosaic, tmin=tmin, tmax=tmax)

if __name__ == "__main__":
    main()
