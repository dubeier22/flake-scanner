"""Local web interface for flake-scanner (run via `flake-scanner ui`).

A small Streamlit app: set the material and target thickness, pick a mosaic,
and get the ranked candidate map + table; or add calibration flakes from an
annotated (boxed) mosaic. Everything runs locally so full-resolution mosaics
never leave the machine.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from flake_scanner.calibration.annotations import FlakeAnnotation, annotate_preview
from flake_scanner.calibration.build import build_from_id_mosaic, read_annotations
from flake_scanner.calibration.store import CalibrationStore
from flake_scanner.detection.scan import scan as scan_mosaic
from flake_scanner.io.report import write_csv, write_map

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _images_in(folder: str) -> list[str]:
    p = Path(folder).expanduser()
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def _materials(db_path: str) -> list[str]:
    p = Path(db_path)
    if not p.exists():
        return []
    store = CalibrationStore(p)
    return sorted({r.material for r in store.records})


def _substrates(db_path: str, material: str) -> list[str]:
    p = Path(db_path)
    if not p.exists():
        return ["SiO2-285nm"]
    store = CalibrationStore(p)
    subs = sorted({r.substrate for r in store.records if r.material == material})
    return subs or ["SiO2-285nm"]


def _pick_image(label: str, key: str) -> str | None:
    """Folder + dropdown image picker with a manual-path fallback."""
    folder = st.text_input(f"{label} folder", value="Full Chips", key=f"{key}_folder")
    files = _images_in(folder)
    if files:
        choice = st.selectbox(
            f"{label} image", files, format_func=lambda p: Path(p).name, key=f"{key}_sel"
        )
        return choice
    manual = st.text_input(f"{label} image path", key=f"{key}_path")
    return manual or None


def scan_tab() -> None:
    st.header("Scan a mosaic")
    db = st.text_input("Calibration database", value="calibration.csv", key="scan_db")
    mats = _materials(db)
    if not mats:
        st.warning(f"No calibration database at '{db}'. Add flakes in the Calibrate tab first.")
        return
    c1, c2 = st.columns(2)
    material = c1.selectbox("Material", mats)
    substrate = c2.selectbox("Substrate", _substrates(db, material))
    c3, c4 = st.columns(2)
    tmin = c3.number_input("Min thickness (nm)", value=35.0, step=1.0)
    tmax = c4.number_input("Max thickness (nm)", value=45.0, step=1.0)
    c5, c6, c7 = st.columns(3)
    pixel_scale = c5.number_input("Pixel scale (µm/px)", value=0.30, step=0.01, format="%.3f")
    min_size = c6.number_input("Min flake size (µm)", value=23.0, step=1.0)
    threshold = c7.slider("Match strictness (lower = stricter)", 1.0, 4.0, 2.0, 0.1)

    image = _pick_image("Mosaic to scan", "scan")

    if st.button("Run scan", type="primary") and image:
        try:
            sig = CalibrationStore(db).signature(material, substrate, tmin, tmax)
        except ValueError as e:
            st.error(str(e))
            return
        st.caption(f"Signature: {sig.n_flakes} calibration flakes, mean ndi {sig.mean.round(3)}")
        img = cv2.imread(image)
        if img is None:
            st.error(f"Could not read image: {image}")
            return
        with st.spinner("Scanning… (a full mosaic takes ~1 minute)"):
            cands = scan_mosaic(img, sig, pixel_scale, min_size_um=min_size, match_threshold=threshold)
        st.success(f"{len(cands)} candidate flakes found.")

        out = Path("scan_results")
        stem = Path(image).stem
        write_csv(cands, out / f"{stem}_candidates.csv")
        map_path = out / f"{stem}_candidate_map.jpg"
        write_map(img, cands, map_path)

        st.image(str(map_path), caption="Candidates (ranked by colour match, numbered)", use_container_width=True)
        df = pd.DataFrame(
            [
                {"rank": c.rank, "size µm": f"{c.w_um:.0f}×{c.h_um:.0f}", "match": round(c.maha, 2),
                 "x": c.cx, "y": c.cy}
                for c in cands
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download candidates CSV",
            (out / f"{stem}_candidates.csv").read_bytes(),
            file_name=f"{stem}_candidates.csv",
        )


def calibrate_tab() -> None:
    st.header("Add calibration flakes")
    st.caption(
        "In your image editor, draw a **magenta (#FF00FF) box** around each AFM-measured "
        "flake and write its **flake-ID number in blue (#0000FF)** next to the box (no "
        "connecting lines). Two colours keep the box and number from interfering. Enter "
        "the matching thicknesses below."
    )
    db = st.text_input("Calibration database", value="calibration.csv", key="cal_db")
    c1, c2, c3, c4 = st.columns(4)
    material = c1.text_input("Material", value="HBN")
    date = c2.text_input("Date (YYYY_MM_DD)", value="2026_07_07")
    chip = c3.text_input("Chip letter", value="A")
    substrate = c4.text_input("Substrate", value="SiO2-285nm")
    thick_str = st.text_input(
        "Thicknesses (nm), comma-separated in flake-ID order (flake 1, 2, …)",
        placeholder="e.g. 25,26,250,105,…",
    )
    image = _pick_image("Annotated mosaic", "cal")

    if st.button("Read flakes from map", type="primary") and image:
        with st.spinner("Reading magenta annotations…"):
            try:
                anns = read_annotations(image)
            except RuntimeError as e:
                st.error(str(e))
                return
        st.session_state["cal_anns"] = [(a.flake_id, a.x, a.y, a.w, a.h) for a in anns]
        st.session_state["cal_image"] = image

    anns_state = st.session_state.get("cal_anns")
    if anns_state and st.session_state.get("cal_image") == image:
        anns = [FlakeAnnotation(fid, x, y, w, h) for fid, x, y, w, h in anns_state]
        st.write(
            f"**Detected {len(anns)} flakes**, listed top-to-bottom. Each box on the map is "
            "labelled `position: ID` — check the ID against your blue number and fix it here."
        )
        st.image(annotate_preview(cv2.imread(image), anns), caption="Detected boxes (position: ID)",
                 use_container_width=True)
        df = pd.DataFrame(
            [{"position": i, "detected_id": a.flake_id, "x": a.x, "y": a.y, "w": a.w, "h": a.h}
             for i, a in enumerate(anns, start=1)]
        )
        edited = st.data_editor(
            df, use_container_width=True, hide_index=True, key="cal_edit",
            column_config={"position": st.column_config.NumberColumn(disabled=True),
                           "x": None, "y": None, "w": None, "h": None},
        )
        if st.button("Save to database", type="primary"):
            tmap = {i: float(v) for i, v in enumerate(thick_str.split(","), 1) if v.strip()}
            fixed = [
                FlakeAnnotation(int(r["detected_id"]) if pd.notna(r["detected_id"]) else None,
                                int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
                for _, r in edited.iterrows()
            ]
            store = CalibrationStore(db)
            results = build_from_id_mosaic(image, store, material, date, chip, substrate, tmap, fixed)
            store.save()
            added = [r for r in results if r.added]
            st.success(f"Added/updated {len(added)} flakes → {db} ({len(store.records)} total).")
            missing = sorted(set(tmap) - {r.id_number for r in results if r.added})
            if missing:
                st.warning(f"No flake saved for IDs {missing} — fix the ID or thickness list.")
            st.session_state.pop("cal_anns", None)


def main() -> None:
    st.set_page_config(page_title="Flake Scanner", page_icon="🔬", layout="wide")
    st.title("🔬 Flake Scanner")
    scan, calib = st.tabs(["Scan", "Calibrate"])
    with scan:
        scan_tab()
    with calib:
        calibrate_tab()


if __name__ == "__main__":
    main()
