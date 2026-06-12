"""Seismic Workbench — M1 pre-stack pipeline + M2 post-stack attributes.

Run with:  streamlit run app/streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seisproc.attributes import ATTRIBUTES, compute
from seisproc.nmo import nmo_correct, stack, velocity_function
from seisproc.processing import agc, bandpass, normalize
from seisproc.segy import SegyData, read_segy
from seisproc.synthetic import cmp_gather, synthetic_volume
from seisproc.velocity import pick_velocities, semblance
from seisproc.viz import density, semblance_panel, wiggle
from seisproc.volume import Volume, from_npy, read_segy_volume

st.set_page_config(page_title="Seismic Workbench", layout="wide")
st.title("Seismic Workbench")

mode = st.sidebar.radio("Mode", ["Pre-stack (gathers)", "Post-stack (volume)"])


# ============================================================ pre-stack mode
def prestack_mode():
    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Synthetic CMP gather", "Upload SEG-Y"])

        if source == "Synthetic CMP gather":
            noise = st.slider("Noise level", 0.0, 0.5, 0.1, 0.05)
            if st.button("Generate") or "segy" not in st.session_state:
                data, t, offsets = cmp_gather(noise=noise)
                st.session_state.segy = SegyData(
                    data=data.astype(np.float32), dt=0.002, offsets=offsets,
                    text_header="synthetic 5-layer CMP gather",
                )
                st.session_state.pop("picks", None)
        else:
            upload = st.file_uploader("SEG-Y file", type=["sgy", "segy"])
            if upload is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sgy") as tmp:
                    tmp.write(upload.getvalue())
                st.session_state.segy = read_segy(tmp.name)
                st.session_state.pop("picks", None)

        if "segy" not in st.session_state:
            st.stop()
        sd: SegyData = st.session_state.segy

        st.header("Processing chain")
        use_bp = st.checkbox("Bandpass", value=False)
        if use_bp:
            nyq = 0.5 / sd.dt
            bp_low, bp_high = st.slider("Corners (Hz)", 1.0, float(int(nyq)), (10.0, 60.0))
        use_agc = st.checkbox("AGC", value=False)
        if use_agc:
            agc_win = st.slider("AGC window (s)", 0.1, 1.5, 0.5, 0.1)
        use_norm = st.checkbox("Normalize (RMS)", value=False)

    proc = sd.data.copy()
    if use_bp:
        proc = bandpass(proc, sd.dt, bp_low, bp_high)
    if use_agc:
        proc = agc(proc, sd.dt, agc_win)
    if use_norm:
        proc = normalize(proc, "rms")

    with st.expander("File metadata", expanded=False):
        st.json(sd.summary())
        if sd.text_header:
            st.code(sd.text_header[:2000], language=None)

    tab_view, tab_vel, tab_stack = st.tabs(["Gather viewer", "Velocity analysis", "NMO & Stack"])

    with tab_view:
        col1, col2 = st.columns([1, 4])
        with col1:
            style = st.radio("Display", ["Wiggle", "Variable density"])
            gain = st.slider("Gain", 0.2, 5.0, 1.0, 0.2)
        with col2:
            fig, ax = plt.subplots(figsize=(9, 6))
            if style == "Wiggle":
                wiggle(proc, sd.dt, offsets=sd.offsets, ax=ax, gain=gain)
            else:
                density(proc * gain, sd.dt, offsets=sd.offsets, ax=ax)
            st.pyplot(fig, clear_figure=True)

    has_offsets = len(sd.offsets) > 1 and sd.offsets.max() > 0

    with tab_vel:
        if not has_offsets:
            st.info("No offset information in trace headers — this looks like "
                    "post-stack data. Velocity analysis needs pre-stack gathers.")
        else:
            c1, c2 = st.columns([1, 3])
            with c1:
                v_min, v_max = st.slider("Velocity scan (m/s)", 1000, 6000, (1300, 3300), 100)
                v_step = st.select_slider("Step (m/s)", [10, 25, 50, 100], value=25)
                smooth = st.slider("Semblance window (s)", 0.01, 0.10, 0.04, 0.01)
                auto = st.button("Auto-pick velocities")

            vels = np.arange(float(v_min), float(v_max), float(v_step))
            spec = semblance(proc, sd.offsets, sd.dt, vels, smooth=smooth)

            if auto or "picks" not in st.session_state:
                times, pvels, _ = pick_velocities(spec, vels, sd.dt)
                st.session_state.picks = pd.DataFrame({"t0 (s)": times, "v_rms (m/s)": pvels})

            with c1:
                st.caption("Edit picks directly — add or delete rows.")
                st.session_state.picks = st.data_editor(
                    st.session_state.picks, num_rows="dynamic", use_container_width=True
                )

            picks = st.session_state.picks.dropna()
            with c2:
                fig, ax = plt.subplots(figsize=(6, 6))
                overlay = None
                if len(picks):
                    overlay = (picks["t0 (s)"].to_numpy(), picks["v_rms (m/s)"].to_numpy())
                semblance_panel(spec, vels, sd.dt, picks=overlay, ax=ax)
                st.pyplot(fig, clear_figure=True)

    with tab_stack:
        if not has_offsets:
            st.info("Pre-stack gathers required.")
        elif "picks" not in st.session_state or not len(st.session_state.picks.dropna()):
            st.info("Pick a velocity function in the Velocity analysis tab first.")
        else:
            picks = st.session_state.picks.dropna()
            mute = st.slider("Stretch mute", 0.1, 1.0, 0.3, 0.05)
            v_t = velocity_function(sd.t, picks["t0 (s)"], picks["v_rms (m/s)"])
            corrected = nmo_correct(proc, sd.offsets, sd.dt, v_t, stretch_mute=mute)
            stk = stack(corrected)

            c1, c2 = st.columns([3, 1])
            with c1:
                fig, ax = plt.subplots(figsize=(8, 6))
                wiggle(corrected, sd.dt, offsets=sd.offsets, ax=ax)
                ax.set_title("NMO-corrected gather")
                st.pyplot(fig, clear_figure=True)
            with c2:
                fig, ax = plt.subplots(figsize=(2.5, 6))
                ax.plot(stk, sd.t, "k", linewidth=0.8)
                ax.fill_betweenx(sd.t, 0, stk, where=stk > 0, color="k", linewidth=0)
                ax.set_ylim(sd.t[-1], 0)
                ax.set_title("Stack")
                ax.set_ylabel("Time (s)")
                st.pyplot(fig, clear_figure=True)


# =========================================================== post-stack mode
SIGNED_ATTRS = {"Amplitude", "Instantaneous phase"}  # diverging colour scale


def _load_volume(vol: Volume):
    st.session_state.volume = vol
    st.session_state.attr_cache = {}


def _attr_volume(name: str) -> np.ndarray:
    cache = st.session_state.setdefault("attr_cache", {})
    if name not in cache:
        vol: Volume = st.session_state.volume
        with st.spinner(f"Computing {name} ..."):
            cache[name] = compute(name, vol.data, vol.dt)
    return cache[name]


def poststack_mode():
    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Synthetic 3D volume", "Upload SEG-Y (3D)", "F3 .npy volume"])

        if source == "Synthetic 3D volume":
            if st.button("Generate") or "volume" not in st.session_state:
                data, t = synthetic_volume()
                _load_volume(Volume(
                    data=data.astype(np.float32),
                    ilines=np.arange(100, 100 + data.shape[0]),
                    xlines=np.arange(300, 300 + data.shape[1]),
                    dt=0.004,
                ))
        elif source == "Upload SEG-Y (3D)":
            upload = st.file_uploader("3D SEG-Y with inline/crossline headers", type=["sgy", "segy"])
            if upload is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sgy") as tmp:
                    tmp.write(upload.getvalue())
                try:
                    _load_volume(read_segy_volume(tmp.name))
                except RuntimeError as e:
                    st.error(f"Could not infer 3D geometry: {e}")
        else:
            st.caption("Get it with: python scripts/download_f3.py")
            npy_path = st.text_input(".npy path", "data/f3/data/train/train_seismic.npy")
            if st.button("Load") and npy_path:
                p = Path(npy_path)
                if p.exists():
                    _load_volume(from_npy(str(p), dt=0.004, first_iline=100, first_xline=300))
                else:
                    st.error(f"Not found: {p.resolve()}")

        if "volume" not in st.session_state:
            st.stop()
        vol: Volume = st.session_state.volume

        st.header("View")
        view = st.radio("Slice", ["Inline", "Crossline", "Time slice"])
        attr_name = st.selectbox("Attribute", list(ATTRIBUTES.keys()))

    with st.expander("Volume metadata", expanded=False):
        st.json(vol.summary())

    attr = _attr_volume(attr_name)
    signed = attr_name in SIGNED_ATTRS

    if view == "Inline":
        il = st.slider("Inline", int(vol.ilines[0]), int(vol.ilines[-1]), int(vol.ilines[len(vol.ilines) // 2]))
        section = attr[int(np.where(vol.ilines == il)[0][0])]
        xlabel, xaxis = "Crossline", vol.xlines
    elif view == "Crossline":
        xl = st.slider("Crossline", int(vol.xlines[0]), int(vol.xlines[-1]), int(vol.xlines[len(vol.xlines) // 2]))
        section = attr[:, int(np.where(vol.xlines == xl)[0][0])]
        xlabel, xaxis = "Inline", vol.ilines
    else:
        t_slice = st.slider("Time (s)", 0.0, float(vol.t[-1]), float(vol.t[-1] / 2), float(vol.dt))
        section = attr[:, :, int(round(t_slice / vol.dt))]

    fig, ax = plt.subplots(figsize=(9, 6))
    if view == "Time slice":
        vmax = np.percentile(np.abs(section), 98) or 1.0
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax) if signed else dict(cmap="viridis", vmin=0, vmax=vmax)
        im = ax.imshow(section.T, aspect="auto", origin="lower",
                       extent=[vol.ilines[0], vol.ilines[-1], vol.xlines[0], vol.xlines[-1]], **kw)
        ax.set_xlabel("Inline")
        ax.set_ylabel("Crossline")
    else:
        vmax = np.percentile(np.abs(section), 98) or 1.0
        kw = dict(cmap="gray_r", vmin=-vmax, vmax=vmax) if signed else dict(cmap="viridis", vmin=0, vmax=vmax)
        im = ax.imshow(section.T, aspect="auto",
                       extent=[xaxis[0], xaxis[-1], vol.t[-1], 0], **kw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time (s)")
    fig.colorbar(im, ax=ax, label=attr_name)
    st.pyplot(fig, clear_figure=True)


if mode == "Pre-stack (gathers)":
    prestack_mode()
else:
    poststack_mode()
