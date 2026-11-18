"""Streamlit 3-D quantum transport explorer. Run: streamlit run app.py"""
import numpy as np
from PIL import Image
import io
import plotly.graph_objects as go
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from DisorderTransport import (SimulationConfig, gaussian_wavepacket,
    generate_potential, propagate)

st.set_page_config(page_title="DisorderTransport | 3D Lab", layout="wide")
st.title("DisorderTransport")
st.caption("Three-dimensional quantum wavepacket laboratory · Split-step Fourier method")
st.info("Research preview: 3-D runs are a new extension. The archived 192-run dataset is a separate, original 2-D experiment.")

with st.sidebar:
    st.header("Simulation setup")
    n = st.select_slider("Cubic grid resolution", options=[32, 48, 64, 80, 96, 128], value=48)
    length = st.number_input("Domain length", min_value=8.0, max_value=100.0, value=40.0)
    dt = st.number_input("Time step", min_value=0.001, max_value=0.05, value=0.01, format="%.3f")
    steps = st.slider("Time steps", 20, 500, 150, 10)
    snapshot_every = st.slider("Save frame every", 1, 25, 5)
    family = st.selectbox("Potential family", ["custom", "random", "multiscale", "binary", "ordered"])
    level = st.slider("Correlation / smoothing level", 0, 7, 4)
    amplitude = st.slider("Potential amplitude", 0.0, 3.0, 0.8, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=20260903)
    st.divider(); st.subheader("Wavepacket")
    sigma = st.slider("Gaussian width σ", 0.8, 6.0, 2.0, 0.1)
    kx = st.slider("Initial momentum kx", -4.0, 4.0, 2.0, 0.1)
    ky = st.slider("Initial momentum ky", -4.0, 4.0, 0.0, 0.1)
    kz = st.slider("Initial momentum kz", -4.0, 4.0, 0.0, 0.1)
    absorber = st.checkbox("Enable edge absorber", value=True)
    run = st.button("▶ Run 3D simulation", type="primary", use_container_width=True)

if "custom_volume" not in st.session_state or st.session_state.custom_volume.shape != (n,n,n):
    st.session_state.custom_volume = np.zeros((n,n,n), dtype=float)

st.subheader("Build a 3-D potential")
st.write("Draw on individual XY slices, then move through z to build a volumetric landscape. Brush intensity sets the potential value.")
col_a, col_b = st.columns([1, 2])
with col_a:
    z_slice = st.slider("Z slice", 0, n-1, n//2)
    brush = st.slider("Brush intensity", -3.0, 3.0, 1.0, 0.1)
    brush_width = st.slider("Brush width", 1, 40, 8)
    if st.button("Clear custom volume"):
        st.session_state.custom_volume[:] = 0
with col_b:
    if family == "custom":
        canvas = st_canvas(fill_color="rgba(255, 0, 0, 0.15)", stroke_width=brush_width,
            stroke_color="#ff5533", background_color="#101827", height=320, width=320,
            drawing_mode="freedraw", key=f"canvas-{n}-{z_slice}")
        if canvas.image_data is not None:
            rgba = canvas.image_data
            # Convert painted alpha to a signed scalar potential on this z plane.
            rgb = rgba[:, :, :3].astype(float)
            background = np.array([16., 24., 39.])
            strength = np.clip(np.linalg.norm(rgb-background, axis=2) / 300.0, 0, 1)
            plane_img = Image.fromarray((strength*255).astype("uint8")).resize((n,n), Image.Resampling.BILINEAR)
            st.session_state.custom_volume[:, :, z_slice] = np.asarray(plane_img, dtype=float).T / 255.0 * brush
    else:
        st.caption("Select Custom to draw. Generated families are constructed reproducibly from the seed and controls.")

if run:
    cfg = SimulationConfig(n=n, length=float(length), dt=float(dt), n_steps=steps,
                           snapshot_every=snapshot_every)
    with st.spinner(f"Propagating {n}³ grid for {steps} steps…"):
        potential = (st.session_state.custom_volume if family == "custom" else
                     generate_potential(cfg, family, seed=int(seed), level=level, amplitude=amplitude))
        psi0 = gaussian_wavepacket(cfg, sigma=sigma, momentum=(kx,ky,kz))
        result = propagate(psi0, potential, cfg, absorber=absorber)
        st.session_state.result = result
        st.session_state.potential = potential
        st.session_state.config = cfg

if "result" in st.session_state:
    result = st.session_state.result
    potential = st.session_state.potential
    cfg = st.session_state.config
    st.divider(); st.subheader("Wavepacket evolution")
    frame = st.slider("Simulation time frame", 0, len(result.times)-1, 0)
    density = np.abs(result.wavefunctions[frame])**2
    # Volume rendering displays the 3-D probability density, not a 2-D projection.
    # Downsample only the visualization (never the underlying simulation) to keep
    # browser payloads responsive at high solver resolutions.
    stride = max(1, int(np.ceil(cfg.n / 32)))
    coords = np.linspace(-cfg.length/2, cfg.length/2, cfg.n, endpoint=False)[::stride]
    view = density[::stride, ::stride, ::stride]
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    volume_args = dict(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
        isomin=float(view.max())*0.08, isomax=float(view.max())*0.85,
        opacity=0.08, surface_count=18, colorscale="Viridis",
        caps=dict(x_show=False,y_show=False,z_show=False))
    base_volume = go.Volume(value=view.ravel(), **volume_args)
    animation_stride = max(1, int(np.ceil(len(result.times) / 30)))
    plot_frames = []
    for idx in range(0, len(result.times), animation_stride):
        frame_density = np.abs(result.wavefunctions[idx, ::stride, ::stride, ::stride])**2
        plot_frames.append(go.Frame(
            data=[go.Volume(value=frame_density.ravel(), **volume_args)],
            name=f"{idx}",
            layout=go.Layout(title=f"Probability density | t = {result.times[idx]:.3f}")))
    fig = go.Figure(data=[base_volume], frames=plot_frames)
    fig.update_layout(height=560,
        scene=dict(aspectmode="cube", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        title=f"Probability density | t = {result.times[frame]:.3f}",
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.08,
            buttons=[dict(label="▶ Play", method="animate", args=[None, {
                "frame": {"duration": 100, "redraw": True}, "fromcurrent": True,
                "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate", args=[[None], {
                    "frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                    "transition": {"duration": 0}}])])])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Probability norm", f"{result.norms[frame]:.10f}")
        st.metric("3-D mean-square displacement", f"{result.msd[frame]:.5f}")
        st.metric("Inverse participation ratio", f"{result.ipr[frame]:.6g}")
        st.metric("Edge probability", f"{result.edge_probability[frame]:.5f}")
        st.caption("With the absorber enabled, norm decreases as probability is absorbed near the boundaries.")
    st.subheader("Potential landscape cross-section")
    potential_slice = st.slider("Potential z slice", 0, cfg.n-1, cfg.n//2)
    st.plotly_chart(go.Figure(go.Heatmap(z=potential[:, :, potential_slice].T,
        colorscale="RdBu", colorbar=dict(title="V"))).update_layout(
            height=300, xaxis_title="x grid index", yaxis_title="y grid index"),
        use_container_width=True)
    st.subheader("Transport diagnostics")
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=result.times, y=result.msd, name="3-D MSD"))
    chart.add_trace(go.Scatter(x=result.times, y=result.ipr, name="IPR", yaxis="y2"))
    chart.update_layout(height=320, xaxis_title="Time", yaxis=dict(title="MSD"),
                        yaxis2=dict(title="IPR", overlaying="y", side="right"))
    st.plotly_chart(chart, use_container_width=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, times=result.times, norms=result.norms, msd=result.msd,
                        ipr=result.ipr, edge_probability=result.edge_probability)
    st.download_button("Download diagnostics (NPZ)", data=buffer.getvalue(),
                       file_name="transport_diagnostics.npz")
else:
    st.markdown("Choose parameters and run a simulation. For 3-D grids, memory and runtime grow rapidly with resolution (approximately as N³ for arrays and N³ log N per FFT).")
