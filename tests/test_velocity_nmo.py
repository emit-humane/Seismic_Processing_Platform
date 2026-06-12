"""End-to-end validation: semblance recovers true v_rms, NMO flattens
events, stacking improves SNR. These are the M1 exit criteria."""

import numpy as np

from seisproc.nmo import nmo_correct, stack, velocity_function
from seisproc.synthetic import DEFAULT_LAYERS, cmp_gather
from seisproc.velocity import pick_velocities, semblance

DT = 0.002
VELOCITIES = np.arange(1300.0, 3300.0, 25.0)


def _true_v_at(t0):
    times = [layer.t0 for layer in DEFAULT_LAYERS]
    vels = [layer.v_rms for layer in DEFAULT_LAYERS]
    return np.interp(t0, times, vels)


def test_semblance_peaks_recover_true_velocities():
    data, t, offsets = cmp_gather(dt=DT, noise=0.02)
    spec = semblance(data, offsets, DT, VELOCITIES)
    times, vels, vals = pick_velocities(spec, VELOCITIES, DT)

    assert len(times) >= len(DEFAULT_LAYERS)
    for layer in DEFAULT_LAYERS:
        # find the pick closest in time to each true event
        k = np.abs(times - layer.t0).argmin()
        assert abs(times[k] - layer.t0) < 0.05, f"missed event at t0={layer.t0}"
        rel_err = abs(vels[k] - layer.v_rms) / layer.v_rms
        assert rel_err < 0.05, f"v_rms error {rel_err:.1%} at t0={layer.t0}"


def test_nmo_flattens_events():
    data, t, offsets = cmp_gather(dt=DT, noise=0.0)
    v_t = _true_v_at(t)
    # 30% stretch mute: traces stretched beyond that have wavelets too
    # distorted to peak-align — removing them is the mute's whole job
    corrected = nmo_correct(data, offsets, DT, v_t, stretch_mute=0.3)

    for layer in DEFAULT_LAYERS:
        i0 = int(round(layer.t0 / DT))
        window = corrected[:, i0 - 5 : i0 + 6]
        live = np.abs(window).max(axis=1) > 0
        peak_offsets_idx = np.abs(window[live]).argmax(axis=1)
        # after NMO every live trace should peak within +-2 samples of t0
        assert np.all(np.abs(peak_offsets_idx - 5) <= 2), f"event t0={layer.t0} not flat"


def test_stack_improves_snr():
    clean, t, offsets = cmp_gather(dt=DT, noise=0.0)
    noisy, _, _ = cmp_gather(dt=DT, noise=0.3, seed=7)
    v_t = _true_v_at(t)

    ref = stack(nmo_correct(clean, offsets, DT, v_t))
    noise_only = noisy - clean

    single_snr = np.std(clean[0]) / np.std(noise_only[0])
    stacked = stack(nmo_correct(noisy, offsets, DT, v_t))
    residual = stacked - ref
    stack_snr = np.std(ref) / np.std(residual)

    assert stack_snr > 2.0 * single_snr  # sqrt(30 traces) ideal ~5.5x


def test_stretch_mute_zeroes_far_offset_shallow():
    data, t, offsets = cmp_gather(dt=DT, noise=0.0)
    v_t = _true_v_at(t)
    corrected = nmo_correct(data, offsets, DT, v_t, stretch_mute=0.3)
    # shallow samples on the farthest trace must be muted
    far = corrected[-1]
    i_shallow = int(round(0.30 / DT))
    assert np.all(far[: i_shallow + 1] == 0.0)


def test_velocity_function_interpolates_and_extrapolates_flat():
    t = np.arange(0, 2.0, DT)
    v = velocity_function(t, [0.5, 1.5], [1500.0, 2500.0])
    assert v[0] == 1500.0  # flat before first pick
    assert v[-1] == 2500.0  # flat after last pick
    assert np.isclose(np.interp(1.0, t, v), 2000.0, atol=10.0)
