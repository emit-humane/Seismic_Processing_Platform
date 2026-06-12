"""Coherence validated on synthetic ground truth: continuous layers give
coherence ~1, a known fault produces a low-coherence lineament at the
correct crossline."""

import numpy as np

from seisproc.coherence import coherence, fault_likelihood
from seisproc.synthetic import synthetic_faulted_volume, synthetic_volume

DT = 0.004


def test_values_in_unit_interval_and_shape_preserved():
    data, t, _ = synthetic_faulted_volume(noise=0.05)
    coh = coherence(data, DT)
    assert coh.shape == data.shape
    assert coh.min() >= 0.0 and coh.max() <= 1.0 + 1e-9


def test_continuous_flat_layers_are_coherent():
    # flat layers away from the fault; the domed volume is unsuitable as
    # ground truth here because plain semblance coherence (no dip steering)
    # correctly penalizes dipping reflectors
    data, t, fault_xl = synthetic_faulted_volume(noise=0.0)
    coh = coherence(data, DT)
    i = int(round(0.25 / DT))
    away = coh[5:-5, : fault_xl - 5, i - 3 : i + 3]
    assert np.median(away) > 0.95


def test_fault_appears_at_correct_crossline():
    data, t, fault_xl = synthetic_faulted_volume(noise=0.02)
    coh = coherence(data, DT)

    # average coherence per crossline over reflector times, away from edges
    i0, i1 = int(0.2 / DT), int(0.9 / DT)
    profile = coh[5:-5, :, i0:i1].mean(axis=(0, 2))

    detected = int(np.argmin(profile))
    assert abs(detected - fault_xl) <= 1, f"fault detected at xl {detected}, true {fault_xl}"
    # the fault must stand out: clearly lower than background coherence
    background = np.delete(profile, slice(fault_xl - 3, fault_xl + 4))
    assert profile[detected] < 0.7 * background.mean()


def test_fault_lineament_continuous_along_inlines():
    data, t, fault_xl = synthetic_faulted_volume(noise=0.02)
    coh = coherence(data, DT)
    i_evt = int(round(0.45 / DT))
    ts = coh[:, :, i_evt]
    # on this time slice, every interior inline's minimum should sit on the fault
    mins = ts[5:-5].argmin(axis=1)
    assert np.all(np.abs(mins - fault_xl) <= 1)


def test_fault_likelihood_rescaling():
    data, t, _ = synthetic_faulted_volume()
    fl = fault_likelihood(coherence(data, DT))
    assert np.isclose(fl.max(), 1.0)
    assert fl.min() >= 0.0


def test_2d_section_support():
    data, t, fault_xl = synthetic_faulted_volume(noise=0.0)
    section = data[10]  # (n_xl, n_samples)
    coh = coherence(section, DT)
    assert coh.shape == section.shape
    i_evt = int(round(0.45 / DT))
    assert abs(int(coh[:, i_evt].argmin()) - fault_xl) <= 1
