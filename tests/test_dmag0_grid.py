"""Regression tests for survey grids and their count-rate inputs."""

from types import SimpleNamespace

import astropy.units as u
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from exosims_plugins import dmag0  # noqa: E402
from exosims_plugins.dmag0 import dMag0Grid  # noqa: E402
from exosims_plugins.SurveySimulation.AYOSimulation2 import AYOSimulation2  # noqa: E402


def _synthetic_grid(int_times=None):
    # 5 log-spaced alphas chosen so that querying alpha_max lands the log-space
    # index exactly on n_alpha (4.000000000000001 in float64, which truncates
    # to 4) -- this is what exposes the H7 off-by-one at the OWA edge. A range
    # like geomspace(0.05, 0.5, 5) rounds the *other* way in float64 and would
    # not exercise the bug.
    alphas = jnp.geomspace(1.0, 10.0, 5)
    fZs = jnp.geomspace(1e-11, 1e-9, 4)
    kEZs = jnp.array([1.0, 3.0])
    if int_times is None:
        int_times = jnp.array([1.0, 10.0, 100.0])
    # grid shape (n_fZ, n_kEZ, n_alpha, n_int_times); dMag0 rises along the
    # alpha axis (axis 2), constant across fZ/kEZ/int_times.
    dmag_by_alpha = jnp.linspace(20.0, 24.0, 5)
    grid = jnp.broadcast_to(dmag_by_alpha[None, None, :, None], (4, 2, 5, 3))
    return dMag0Grid(fZs=fZs, kEZs=kEZs, alphas=alphas, int_times=int_times, grid=grid)


def test_mask_runs_under_x64():
    """alpha_dMag_mask must not raise under x64 (H6 int32/int64 dtype mix)."""
    g = _synthetic_grid()
    # The regression signal is that this call does not raise a dtype TypeError;
    # assert on the real boolean mask it returns (one flag per int_time) rather
    # than a tautological finite-check on a boolean array.
    out = g.alpha_dMag_mask(
        jnp.array([[3.0]]), jnp.array([[21.0]]), jnp.array([5e-10]), 1.0
    )
    assert out.dtype == jnp.bool_
    assert out.shape == (1, 1, g.int_times.shape[0])


def test_owa_edge_uses_last_cell():
    """A planet at alpha_max must use the true top-cell dMag0, not the prior cell."""
    g = _synthetic_grid()
    alpha_max = g.alphas[-1]
    # True limiting dMag0 at alpha_max is 24.0 (top grid cell); pre-fix the
    # effective limit was 23.0 (second-to-last cell), so this planet would be
    # missed even though it is well within the true limit.
    out = g.alpha_dMag_mask(
        jnp.array([[alpha_max]]), jnp.array([[23.5]]), jnp.array([5e-10]), 1.0
    )
    assert bool(out[..., -1].any())


def test_exposure_index_uses_days_and_clips_to_grid():
    """Use the days axis and clip out-of-range requests to its endpoints."""
    grid = _synthetic_grid(jnp.array([0.1, 1.0, 10.0]))
    assert grid.exposure_index(1.0) == 1
    assert grid.exposure_index(0.5) == 1
    assert grid.exposure_index(0.01) == 0
    assert grid.exposure_index(100.0) == 2


def test_epoch_jez_scales_from_one_au():
    """Scale the 1 AU reference by 1/r**2 with no luminosity factor.

    EXOSIMS 3.6.6 removed the 1/L from JEZ0; a factor of L here would count
    the luminosity a second time in the other direction.
    """
    unit = u.photon / (u.s * u.m**2 * u.arcsec**2)
    mode = {"hex": "test"}
    jez0, nzodi, fbeta, luminosity, r_au = 8.0, 3.0, 0.5, 4.0, 2.0
    tl = SimpleNamespace(
        JEZ0={"test": [1.0, jez0] * unit},
        system_fbeta=np.array([1.0, fbeta]),
        L=np.array([1.0, luminosity]),
    )
    actual = dmag0.epoch_jez(tl, 1, mode, r_au, nzodi).to_value(unit)
    assert actual == jez0 * nzodi * fbeta / r_au**2


def test_grid_cache_key_includes_full_axes(monkeypatch):
    """Distinct grid axes must produce distinct cache paths."""
    tl = SimpleNamespace(
        OpticalSystem=SimpleNamespace(),
        StarCatalog=SimpleNamespace(),
        nStars=1,
        Name=["star"],
    )
    zl = SimpleNamespace(calc_fbeta=lambda _: 1.0, fZMap={"test": None})
    survey = SimpleNamespace(TargetList=tl, ZodiacalLight=zl, Observatory=object())
    mode = dict(
        hex="test", systName="test", SNR=7.0, IWA=0.1 * u.arcsec, OWA=1.0 * u.arcsec
    )
    monkeypatch.setattr(dmag0, "gen_dMag0_hex", lambda *_: "test")
    monkeypatch.setattr(dmag0, "safe_cache_load", lambda path: path.name)

    log_times = np.geomspace(0.1, 10.0, 3) * u.d
    linear_times = np.linspace(0.1, 10.0, 3) * u.d
    nzodi = np.array([1.0, 2.0, 3.0])
    cache = dmag0.dMag0_grid(survey, mode, log_times, nzodi, n_kEZs=3)[0]
    assert cache != dmag0.dMag0_grid(survey, mode, log_times, nzodi, n_kEZs=4)[0]
    assert cache != dmag0.dMag0_grid(survey, mode, linear_times, nzodi, n_kEZs=3)[0]
    assert cache != dmag0.dMag0_grid(survey, mode, log_times, [1.0, 2.5, 3.0])[0]
    zl.calc_fbeta = lambda _: 0.5
    assert cache != dmag0.dMag0_grid(survey, mode, log_times, nzodi, n_kEZs=3)[0]


def test_csv_noise_floor_is_point_source_contrast(tmp_path):
    """Convert the CSV point-source contrast to a stellar fraction with throughput.

    The column is raw contrast (stellar over planet intensity) divided by the
    post-processing factor, so the stellar fraction is that value times the
    planet core throughput, with no aperture-area factor.
    """
    contrast = 1e-9
    csv_path = tmp_path / "noise_floor.csv"
    csv_path.write_text(
        f"Sep (l/D),Noise floor (point source 1-sigma)\n1,{contrast}\n3,{contrast}\n"
    )
    curves = {"throughput": 0.5, "core_area_lod2": 4.0, "leakage_density_lod2": 1e-8}
    record = SimpleNamespace(evaluate=lambda name, sep: np.full_like(sep, curves[name]))
    coro = SimpleNamespace(
        curves=lambda _: record, occulter_transmission=lambda sep: np.ones_like(sep)
    )
    survey = SimpleNamespace(
        _load_yippy_coronagraph=lambda: coro,
        _init_multiaperture_curves=lambda _: None,
        _aperture_interpolators={1.0: record},
        _etc_debug_done=True,
        _coro_OWA_lod=10.0,
        _observatory_initialized=True,
        _base_scene=SimpleNamespace(),
        _base_observation=SimpleNamespace(SNR=7 * u.one),
        _pyedith_observatory=SimpleNamespace(
            telescope=SimpleNamespace(
                Area=1 * u.m**2, toverhead_multi=1.0, toverhead_fixed=0 * u.s
            ),
            coronagraph=SimpleNamespace(nchannels=1, pixscale=1 * u.one),
            detector=SimpleNamespace(QE=[1] * u.one, dQE=[1] * u.one),
            total_throughput=[1] * u.one,
        ),
        TargetList=SimpleNamespace(dist=[10.0] * u.pc),
        det_params={"wavelength": 0.5, "DC": 0.25, "RN": 0.0, "CIC": 0.0},
        aperture_rule=SimpleNamespace(radius_lod=1.0),
        vprint=lambda _: None,
    )
    AYOSimulation2._load_noise_floor_csv(survey, csv_path)
    rates = AYOSimulation2.calc_count_rates(survey, 2.0, 19.0, 10.0)
    assert rates["noise_floor_fraction"].item() == contrast * 0.5
    # Detector settings are read by their mapped names, not the AYO names.
    assert rates["CRbd"].item() == pytest.approx(0.25 * rates["npix"].item())
    assert rates["CRnf"].item() == pytest.approx(
        rates["SNR"] * rates["flux_factor"] * contrast * 0.5
    )
    # The old extra factor of four put the floor above the planet signal.
    times = AYOSimulation2._calc_inttime_vectorized(
        survey, 0, np.array([[19.0]]), np.array([[rates["sep_arcsec"]]])
    )
    assert np.isfinite(times.item())
