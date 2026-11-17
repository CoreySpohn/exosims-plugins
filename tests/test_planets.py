"""Planets adapter sanity: class identity, propagation, dMag pipeline."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from EXOSIMS.util.atomic_io import robust_pickle_load  # noqa: E402
from hwoutils.constants import G, Mearth2kg, Rearth2AU  # noqa: E402
from orbix.kepler import Grid, Markley  # noqa: E402

from exosims_plugins.Completeness.OrbixCompleteness import (  # noqa: E402
    OrbixCompleteness,
)
from exosims_plugins.planets import Planets  # noqa: E402


def _planets():
    one = jnp.ones(1)
    return Planets(
        Ms_kg=2.0e30 * one,
        dist_pc=10.0 * one,
        a_AU=1.0 * one,
        e=0.1 * one,
        W_rad=0.3 * one,
        i_rad=0.5 * one,
        w_rad=0.7 * one,
        M0_rad=0.2 * one,
        t0_d=0.0 * one,
        Mp_Mearth=1.0 * one,
        Rp_Rearth=1.0 * one,
        Ag=0.3 * one,
    )


def test_planets_is_a_class():
    """Planets must remain a real class, not a PjitFunction (H3 regression)."""
    p = _planets()
    assert isinstance(p, Planets)


def test_s_dMag_pipeline_runs():
    """Check classical geometry and Lambert flux from independent primitives.

    Markley tolerance is 128 float64 eps for rotations and logarithms. Grid's
    conservative bilinear tolerance is eight times the sum of squared cell
    spacings, for these smooth e=0.1 orbits away from new phase.
    """
    p = _planets()
    solver = Grid()
    times = np.linspace(0.0, 300.0, 8)
    mean = 0.2 + np.sqrt(G * 2.0e30) * times
    eccentric = mean.copy()
    for _ in range(10):
        eccentric -= (eccentric - 0.1 * np.sin(eccentric) - mean) / (
            1 - 0.1 * np.cos(eccentric)
        )
    x = np.cos(eccentric) - 0.1
    y = np.sqrt(1 - 0.1**2) * np.sin(eccentric)
    peri_x = np.cos(0.7) * x - np.sin(0.7) * y
    peri_y = np.sin(0.7) * x + np.cos(0.7) * y
    east = np.sin(0.3) * peri_x + np.cos(0.3) * np.cos(0.5) * peri_y
    north = np.cos(0.3) * peri_x - np.sin(0.3) * np.cos(0.5) * peri_y
    away = np.sin(0.5) * peri_y
    expected_position = np.array([[east, north, away]])
    expected_s = np.hypot(east, north)[None, :]
    radius = np.sqrt(x**2 + y**2)
    beta = np.arctan2(expected_s[0], away)
    phase = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi
    expected_dmag = -2.5 * np.log10(0.3 * (Rearth2AU / radius) ** 2 * phase)
    floating_tol = 128 * np.finfo(float).eps
    np.testing.assert_allclose(
        p.j_prop_AU(Markley(), times),
        expected_position,
        rtol=floating_tol,
        atol=floating_tol,
    )
    s, dMag = p.j_s_dMag(Markley(), times)
    np.testing.assert_allclose(s, expected_s, rtol=floating_tol)
    np.testing.assert_allclose(dMag, expected_dmag[None, :], rtol=floating_tol)
    s, dMag = p.j_s_dMag(solver, times)
    assert s.shape == dMag.shape == (1, 8)
    assert jnp.isfinite(s).all() and jnp.isfinite(dMag).all()
    grid_tol = 8 * ((2 * np.pi / solver.n_M) ** 2 + (1 / (solver.n_e - 1)) ** 2)
    np.testing.assert_allclose(s, expected_s, atol=grid_tol, rtol=0)
    np.testing.assert_allclose(dMag, expected_dmag[None, :], atol=grid_tol, rtol=0)


def test_planets_cache_round_trip(tmp_path):
    """Cache units are exact; allow 2 eps for the Earth-unit round trip.

    Expected mass and radius are one Earth mass/radius times their owner's
    conversion constants, independent of the cache and constructor outputs.
    """
    original = object.__new__(OrbixCompleteness)
    original.vprint = lambda *args: None
    original._planets = _planets()
    original.s = np.array([[0.5, 1.0]])
    original.dMag = np.array([[20.0, 25.0]])
    original.comp_times = jnp.array([0.0, 10.0])
    planets_path, orbits_path = tmp_path / "planets", tmp_path / "orbits"
    original.save_orbix_cache(planets_path, orbits_path)
    data = robust_pickle_load(str(planets_path))
    assert "Mp_Mearth" not in data and "Rp_Rearth" not in data
    np.testing.assert_array_equal(data["Mp_kg"], [Mearth2kg])
    np.testing.assert_array_equal(data["Rp_AU"], [Rearth2AU])
    restored = object.__new__(OrbixCompleteness)
    restored.vprint = lambda *args: None
    assert restored.load_orbix_cache(planets_path, orbits_path)
    tol = 2 * np.finfo(float).eps
    np.testing.assert_allclose(restored._planets.Mp_kg, [Mearth2kg], rtol=tol)
    np.testing.assert_allclose(restored._planets.Rp_AU, [Rearth2AU], rtol=tol)
    np.testing.assert_array_equal(restored.s, [[0.5, 1.0]])
    np.testing.assert_array_equal(restored.dMag, [[20.0, 25.0]])
    np.testing.assert_array_equal(restored.comp_times, [0.0, 10.0])
