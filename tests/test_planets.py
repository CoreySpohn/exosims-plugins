"""Planets adapter sanity: class identity, propagation, dMag pipeline."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from orbix.kepler.shortcuts.grid import get_grid_solver  # noqa: E402

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
    """j_s_dMag (eqx.filter_jit) returns finite (K, T)-shaped s and dMag."""
    p = _planets()
    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    s, dMag = p.j_s_dMag(solver, jnp.linspace(0.0, 300.0, 8))
    assert s.shape == dMag.shape == (1, 8)
    assert jnp.isfinite(s).all() and jnp.isfinite(dMag).all()
