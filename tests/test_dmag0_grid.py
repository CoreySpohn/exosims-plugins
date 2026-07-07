"""dMag0Grid interpolation contract tests (x64-specific regressions)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from exosims_plugins.dmag0 import dMag0Grid  # noqa: E402


def _synthetic_grid():
    # 5 log-spaced alphas chosen so that querying alpha_max lands the log-space
    # index exactly on n_alpha (4.000000000000001 in float64, which truncates
    # to 4) -- this is what exposes the H7 off-by-one at the OWA edge. A range
    # like geomspace(0.05, 0.5, 5) rounds the *other* way in float64 and would
    # not exercise the bug.
    alphas = jnp.geomspace(1.0, 10.0, 5)
    fZs = jnp.geomspace(1e-11, 1e-9, 4)
    kEZs = jnp.array([1.0, 3.0])
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
