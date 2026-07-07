"""Base planet model.

Composes :class:`orbix.orbit.KeplerianOrbit` for geometric
propagation; RV-specific derived quantities (semi-amplitude, minimum
mass, ``secosw`` / ``sesinw``) live on ``Planets`` for the orbit-only
fitting pipeline.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import orbix.equations.orbit as oe
from hwoutils.constants import (
    AU2m,
    G,
    Mearth2kg,
    Rearth2AU,
    d2s,
    pc2AU,
    rad2arcsec,
    two_pi,
)
from jaxtyping import Array
from orbix.equations.phase import lambert_phase_exact
from orbix.orbit import AbstractOrbit, KeplerianOrbit


class Planets(eqx.Module):
    """JAX-friendly planet model that composes an orbit.

    Geometry lives on ``self.orbit`` (an ``AbstractOrbit``, concretely
    ``KeplerianOrbit`` today). RV and photometric derived quantities
    stay on ``Planets`` for the orbit-only fitting pipeline.

    Constructor arguments (unit-suffix naming, phase 0c of the
    skyscapes migration):
        Ms_kg: stellar mass in kg
        dist_pc: distance to star in pc
        a_AU, e, W_rad, i_rad, w_rad, M0_rad, t0_d: orbital elements
        Mp_Mearth: planet mass in Earth masses (stored as Mp_kg)
        Rp_Rearth: planet radius in Earth radii (stored as Rp_AU)
        Ag: geometric albedo (dimensionless)

    Instances are construct-only: build a new Planets rather than
    tree_at-updating elements, because derived fields (Mp_kg, Rp_AU,
    Mp_min_kg) are cached at construction.
    """

    # Composed geometry
    orbit: AbstractOrbit

    # Retained public fields (unit-suffixed)
    Ms_kg: Array
    dist_pc: Array
    a_AU: Array
    e: Array
    W_rad: Array
    i_rad: Array
    w_rad: Array
    M0_rad: Array
    t0_d: Array
    Mp_kg: Array
    Rp_AU: Array
    Ag: Array

    # Derived
    mu: Array
    n_radpd: Array
    T_d: Array
    tp_d: Array
    w_p_rad: Array
    w_s_rad: Array
    secosw: Array
    sesinw: Array
    K_mps: Array
    Mp_min_kg: Array
    a_arcsec: Array
    _rad2arcsec_dist: Array

    def __init__(
        self,
        Ms_kg,
        dist_pc,
        a_AU,
        e,
        W_rad,
        i_rad,
        w_rad,
        M0_rad,
        t0_d,
        Mp_Mearth,
        Rp_Rearth,
        Ag,
    ):
        """Build a Planets instance with a composed KeplerianOrbit.

        See the class docstring for parameter units.
        """
        self.Ms_kg = Ms_kg
        self.dist_pc = dist_pc
        self.a_AU = a_AU
        self.e = e
        self.W_rad = W_rad
        self.i_rad = i_rad
        self.w_rad = w_rad
        self.M0_rad = M0_rad
        self.t0_d = t0_d
        self.Mp_kg = Mp_Mearth * Mearth2kg
        self.Rp_AU = Rp_Rearth * Rearth2AU
        self.Ag = Ag

        # Geometry via KeplerianOrbit
        self.orbit = KeplerianOrbit(
            a_AU=a_AU,
            e=e,
            W_rad=W_rad,
            i_rad=i_rad,
            w_rad=w_rad,
            M0_rad=M0_rad,
            t0_d=t0_d,
        )

        # Derived quantities that need stellar context
        self.mu = G * Ms_kg
        self.n_radpd = oe.mean_motion(a_AU, self.mu)
        self.T_d = oe.period_n(self.n_radpd)
        T_e = self.T_d * self.M0_rad / two_pi
        self.tp_d = self.t0_d - T_e
        self.w_p_rad = self.w_rad
        self.w_s_rad = (self.w_rad + jnp.pi) % two_pi

        sqrt_1me2 = jnp.sqrt(1 - e**2)
        se = jnp.sqrt(e)
        self.secosw = se * jnp.cos(w_rad)
        self.sesinw = se * jnp.sin(w_rad)
        self.Mp_min_kg = self.Mp_kg * jnp.sin(i_rad)
        # semi_amplitude_reduced returns K in AU/d (G is in AU^3/(kg*d^2)).
        # Convert to m/s for the canonical RV-literature unit.
        self.K_mps = (
            oe.semi_amplitude_reduced(
                self.T_d,
                Ms_kg,
                self.Mp_min_kg,
                sqrt_1me2,
            )
            * AU2m
            / d2s
        )

        # On-sky projection scale
        _dist_AU = dist_pc * pc2AU
        self._rad2arcsec_dist = rad2arcsec / _dist_AU
        self.a_arcsec = self.a_AU * self._rad2arcsec_dist

    # --- Propagation (delegates to self.orbit) ---

    def prop_AU(self, trig_solver, t):
        """Position vectors in AU, shape ``(K, 3, T)``."""
        r_AU, _, _ = self.orbit.propagate(trig_solver, t, Ms_kg=self.Ms_kg)
        return r_AU

    def prop_as(self, trig_solver, t):
        """Position vectors in arcsec, shape ``(K, 3, T)``."""
        r_AU = self.prop_AU(trig_solver, t)
        return r_AU * self._rad2arcsec_dist[:, None, None]

    def prop_ra_dec(self, trig_solver, t):
        """(RA_arcsec, Dec_arcsec) each ``(K, T)``."""
        r_as = self.prop_as(trig_solver, t)
        return r_as[:, 0], r_as[:, 1]

    def s_dMag(self, trig_solver, t):
        """Return (projected separation AU, dMag) each ``(K, T)``."""
        r_AU, phase_angle_rad, dist_AU = self.orbit.propagate(
            trig_solver,
            t,
            Ms_kg=self.Ms_kg,
        )
        s = jnp.sqrt(r_AU[:, 0] ** 2 + r_AU[:, 1] ** 2)

        cosbeta = jnp.cos(phase_angle_rad)
        sinbeta = jnp.sin(phase_angle_rad)
        phase = lambert_phase_exact(cosbeta, sinbeta)

        log_arg = self.Ag[:, None] * (self.Rp_AU[:, None] / dist_AU) ** 2 * phase
        eps = jnp.finfo(log_arg.dtype).tiny
        dMag = -2.5 * jnp.log10(log_arg + eps)
        return s, dMag

    def alpha_dMag(self, trig_solver, t):
        """Return (projected separation arcsec, dMag) each ``(K, T)``."""
        s, dMag = self.s_dMag(trig_solver, t)
        alpha = s * self._rad2arcsec_dist[:, None]
        return alpha, dMag

    # --- jitted variants (retained API) ---

    @eqx.filter_jit
    def j_prop_AU(self, trig_solver, t):
        """JITted ``prop_AU``."""
        return self.prop_AU(trig_solver, t)

    @eqx.filter_jit
    def j_prop_as(self, trig_solver, t):
        """JITted ``prop_as``."""
        return self.prop_as(trig_solver, t)

    @eqx.filter_jit
    def j_prop_ra_dec(self, trig_solver, t):
        """JITted ``prop_ra_dec``."""
        return self.prop_ra_dec(trig_solver, t)

    @eqx.filter_jit
    def j_s_dMag(self, trig_solver, t):
        """JITted ``s_dMag``."""
        return self.s_dMag(trig_solver, t)

    @eqx.filter_jit
    def j_alpha_dMag(self, trig_solver, t):
        """JITted ``alpha_dMag``."""
        return self.alpha_dMag(trig_solver, t)
