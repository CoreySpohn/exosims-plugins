"""Simulated universe that uses orbix to calculate the detection probability."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from EXOSIMS.SimulatedUniverse.DulzPlavchanUniverseEarthsOnly import (
    DulzPlavchanUniverseEarthsOnly,
)
from orbix.system import Planets


class OrbixUniverse(DulzPlavchanUniverseEarthsOnly):
    """A simulated universe that uses orbix to calculate the detection probability.

    This class is a subclass of DulzPlavchanUniverseEarthsOnly for now,
    probably shouldn't be permanently.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the OrbixUniverse."""
        super().__init__(*args, **kwargs)
        # Create orbix Planets for each generated planet
        # Index by pInd, create when promoting to characterization
        self.orbix_planets = {}

    def create_orbix_planets(self, pInd, TK, SS, err=0.025, norb=500):
        """Create orbix Planets Equinox object for this planet.

        The Planets object can be used with a dMag0Grid to calculate the
        probability of detection at a given time.

        Args:
            pInd (int):
                Index of the planet
            TK (TimeKeeping):
                TimeKeeping object
            SS:
                SurveySimulation object
            err (float):
                Fractional 1-sigma error on the uncertain parameters (default: 0.025)
            norb (int):
                Number of planets to create (default: 500)
        """
        # Validate inputs
        if err <= 0:
            raise ValueError(f"Error parameter must be positive, got {err}")
        if norb <= 0:
            raise ValueError(f"Number of orbits must be positive, got {norb}")

        # Semi-major axis sampling (log-normal to ensure positive values)
        a_mean = self.a[pInd].to_value(u.AU)
        log_a_mean = np.log(a_mean)
        log_a_std = err  # Fractional error in log space
        _a = jnp.array(np.random.lognormal(log_a_mean, log_a_std, (norb,)))

        # Albedo sampling (log-normal to ensure positive values)
        p_mean = self.p[pInd]
        if p_mean <= 0:
            raise ValueError(f"Planet albedo must be positive, got {p_mean}")
        log_p_mean = np.log(p_mean)
        log_p_std = err  # Fractional error in log space
        _p = jnp.array(np.random.lognormal(log_p_mean, log_p_std, (norb,)))

        # Validate results
        if jnp.any(_a <= 0):
            raise RuntimeError("Generated negative or zero semi-major axis values")
        if jnp.any(_p <= 0):
            raise RuntimeError("Generated negative or zero albedo values")
        if jnp.any(_p > 1):
            # Clip albedos that are > 1 (physically unrealistic)
            _p = jnp.clip(_p, 0, 1)

        # All other parameters we just stack as constant arrays
        _e = jnp.full(norb, self.e[pInd])
        _W = jnp.full(norb, self.O[pInd].to_value(u.rad))
        _i = jnp.full(norb, self.I[pInd].to_value(u.rad))
        _w = jnp.full(norb, self.w[pInd].to_value(u.rad))
        _Mp = jnp.full(norb, self.Mp[pInd].to_value(u.M_earth))
        _Rp = jnp.full(norb, self.Rp[pInd].to_value(u.R_earth))

        # Get the host star parameters as single values
        sInd = self.plan2star[pInd]
        _Ms = jnp.array([self.TargetList.MsTrue[sInd].to_value(u.kg)])
        _dist = jnp.array([self.TargetList.dist[sInd].to_value(u.pc)])

        # Set t0 to current time and calculate the corresponding M0 based on
        # the values of t and M0 from instantiation of SimulatedUniverse
        t = TK.currentTimeNorm.to_value(u.d)
        # Use this to propagate the generated orbits
        # n = jnp.sqrt(self.mu_AU3_div_day2[pInd] / _a**3)
        n = jnp.sqrt(self.mu_AU3_div_day2[pInd] / self.a[pInd].to_value(u.AU) ** 3)
        # n is rad/day, t is days, M0 is rad, so final value is rad
        M0 = (n * t + self.M0[pInd].to_value(u.rad)) % (2 * jnp.pi)
        _t0 = jnp.full(norb, t)
        _M0 = jnp.full(norb, M0)
        # _t0 = jnp.full(norb, 0)
        # _M0 = jnp.full(norb, self.M0[pInd].to_value(u.rad))

        _planets = Planets(_Ms, _dist, _a, _e, _W, _i, _w, _M0, _t0, _Mp, _Rp, _p)
        self.orbix_planets[pInd] = _planets
