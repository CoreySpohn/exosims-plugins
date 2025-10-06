"""Completeness model that uses orbix."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.time import Time
from EXOSIMS.Completeness.BrownCompleteness import BrownCompleteness
from EXOSIMS.util.atomic_io import atomic_pickle_dump, robust_pickle_load
from orbix.constants import Msun2kg, rad2arcsec
from orbix.integrations.exosims.dMag0 import gen_dMag0_hex
from orbix.system import Planets
from tqdm import tqdm


@dataclass
class StarEnsemble:
    """Class to track dynamic completeness for each star."""

    sInd: int
    # Boolean mask of orbits that haven't been ruled out
    valid_orbits: np.ndarray
    # total number of orbits drawn in OrbixCompleteness
    n_orbits: int
    # number of times we've failed to detect a planet around this star
    n_failures: int = 0
    # mjd of scheduled follow‑up
    next_available_time: float | None = None
    # fraction of remaining orbits that are valid
    f_valid: float = 1.0
    # dyn_comp_times
    max_dyn_comp_int_time: float = 0.0
    # best integration time at each time
    best_int_times: np.ndarray[float] = field(
        default_factory=lambda: np.zeros(shape=(1,), dtype=np.float64)
    )
    # best completeness/int_time at each time
    best_comp_div_intTime: np.ndarray[float] = field(
        default_factory=lambda: np.zeros(shape=(1,), dtype=np.float64)
    )
    # MJD times for the above metrics
    times: np.ndarray[float] = field(
        default_factory=lambda: np.zeros(shape=(1,), dtype=np.float64)
    )
    # Normalized time representing when to wait for follow-up
    next_available_time: float = 0


class OrbixCompleteness(BrownCompleteness):
    """Completeness model that uses orbix to calculate the detection probability."""

    def _get_planets_cache_subdir(self, TK):
        """Generate a unique subdirectory for planets/orbits caches."""
        base_name = (
            self.PlanetPopulation.__class__.__name__
            + self.PlanetPhysicalModel.__class__.__name__
            + self.__class__.__name__
            + str(self.Nplanets)
            + self.PlanetPhysicalModel.whichPlanetPhaseFunction
        )
        mission_start = TK.missionStart.mjd
        mission_life_d = TK.missionLife_d
        atts = list(self.PlanetPopulation.__dict__)
        extstr = ""
        for att in sorted(atts, key=str.lower):
            if (
                not (callable(getattr(self.PlanetPopulation, att)))
                and (att != "PlanetPhysicalModel")
                and (att != "cachedir")
                and (att != "_outspec")
            ):
                extstr += "%s: " % att + str(getattr(self.PlanetPopulation, att))
        extstr += "missionStart: " + str(mission_start)
        extstr += "missionLife_d: " + str(mission_life_d)
        ext = hashlib.md5(extstr.encode("utf-8")).hexdigest()
        subdir = Path(self.cachedir) / f"orbix_planets_{ext}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir, base_name, ext

    def generate_orbix_cache_path(self, TK):
        """Generate unique filenames for cached orbix products."""
        cache_subdir, base_name, ext = self._get_planets_cache_subdir(TK)
        self.planets_filename = base_name + "_planets_" + ext
        self.orbits_filename = base_name + "_orbits_" + ext
        self.planets_path = (
            cache_subdir / f"{self.planets_filename.replace(' ', '')}.planets"
        )
        self.orbits_path = (
            cache_subdir / f"{self.orbits_filename.replace(' ', '')}.orbits"
        )
        return self.planets_path, self.orbits_path

    def generate_orbix_cache_manifest_path(self, TK, SS):
        """Path for a simple planets/orbits cache manifest."""
        cache_subdir, base_name, ext = self._get_planets_cache_subdir(TK)
        fname = base_name + "_orbix_cache_manifest_" + ext
        return cache_subdir / f"{fname.replace(' ', '')}.txt"

    def save_orbix_cache(self, planets_path, orbits_path):
        """Save the planets object and orbital data to cache files.

        Args:
            planets_path (Path):
                Path to save the Planets object
            orbits_path (Path):
                Path to save the s and dMag orbital data
        """
        # Save Planets object parameters
        planets_data = {
            "Ms": np.array(self._planets.Ms),
            "dist": np.array(self._planets.dist),
            "a": np.array(self._planets.a),
            "e": np.array(self._planets.e),
            "W": np.array(self._planets.W),
            "i": np.array(self._planets.i),
            "w": np.array(self._planets.w),
            "M0": np.array(self._planets.M0),
            "t0": np.array(self._planets.t0),
            "Mp": np.array(self._planets.Mp),
            "Rp": np.array(self._planets.Rp),
            "p": np.array(self._planets.p),
        }

        atomic_pickle_dump(planets_data, str(planets_path))

        # Save orbital data using standard pickle
        orbits_data = {
            "s": self.s,
            "dMag": self.dMag,
            "comp_times": np.array(self.comp_times),
        }

        atomic_pickle_dump(orbits_data, str(orbits_path))

        self.vprint("Orbix planets object stored in %r" % planets_path)
        self.vprint("Orbix orbital data stored in %r" % orbits_path)

    def load_orbix_cache(self, planets_path, orbits_path):
        """Load the planets object and orbital data from cache files.

        Args:
            planets_path (Path):
                Path to load the Planets object from
            orbits_path (Path):
                Path to load the s and dMag orbital data from

        Returns:
            bool:
                True if loading was successful, False otherwise
        """
        try:
            # Load Planets object data
            planets_data = robust_pickle_load(str(planets_path))

            # Reconstruct Planets object
            self._planets = Planets(
                jnp.array(planets_data["Ms"]),
                jnp.array(planets_data["dist"]),
                jnp.array(planets_data["a"]),
                jnp.array(planets_data["e"]),
                jnp.array(planets_data["W"]),
                jnp.array(planets_data["i"]),
                jnp.array(planets_data["w"]),
                jnp.array(planets_data["M0"]),
                jnp.array(planets_data["t0"]),
                jnp.array(planets_data["Mp"]),
                jnp.array(planets_data["Rp"]),
                jnp.array(planets_data["p"]),
            )

            # Load orbital data
            orbits_data = robust_pickle_load(str(orbits_path))

            self.s = orbits_data["s"]
            self.dMag = orbits_data["dMag"]
            self.comp_times = jnp.array(orbits_data["comp_times"])

            self.vprint('Loading cached orbix planets from "%s".' % planets_path)
            self.vprint('Loading cached orbix orbits from "%s".' % orbits_path)

            return True

        except Exception as e:
            self.vprint(f"Failed to load orbix cache: {e}")
            return False

    def _generate_orbix_data(self, trig_solver, TK, PPop):
        """Generate the planets object and propagate orbits to get s and dMag values.

        Args:
            trig_solver:
                Trigonometric solver for orbital calculations
            TK (TimeKeeping):
                TimeKeeping object
            PPop (PlanetPopulation):
                PlanetPopulation object
        """
        # sample quantities
        a, e, p, Rp = PPop.gen_plan_params(self.Nplanets)
        i, W, w = PPop.gen_angles(self.Nplanets)

        _a, _e, _p, _Rp = jnp.array(a), jnp.array(e), jnp.array(p), jnp.array(Rp)
        _i, _W, _w = jnp.array(i), jnp.array(W), jnp.array(w)

        # Mean anomaly should be uniformly distributed
        _M0 = jnp.array(np.random.uniform(high=2.0 * np.pi, size=self.Nplanets))
        _t0 = jnp.zeros(self.Nplanets)

        # Mp doesn't matter for completeness
        _Mp = jnp.ones(self.Nplanets)
        # Distance and mass will be handled later
        _dist = jnp.ones(self.Nplanets)
        _Ms = jnp.ones(self.Nplanets) * Msun2kg
        self._planets = Planets(_Ms, _dist, _a, _e, _W, _i, _w, _M0, _t0, _Mp, _Rp, _p)

        # Propagate the orbits
        end_time = TK.missionLife_d
        # self.comp_times = jnp.arange(0, end_time, 1)
        self.comp_times = jnp.linspace(0, end_time, 1000)

        # Shapes: (Nplanets, Ntimes)
        self.s, self.dMag = self._planets.j_s_dMag(trig_solver, self.comp_times)
        self.s = self.s.astype(np.float16)
        self.dMag = self.dMag.astype(np.float16)

        # Convert to numpy arrays for further processing
        self.s = np.array(self.s)
        self.dMag = np.array(self.dMag)

    def _get_star_ensemble_subdir(self, TK, SS):
        """Generate a unique subdirectory for star ensemble caches."""
        base_name = (
            self.PlanetPopulation.__class__.__name__
            + self.PlanetPhysicalModel.__class__.__name__
            + self.__class__.__name__
            + str(self.Nplanets)
            + self.PlanetPhysicalModel.whichPlanetPhaseFunction
        )
        mission_start = TK.missionStart.mjd
        mission_life_d = TK.missionLife_d
        atts = list(self.PlanetPopulation.__dict__)
        extstr = ""
        for att in sorted(atts, key=str.lower):
            if (
                not (callable(getattr(self.PlanetPopulation, att)))
                and (att != "PlanetPhysicalModel")
                and (att != "cachedir")
                and (att != "_outspec")
            ):
                extstr += "%s: " % att + str(getattr(self.PlanetPopulation, att))
        extstr += "missionStart: " + str(mission_start)
        extstr += "missionLife_d: " + str(mission_life_d)
        char_mode = SS.base_char_mode
        extstr += "char_mode: " + str(char_mode["hex"])
        extstr += "OpticalSystem: " + SS.OpticalSystem.__class__.__name__
        extstr += "ZodiacalLight: " + SS.ZodiacalLight.__class__.__name__
        extstr += "dMag0_hex: " + gen_dMag0_hex(char_mode, SS)
        ext = hashlib.md5(extstr.encode("utf-8")).hexdigest()
        subdir = Path(self.cachedir) / f"orbix_ensembles_{ext}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir, base_name, extstr

    def generate_star_ensemble_cache_path(self, TK, SS, sInd):
        """Generate unique filename for cached star ensemble data for a specific star.

        Args:
            TK (TimeKeeping):
                TimeKeeping object for mission timing parameters
            SS (SurveySimulation):
                SurveySimulation object for mode and target list info
            sInd (int):
                Star index

        Returns:
            Path:
                Path object for the star ensemble cache file
        """
        ensemble_subdir, base_name, extstr_base = self._get_star_ensemble_subdir(TK, SS)
        extstr = extstr_base + "sInd: " + str(sInd)
        ext = hashlib.md5(extstr.encode("utf-8")).hexdigest()
        ensemble_filename = base_name + f"_ensemble_s{sInd}_" + ext
        return ensemble_subdir / f"{ensemble_filename.replace(' ', '')}.ensemble"

    def generate_star_ensemble_manifest_path(self, TK, SS):
        """Generate a manifest filename that lists all star-ensemble files used."""
        ensemble_subdir, base_name, _ = self._get_star_ensemble_subdir(TK, SS)
        # We need the hash from the *original* implementation to stay consistent
        # with the subdirectory name itself.
        manifest_ext = ensemble_subdir.name.split("_")[-1]
        manifest_filename = base_name + "_ensemble_manifest_" + manifest_ext
        return ensemble_subdir / f"{manifest_filename.replace(' ', '')}.txt"

    def load_star_ensemble_cache(self, ensemble_path):
        """Load a single star ensemble object from cache file.

        Args:
            ensemble_path (Path):
                Path to load the star ensemble data from

        Returns:
            StarEnsemble or None:
                StarEnsemble object if loading was successful, None otherwise
        """
        try:
            # Load star ensemble directly
            star_ensemble = robust_pickle_load(str(ensemble_path))
            return star_ensemble

        except Exception as e:
            self.vprint(f"Failed to load star ensemble cache: {e}")
            return None

    def _generate_single_star_ensemble(self, SS, TK, sInd, fZ):
        """Generate a single star ensemble object with mask calculation.

        Args:
            SS (SurveySimulation):
                SurveySimulation object
            TK (TimeKeeping):
                TimeKeeping object
            sInd (int):
                Star index
            fZ (np.ndarray):
                Zodiacal light brightness.

        Returns:
            StarEnsemble:
                Generated star ensemble object
        """
        char_mode = SS.base_char_mode
        fZ = fZ[sInd]
        kEZ = SS.exact_kEZs[sInd]

        # Create the initial mask for the star ensemble to represent all the
        # planets that we can feasibly characterize
        dMag0 = SS.dMag0s[char_mode["hex"]][sInd]
        alpha = self.s * self.alpha_factors[sInd]
        # Any detectable planets at maximum integration time
        # _mask = dMag0.alpha_dMag_mask(alpha, self.dMag, fZ)
        mask = np.any(
            dMag0.alpha_dMag_mask(alpha, self.dMag, fZ, kEZ)[:, :, -1], axis=1
        )
        f_valid = mask.sum() / self.Nplanets
        if f_valid > 0.0:
            det_dMag0 = SS.dMag0s[SS.base_det_mode["hex"]][sInd]
            dyn_comp_div_intTime = np.array(
                det_dMag0.dyn_comp_vec(SS.solver, alpha, self.dMag, fZ, kEZ, mask)
            )
            max_dyn_comp_inds = np.argmax(dyn_comp_div_intTime, axis=1)
            best_int_times = np.array(dMag0.int_times)[max_dyn_comp_inds]
            best_comp_div_intTime = np.array(dyn_comp_div_intTime)[
                np.arange(dyn_comp_div_intTime.shape[0]), max_dyn_comp_inds
            ]
        else:
            dyn_comp_div_intTime = np.zeros(self.Nplanets)
            best_int_times = np.zeros(self.Nplanets)
            best_comp_div_intTime = np.zeros(self.Nplanets)
        dtimes = np.array(self.comp_times)

        return StarEnsemble(
            sInd=sInd,
            valid_orbits=mask,
            n_orbits=self.Nplanets,
            f_valid=f_valid,
            best_int_times=best_int_times,
            best_comp_div_intTime=best_comp_div_intTime,
            times=dtimes,
        )

    def _generate_star_ensembles(self, SS, TK):
        """Generate star ensemble objects with mask calculations.

        Args:
            SS (SurveySimulation):
                SurveySimulation object
            TK (TimeKeeping):
                TimeKeeping object
        """
        self.star_ensembles = {}
        sInds = np.arange(SS.TargetList.nStars)
        fZ = np.zeros([sInds.shape[0], len(self.comp_times)])
        times_mjd = Time(self.comp_times + TK.missionStart.mjd, format="mjd")
        for i in range(len(self.comp_times)):  # iterate through all times of year
            fZ[:, i] = SS.ZodiacalLight.fZ(
                SS.Observatory,
                SS.TargetList,
                sInds,
                times_mjd[i],
                SS.base_char_mode,
            ).to_value(SS.fZ_unit)

        # Prepare list to log all ensemble paths used in this run
        manifest_paths = []

        for sInd in tqdm(range(SS.TargetList.nStars), desc="Creating star ensembles"):
            # Generate cache path for this star
            ensemble_path = self.generate_star_ensemble_cache_path(TK, SS, sInd)
            manifest_paths.append(ensemble_path)

            # Check if cached ensemble exists for this star
            if ensemble_path.exists():
                star_ensemble = self.load_star_ensemble_cache(ensemble_path)
                if star_ensemble is not None:
                    self.star_ensembles[sInd] = star_ensemble
                    continue

            # Generate new ensemble for this star
            star_ensemble = self._generate_single_star_ensemble(SS, TK, sInd, fZ)
            self.star_ensembles[sInd] = star_ensemble

            # Save the ensemble to cache
            atomic_pickle_dump(star_ensemble, str(ensemble_path))

        # Write manifest file as a simple newline-delimited list of existing files.
        # This allows:  xargs -a manifest.txt rm -v --
        try:
            manifest_path = self.generate_star_ensemble_manifest_path(TK, SS)
            with open(manifest_path, "w") as f:
                for p in manifest_paths:
                    if Path(p).exists():
                        f.write(str(p) + "\n")
            self.vprint(
                f"Wrote star ensemble manifest (paths only) to '{manifest_path}'."
            )
        except Exception as e:
            self.vprint(f"Failed to write star ensemble manifest: {e}")

    def orbix_setup(self, trig_solver, SS):
        """Setup completeness by generating a set of orbits and propagating them."""
        PPop = self.PlanetPopulation
        TK, TL = SS.TimeKeeping, SS.TargetList

        # Generate cache filenames
        planets_path, orbits_path = self.generate_orbix_cache_path(TK)

        # Check if cached files exist and load them
        if planets_path.exists() and orbits_path.exists():
            if self.load_orbix_cache(planets_path, orbits_path):
                self.vprint("Orbix cache loaded successfully.")
            else:
                self._generate_orbix_data(trig_solver, TK, PPop)
                self.save_orbix_cache(planets_path, orbits_path)
        else:
            # Generate new data and cache it
            self.vprint("Cached orbix files not found, generating new data...")
            self._generate_orbix_data(trig_solver, TK, PPop)
            self.save_orbix_cache(planets_path, orbits_path)

        # Write an additional manifest for planets/orbits caches to aid debugging
        try:
            manifest_path = self.generate_orbix_cache_manifest_path(TK, SS)
            with open(manifest_path, "w") as f:
                if Path(planets_path).exists():
                    f.write(str(planets_path) + "\n")
                if Path(orbits_path).exists():
                    f.write(str(orbits_path) + "\n")
            self.vprint(
                f"Wrote orbix cache manifest (paths only) to '{manifest_path}'."
            )
        except Exception as e:
            self.vprint(f"Failed to write orbix cache manifest: {e}")

        # Alpha depends on the star's distance, computing it for each star
        self.alpha_factors = rad2arcsec / TL.dist.to_value(u.AU)

        # Generate star ensembles (with per-star caching)
        self._generate_star_ensembles(SS, TK)
        f_valid_vals = np.array([ens.f_valid for ens in self.star_ensembles.values()])
        valid_ens = np.where(f_valid_vals > 0.0)[0]
        self.characterizable_sInds = valid_ens
