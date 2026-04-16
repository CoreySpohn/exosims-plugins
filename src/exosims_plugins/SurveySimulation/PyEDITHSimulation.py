"""PyEDITHSimulation - Uses pyEDITH ETC for AYO-style yield calculation.

This module implements AYO's completeness methodology using pyEDITH for
integration time calculations, enabling direct comparison with EXOSIMS-based
calculations (AYOSimulation).

Key features:
1. Loads AYO's .ayo configuration file for ETC parameters
2. Loads AYO's observation schedule from CSV
3. Generates orbit×phase integration time grids using pyEDITH ETC
4. Calculates completeness and characterization times

Comparison with AYOSimulation:
- AYOSimulation uses EXOSIMS OpticalSystem.calc_intTime()
- PyEDITHSimulation uses pyEDITH calculate_exposure_time_or_snr()
"""

import copy
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation

# pyEDITH imports
from pyEDITH import (
    AstrophysicalScene,
    Observation,
    ObservatoryBuilder,
    calculate_exposure_time_or_snr,
    parse_input,
)
from tqdm import tqdm
from yippy import Coronagraph


class PyEDITHSimulation(SurveySimulation):
    """Survey simulation using pyEDITH ETC for integration time calculations.

    This module bypasses EXOSIMS scheduling and ETC calculations, using pyEDITH
    to match AYO's internal methodology for direct comparison.

    Args:
        ayo_config_file (str): Path to .ayo configuration file for pyEDITH
        ayo_observations_file (str): Path to AYO observations CSV
        n_orbits (int): Number of orbits to generate (default: 10000)
        n_phases (int): Number of phases per orbit (default: 100)
        eta_earth (float): Fraction of stars with Earth-like planets (default: 0.24)
        **specs: Additional specifications passed to parent class
    """

    def __init__(
        self,
        ayo_config_file=None,
        ayo_observations_file=None,
        coronagraph_path=None,
        sci_eng_dir=None,
        n_orbits=10000,
        n_phases=100,
        eta_earth=0.24,
        **specs,
    ):
        """Initialize PyEDITHSimulation.

        Args:
            ayo_config_file: Path to .ayo configuration file
            ayo_observations_file: Path to AYO observations CSV
            coronagraph_path: Direct path to YIP coronagraph files
            sci_eng_dir: Path to Sci-Eng-Interface (for EAC telescope/detector)
            n_orbits: Number of orbits to generate (default: 10000)
            n_phases: Number of phases per orbit (default: 100)
            eta_earth: Fraction of stars with Earth-like planets (default: 0.24)
            **specs: Additional specifications passed to parent class
        """
        # Store config before parent init
        self.ayo_config_file = ayo_config_file
        self.ayo_observations_file = ayo_observations_file
        self.coronagraph_path = coronagraph_path
        self.sci_eng_dir = sci_eng_dir
        self.n_orbits = n_orbits
        self.n_phases = n_phases
        self.eta_earth = eta_earth

        # Call parent init
        SurveySimulation.__init__(self, **specs)

        # Store in outspec for reproducibility
        self._outspec["ayo_config_file"] = self.ayo_config_file
        self._outspec["ayo_observations_file"] = self.ayo_observations_file
        self._outspec["coronagraph_path"] = self.coronagraph_path
        self._outspec["sci_eng_dir"] = self.sci_eng_dir
        self._outspec["n_orbits"] = self.n_orbits
        self._outspec["n_phases"] = self.n_phases
        self._outspec["eta_earth"] = self.eta_earth

        # Load pyEDITH configuration
        if self.ayo_config_file is not None:
            self._load_pyedith_config()
        else:
            self.pyedith_params = None

        # Load AYO observations and match stars
        if self.ayo_observations_file is not None:
            self.ayo_df = self._load_ayo_observations()
            self.star_mapping = self._match_stars_to_targetlist()
        else:
            self.ayo_df = None
            self.star_mapping = {}

        # Cache for integration time grids per star
        self.det_inttime_grids = {}  # {sInd: array[n_orbits, n_phases]}
        self.char_inttime_grids = {}  # For characterization mode

        # Pre-create pyEDITH observatory (shared across all ETC calls)
        self._pyedith_observatory = None
        self._pyedith_nlambda = None
        self._observatory_initialized = False
        self._base_observation = None
        self._base_scene = None

        # Number of samples for fast ETC (reduces 20000 to 30 calls per star)
        self.n_samples = 30

        # Vectorized ETC: yippy coronagraph with spline interpolators
        self._yippy_coronagraph = None  # Loaded on first use

        # Vectorized grid cache: {sInd: {"inttime": grid, "dMag": grid, "WA": grid}}
        self._grid_cache = {}

        # Detected planets tracking for revisits: {sInd: boolean array [n_orbits]}
        self._detected_planets = {}

    def _load_pyedith_config(self):
        """Load and parse the .ayo configuration file for pyEDITH."""
        self.vprint(f"Loading pyEDITH config from: {self.ayo_config_file}")

        # Parse the .ayo file (raw, without parse_parameters)
        raw_params, _ = parse_input.parse_input_file(
            self.ayo_config_file, secondary_flag=False
        )

        # Store raw params - needed for full pyEDITH initialization
        # (coronagraph loading requires complete parameter set)
        self.pyedith_raw_params = raw_params.copy()

        # Apply standard mappings (lambda -> wavelength, etc.)
        # These are CRITICAL for pyEDITH Observatory initialization
        for ayo_key, pyedith_key in {
            "lambda": "wavelength",
            "D": "diameter",
            "SNR": "snr",
            "photap_rad": "photometric_aperture_radius",
            "nexozodis": "nzodis",
            # Throughput params - CRITICAL for correct count rates
            "Toptical": "T_optical",
            "Tcontam": "T_contamination",
            # Detector params - pyEDITH expects these without det_ prefix
            "det_QE": "QE",
            "det_dQE": "dQE",
            "det_DC": "DC",
            "det_RN": "RN",
            "det_CIC": "CIC",
            "det_tread": "tread",
            "det_npix_multiplier": "npix_multiplier",
            "det_pixscale_mas": "pixscale_mas",
        }.items():
            if ayo_key in self.pyedith_raw_params:
                self.pyedith_raw_params[pyedith_key] = self.pyedith_raw_params[ayo_key]

        self.pyedith_raw_params["observing_mode"] = "IFS"  # Default to IFS

        # Map AYO parameter names to pyEDITH expected names
        self.det_params = self._map_ayo_to_pyedith_params(raw_params, is_detection=True)
        self.char_params = self._map_ayo_to_pyedith_params(
            raw_params, is_detection=False
        )

        self.vprint("pyEDITH configuration loaded successfully")

    def _map_ayo_to_pyedith_params(self, raw_params, is_detection=True):
        """Map AYO parameter names to pyEDITH expected names.

        Args:
            raw_params: Raw parameters from parse_input_file
            is_detection: If True, use detection params; else use char (sc_) params
        """
        params = {}

        # Direct mappings (AYO name -> pyEDITH name)

        ayo_to_pyedith = {
            "lambda": "wavelength",
            "D": "diameter",
            "SNR": "snr",
            "Toptical": "T_optical",
            "Tcontam": "T_contamination",
            "epswarmTrcold": "epswarmTrcold",
            "IWA": "minimum_IWA",
            "OWA": "maximum_OWA",
            "nexozodis": "nzodis",
            "photap_rad": "photometric_aperture_radius",
            "toverhead_fixed": "toverhead_fixed",
            "toverhead_multi": "toverhead_multi",
            "noisefloor_PPF": "noisefloor_PPF",
            "CRb_multiplier": "CRb_multiplier",
            "temperature": "temperature",
            "det_QE": "QE",
            "det_dQE": "dQE",
            "det_DC": "DC",
            "det_RN": "RN",
            "det_CIC": "CIC",
            "det_tread": "tread",
            "det_npix_multiplier": "npix_multiplier",
            "det_pixscale_mas": "pixscale_mas",
            "nchannels": "nchannels",
            "td_limit": "td_limit",
        }

        # Apply mappings with sc_ prefix for characterization
        for ayo_key, pyedith_key in ayo_to_pyedith.items():
            # For char mode, look for sc_ prefixed keys first
            if not is_detection:
                sc_key = f"sc_{ayo_key}"
                if sc_key in raw_params:
                    params[pyedith_key] = raw_params[sc_key]
                    continue
            # Otherwise use the non-prefixed key
            if ayo_key in raw_params:
                params[pyedith_key] = raw_params[ayo_key]

        # Special: observing mode based on broadband flag
        broadband_key = "sc_broadband" if not is_detection else "broadband"
        if raw_params.get(broadband_key, 0) == 1:
            params["observing_mode"] = "IMAGER"
        else:
            params["observing_mode"] = "IFS"

        # Special: coronagraph path
        coro_key = "sc_coronagraph1" if not is_detection else "coronagraph1"
        if coro_key in raw_params:
            params["coronagraph_path"] = raw_params[coro_key]
            params["coronagraph_type"] = "YIP"

        # Copy through other necessary params
        params["telescope_type"] = "Unobscured"
        params["detector_type"] = "H2RG"

        # Store raw params for reference
        params["_raw"] = raw_params

        return params

    def _load_ayo_observations(self):
        """Load and validate AYO observations CSV file."""
        self.vprint(f"Loading AYO observations from: {self.ayo_observations_file}")

        df = pd.read_csv(self.ayo_observations_file)

        # Validate required columns
        required_cols = ["HIP", "Exp Time (days)", "exoEarth candidate yield"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.vprint(f"Loaded {len(df)} AYO observations")
        return df

    def _match_stars_to_targetlist(self):
        """Match AYO stars to EXOSIMS TargetList via Hipparcos ID.

        Uses StarCatalog.get_name_by_hip() to convert HIP numbers to catalog
        names (e.g., TIC IDs), then matches against TargetList.Name.
        """
        TL = self.TargetList
        SC = self.StarCatalog
        star_mapping = {}

        ayo_hips = self.ayo_df["HIP"].unique()
        matched = 0

        # Use StarCatalog's get_name_by_hip() to convert HIP -> star_name
        # Then match star_name against TargetList.Name
        if hasattr(SC, "get_name_by_hip"):
            for hip in ayo_hips:
                if hip < 0:
                    continue
                hip = int(hip)

                # Get catalog name from HIP number
                star_name = SC.get_name_by_hip(hip)
                if star_name is None:
                    continue

                # Find matching index in TargetList
                matches = np.where(TL.Name == star_name)[0]
                if len(matches) > 0:
                    star_mapping[hip] = matches[0]
                    matched += 1
        else:
            # Fallback: try matching by Name string directly
            for hip in ayo_hips:
                if hip < 0:
                    continue
                hip = int(hip)
                matches = np.where(TL.Name == f"HIP {hip}")[0]
                if len(matches) > 0:
                    star_mapping[hip] = matches[0]
                    matched += 1

        self.vprint(f"Matched {matched}/{len(ayo_hips)} AYO stars to TargetList")
        return star_mapping

    def _create_orbix_planets(self, sInd):
        """Create an orbix Planets object for a given star.

        Uses the EXOSIMS PlanetPopulation to generate orbital elements,
        then constructs an orbix Planets object for JIT-compiled propagation.

        Args:
            sInd: Star index in TargetList

        Returns:
            planets: orbix.system.Planets object
        """
        import jax.numpy as jnp
        from orbix.kepler.shortcuts import get_grid_solver
        from orbix.system.planets import Planets

        TL = self.TargetList
        PPop = self.PlanetPopulation

        # Get star properties
        dist_pc = float(TL.dist[sInd].to(u.pc).value)
        Ms_kg = float(
            TL.MsTrue[sInd].to(u.kg).value
        )  # MsTrue already has solar mass units
        L_star = TL.L[sInd]  # Stellar luminosity in solar luminosities
        L_val = L_star.value if hasattr(L_star, "value") else float(L_star)

        # Sample orbital elements from PlanetPopulation
        a, e, p, Rp = PPop.gen_plan_params(self.n_orbits)
        inc, W, w = PPop.gen_angles(self.n_orbits)

        # Convert to JAX arrays with proper units (orbix expects radians, AU, kg)
        # CRITICAL: Scale semi-major axis by sqrt(L*) for HZ scaling
        # PlanetPopulation returns orbits for 1 L_sun star; must scale for actual luminosity
        a_au = a.to(u.AU).value
        a_scaled = a_au * np.sqrt(L_val)  # HZ scaling: a_HZ ~ sqrt(L*)
        a_jnp = jnp.array(a_scaled)
        e_jnp = jnp.array(e)
        W_jnp = jnp.array(W.to(u.rad).value)
        i_jnp = jnp.array(inc.to(u.rad).value)
        w_jnp = jnp.array(w.to(u.rad).value)
        M0_jnp = jnp.array(np.random.uniform(0, 2 * np.pi, self.n_orbits))
        t0_jnp = jnp.zeros(self.n_orbits)  # Epoch = 0 (reference time)
        Mp_jnp = jnp.ones(self.n_orbits)  # 1 Earth mass (not used for dMag)
        Rp_jnp = jnp.array(Rp.to(u.earthRad).value)
        p_jnp = jnp.array(p)

        # Create star mass and distance arrays (broadcast to n_orbits)
        Ms_jnp = jnp.full(self.n_orbits, Ms_kg)
        dist_jnp = jnp.full(self.n_orbits, dist_pc)

        # Create orbix Planets object
        planets = Planets(
            Ms_kg=Ms_jnp,
            dist_pc=dist_jnp,
            a_AU=a_jnp,
            e=e_jnp,
            W_rad=W_jnp,
            i_rad=i_jnp,
            w_rad=w_jnp,
            M0_rad=M0_jnp,
            t0_d=t0_jnp,
            Mp_Mearth=Mp_jnp,
            Rp_Rearth=Rp_jnp,
            Ag=p_jnp,
        )

        return planets

    def _calc_dmag_wa_phases(self, planets, n_phases):
        """Calculate dMag and WA at phase times using orbix JIT propagation.

        Args:
            planets: orbix.system.Planets object
            n_phases: Number of phase samples per orbit

        Returns:
            dMag: array [n_orbits, n_phases]
            WA: array [n_orbits, n_phases] in arcsec
        """
        import jax.numpy as jnp
        from orbix.kepler.shortcuts import get_grid_solver

        # Get the Kepler equation solver (trig-only returns sinE, cosE)
        solver = get_grid_solver(E=False, trig=True)

        # Get orbital periods (T is in days)
        T = np.array(planets.T_d)
        max_T = np.max(T)

        # Sample times uniformly over one orbital period
        # Use linspace over [0, max_T] to capture all phases
        phase_times = jnp.linspace(0.0, max_T, n_phases)

        # Use JIT-compiled propagation to get alpha (arcsec) and dMag
        alpha, dMag = planets.j_alpha_dMag(solver, phase_times)

        # Convert to numpy
        WA = np.array(alpha)  # Already in arcsec
        dMag_np = np.array(dMag)

        return dMag_np, WA

    def _build_inttime_grid(self, sInd, is_detection=True):
        """Pre-compute integration times for all orbit×phase combinations.

        This implements AYO's approach of caching integration times for
        fast completeness lookups.

        Args:
            sInd: Star index in TargetList
            is_detection: If True, use detection mode; else use char mode

        Returns:
            inttime_grid: array [n_orbits, n_phases] in days
        """
        TL = self.TargetList

        # Create orbix Planets object for this star (uses JIT-compiled propagation)
        planets = self._create_orbix_planets(sInd)

        # Calculate dMag and WA for all phases using orbix
        dMag, WA = self._calc_dmag_wa_phases(planets, self.n_phases)

        # Select parameters for detection or characterization
        params = copy.deepcopy(self.det_params if is_detection else self.char_params)

        # Get star-specific parameters
        dist_pc = float(TL.dist[sInd].to(u.pc).value)
        L_star = TL.L[sInd]
        params["distance"] = dist_pc
        params["Lstar"] = L_star.value if hasattr(L_star, "value") else L_star
        params["vmag"] = float(TL.Vmag[sInd])

        # Get star coordinates for zodi calculation
        coords = TL.coords[sInd]
        params["RA"] = coords.ra.deg
        params["dec"] = coords.dec.deg

        # Initialize grid
        inttime_grid = np.full((self.n_orbits, self.n_phases), np.inf)

        # Process each orbit×phase (pyEDITH handles IWA/OWA filtering internally)

        for orbit_idx in tqdm(
            range(self.n_orbits),
            desc=f"Building inttime grid for star {sInd}",
            leave=False,
        ):
            for phase_idx in range(self.n_phases):
                wa = WA[orbit_idx, phase_idx]
                dmag = dMag[orbit_idx, phase_idx]

                # Skip if outside working angle or invalid
                if np.isnan(dmag) or np.isinf(dmag):
                    continue

                # Set planet-specific parameters
                phase_params = copy.deepcopy(params)
                phase_params["angular_separation"] = wa  # arcsec
                phase_params["Fp_over_Fs"] = 10 ** (-dmag / 2.5)

                try:
                    # Calculate integration time using pyEDITH
                    texp = self._calc_pyedith_inttime(phase_params)
                    if np.isfinite(texp):
                        inttime_grid[orbit_idx, phase_idx] = texp
                except Exception:
                    continue

        return inttime_grid

    def _ensure_array_quantity(self, obj, attr_name, nlambda):
        """Ensure a Quantity attribute is an array of length nlambda."""
        if not hasattr(obj, attr_name):
            return
        val = getattr(obj, attr_name)
        if not isinstance(val, u.Quantity):
            return
        if np.isscalar(val.value) or val.ndim == 0:
            new_val = np.full(nlambda, float(val.value)) * val.unit
            setattr(obj, attr_name, new_val)

    def _get_pyedith_observatory(self, is_detection=True):
        """Get or create the pyEDITH observatory (cached for efficiency)."""
        if self._pyedith_observatory is not None:
            return self._pyedith_observatory

        # Determine coronagraph path
        params = self.det_params if is_detection else self.char_params
        coro_path = self.coronagraph_path or params.get("coronagraph_path")

        if coro_path is None:
            raise ValueError(
                "coronagraph_path must be specified either in __init__ or AYO config"
            )

        # Build observatory config
        observatory_config = {
            "telescope": "EAC1",
            "detector": "EAC1",
            "coronagraph_path": coro_path,
        }
        if self.sci_eng_dir:
            observatory_config["sci_eng_dir"] = self.sci_eng_dir

        self._pyedith_observatory = ObservatoryBuilder.create_observatory(
            observatory_config
        )
        return self._pyedith_observatory

    def _init_observatory_once(self, ref_params):
        """Initialize observatory once with full coronagraph load.

        This performs the expensive coronagraph initialization using a reference
        star, then allows fast ETC calls that only update separation/dMag.

        Args:
            ref_params: Dict with reference star parameters (distance, vmag, etc.)
        """
        if self._observatory_initialized:
            return

        self.vprint("Initializing pyEDITH observatory (single coronagraph load)...")

        # Get nlambda from wavelength
        wavelength = self.pyedith_raw_params.get("wavelength")
        if hasattr(wavelength, "__len__"):
            self._pyedith_nlambda = len(wavelength)
        else:
            self._pyedith_nlambda = 1
        nlambda = self._pyedith_nlambda

        # Build full params for reference star using raw AYO params as base
        # This is critical - pyEDITH's coronagraph loader needs the full param set
        params = self.pyedith_raw_params.copy()
        params.update(ref_params)
        params["regrid_wavelength"] = False  # Raw params already have correct format

        # Parse through pyEDITH
        parsed = parse_input.parse_parameters(params)
        if "epswarmTrcold" in parsed:
            del parsed["epswarmTrcold"]
        for k in ["snr", "photometric_aperture_radius", "nzodis", "observing_mode"]:
            if k in params:
                parsed[k] = params[k]

        # Ensure scene has required magnitude parameters
        # pyEDITH requires magV (scalar), mag (array), and delta_mag (array)
        # for magnitude-based flux calculation. magV must be scalar for exozodi calc.
        if "vmag" in ref_params:
            vmag_val = ref_params["vmag"]
            if hasattr(vmag_val, "__len__"):
                vmag_val = float(vmag_val[0])
            else:
                vmag_val = float(vmag_val)
            parsed["magV"] = vmag_val  # SCALAR, not array - required for exozodi
            parsed["mag"] = np.full(nlambda, vmag_val)  # Array of nlambda
        if "delta_mag" in ref_params:
            dmag_val = ref_params["delta_mag"]
            if hasattr(dmag_val, "__len__"):
                dmag_val = float(dmag_val[0])
            else:
                dmag_val = float(dmag_val)
            parsed["delta_mag"] = np.full(nlambda, dmag_val)
        else:
            # Default reference delta_mag for initialization
            parsed["delta_mag"] = np.full(nlambda, 25.0)

        # Create base observation
        self._base_observation = Observation()
        self._base_observation.load_configuration(parsed)
        self._base_observation.set_output_arrays()
        for attr in ["SNR", "photometric_aperture_radius"]:
            self._ensure_array_quantity(self._base_observation, attr, nlambda)

        # Create base scene
        self._base_scene = AstrophysicalScene()
        self._base_scene.load_configuration(parsed)
        self._base_scene.calculate_zodi_exozodi(parsed)
        for attr in ["Fp_over_Fs", "Fplanet"]:
            self._ensure_array_quantity(self._base_scene, attr, nlambda)

        # Get or create observatory
        observatory = self._get_pyedith_observatory()

        # Full configuration (loads coronagraph)
        ObservatoryBuilder.configure_observatory(
            observatory, parsed, self._base_observation, self._base_scene
        )

        self._observatory_initialized = True
        self.vprint("Observatory initialized successfully")

    def _calc_inttime_fast(self, dmag, wa):
        """Fast integration time calculation for a single observation.

        Uses pre-initialized observatory and only updates separation/dMag.
        Much faster than _calc_pyedith_inttime which reconfigures everything.

        Args:
            dmag: Planet delta magnitude
            wa: Working angle in arcsec

        Returns:
            Integration time in days
        """
        nlambda = self._pyedith_nlambda

        # Clone base objects to avoid mutation
        obs = Observation()
        obs.__dict__.update(self._base_observation.__dict__)

        scene = AstrophysicalScene()
        scene.__dict__.update(self._base_scene.__dict__)

        # Update only what changes per sample
        scene.separation = wa * u.arcsec
        scene.Fp_over_Fs = (
            np.full(nlambda, 10 ** (-0.4 * dmag)) * u.dimensionless_unscaled
        )

        # Run ETC without reconfiguring observatory
        calculate_exposure_time_or_snr(
            obs, scene, self._pyedith_observatory, verbose=False, mode="exposure_time"
        )

        exptime = obs.exptime
        if hasattr(exptime, "unit"):
            total_sec = float(np.nanmax(exptime.to(u.second).value))
        else:
            total_sec = float(np.nanmax(exptime)) * 3600  # hours to seconds

        return total_sec / 86400.0  # Convert to days

    def _calc_pyedith_inttime(self, params):
        """Calculate integration time using pyEDITH ETC.

        Args:
            params: Dictionary with all parameters needed for ETC

        Returns:
            Integration time in days
        """
        # Get nlambda from wavelength array
        wavelength = params.get("wavelength")
        if hasattr(wavelength, "__len__"):
            nlambda = len(wavelength)
        else:
            nlambda = 1
        self._pyedith_nlambda = nlambda

        # Ensure regrid_wavelength is set for IFS mode
        if params.get("observing_mode") == "IFS":
            params["regrid_wavelength"] = False

        # Parse parameters through pyEDITH
        parsed_params = parse_input.parse_parameters(params)

        # Remove epswarmTrcold - Observatory calculates it
        if "epswarmTrcold" in parsed_params:
            del parsed_params["epswarmTrcold"]

        # Ensure essential params are preserved
        for key in ["observing_mode", "snr", "photometric_aperture_radius", "nzodis"]:
            if key in params:
                parsed_params[key] = params[key]

        # Add star/planet params if arrays needed
        if "mag" not in parsed_params and "vmag" in params:
            parsed_params["mag"] = np.full(nlambda, float(params["vmag"]))
        if "delta_mag" not in parsed_params and "Fp_over_Fs" in params:
            fp_fs = params["Fp_over_Fs"]
            dmag = -2.5 * np.log10(fp_fs) if fp_fs > 0 else 30.0
            parsed_params["delta_mag"] = np.full(nlambda, dmag)

        # Ensure separation is set properly
        if "angular_separation" in params:
            parsed_params["separation"] = params["angular_separation"]

        # Set up Observation
        observation = Observation()
        observation.load_configuration(parsed_params)
        observation.set_output_arrays()

        # Ensure wavelength-indexed Quantities are arrays
        for attr in ["SNR", "photometric_aperture_radius", "psf_trunc_ratio"]:
            self._ensure_array_quantity(observation, attr, nlambda)

        # Set up AstrophysicalScene
        scene = AstrophysicalScene()
        scene.load_configuration(parsed_params)
        scene.calculate_zodi_exozodi(parsed_params)

        for attr in ["Fp_over_Fs", "Fplanet", "Fstar"]:
            self._ensure_array_quantity(scene, attr, nlambda)

        # Get or create observatory
        observatory = self._get_pyedith_observatory()

        # Configure observatory for this observation
        ObservatoryBuilder.configure_observatory(
            observatory, parsed_params, observation, scene
        )

        # Calculate exposure time
        calculate_exposure_time_or_snr(
            observation,
            scene,
            observatory,
            verbose=False,
            mode="exposure_time",
        )

        # Return total exposure time in days
        exptime = observation.exptime
        if hasattr(exptime, "unit"):
            total_hours = float(np.nansum(exptime.to(u.hour).value))
        else:
            total_hours = float(np.nansum(exptime))

        return total_hours / 24.0  # Convert to days

    def _get_completeness_fast(self, sInd, int_time):
        """Calculate detection completeness using sampling-based ETC.

        This is much faster than _get_completeness which computes a full
        n_orbits × n_phases grid. Instead, we:
        1. Generate orbit/phase points with orbix
        2. Sample up to n_samples valid observations
        3. Calculate min integration time for those samples
        4. Estimate completeness from fraction detectable

        Args:
            sInd: Star index in TargetList
            int_time: Available integration time [days]

        Returns:
            Estimated completeness value
        """
        TL = self.TargetList

        # Get nlambda for array sizing
        wavelength = self.det_params.get("wavelength")
        if hasattr(wavelength, "__len__"):
            nlambda = len(wavelength)
        else:
            nlambda = 1

        # Update scene for this star
        dist_pc = float(TL.dist[sInd].to(u.pc).value)
        vmag = float(TL.Vmag[sInd])
        L_star = TL.L[sInd]
        L_val = L_star.value if hasattr(L_star, "value") else L_star

        # Get star coordinates
        coords = TL.coords[sInd]

        # Get stellar temperature (Teff) if available
        if hasattr(TL, "Teff"):
            teff = TL.Teff[sInd]
            stellar_temp = float(teff.value) if hasattr(teff, "value") else float(teff)
        else:
            # Estimate from luminosity (rough MS approximation)
            stellar_temp = 5778.0 * (L_val**0.125)  # T ~ L^(1/8) for MS

        # Build params for this star
        # Estimate stellar radius from luminosity (R ~ sqrt(L) for main sequence)
        stellar_radius = np.sqrt(L_val) if L_val > 0 else 1.0

        star_params = {
            "distance": dist_pc,
            "Lstar": L_val,
            "vmag": vmag,
            "magV": vmag,  # Required by pyEDITH astrophysical_scene
            "mag": np.full(nlambda, vmag),  # Must be array of length nlambda
            "ra": coords.ra.deg,  # pyEDITH expects lowercase
            "dec": coords.dec.deg,
            "separation": 0.1,  # Will be overridden
            "delta_mag": np.full(nlambda, 25.0),  # Must be array of length nlambda
            "stellar_radius": stellar_radius,  # In solar radii
            "stellar_temperature": stellar_temp,  # Required for exozodi calculation
            "nzodis": self.det_params.get("nzodis", 3.0),
        }

        # Initialize observatory if needed
        self._init_observatory_once(star_params)

        # Generate planets and propagate orbits
        planets = self._create_orbix_planets(sInd)
        dMag, WA = self._calc_dmag_wa_phases(planets, self.n_phases)

        # Get OWA from params
        OWA_lod = self.det_params.get("maximum_OWA", 32.0)
        wavelength = self.det_params.get("wavelength")
        if hasattr(wavelength, "__len__"):
            lam = wavelength[0]
        else:
            lam = wavelength
        lam_nm = float(lam * 1e9) if lam < 1e-6 else float(lam)
        D_m = self.det_params.get("diameter", 6.0)
        lod_arcsec = (lam_nm * 1e-9 / D_m) * 206265
        OWA_arcsec = OWA_lod * lod_arcsec

        # Find valid observations (only OWA check - soft IWA)
        valid_mask = (WA <= OWA_arcsec) & np.isfinite(dMag)
        valid_indices = np.where(valid_mask)
        n_total_valid = len(valid_indices[0])

        if n_total_valid == 0:
            return 0.0

        # Sample up to n_samples points
        n_to_sample = min(self.n_samples, n_total_valid)
        if n_total_valid > n_to_sample:
            sample_idx = np.random.choice(n_total_valid, n_to_sample, replace=False)
        else:
            sample_idx = np.arange(n_total_valid)

        orbit_idx = valid_indices[0][sample_idx]
        phase_idx = valid_indices[1][sample_idx]

        # Calculate integration times for samples
        n_detectable = 0
        for i in range(len(orbit_idx)):
            dmag = dMag[orbit_idx[i], phase_idx[i]]
            wa = WA[orbit_idx[i], phase_idx[i]]

            try:
                sample_inttime = self._calc_inttime_fast(dmag, wa)
                if np.isfinite(sample_inttime) and sample_inttime <= int_time:
                    n_detectable += 1
            except Exception:
                continue

        # Estimate completeness: scale sample detection rate to full population
        # (unused but kept for debugging)
        # est_n_valid_detectable = sample_rate * n_total_valid

        # Completeness = (eta_earth / n_orbits) × expected detectable orbits
        # Each orbit can have multiple valid phases, so normalize properly
        n_orbits_with_valid = len(np.unique(orbit_idx))
        if n_orbits_with_valid > 0:
            orbit_detection_rate = n_detectable / n_to_sample
            completeness = self.eta_earth * orbit_detection_rate
        else:
            completeness = 0.0

        return completeness

    # =========================================================================
    # Vectorized ETC Methods (uses yippy interpolators for ~1000x speedup)
    # =========================================================================

    def _load_yippy_coronagraph(self):
        """Load yippy Coronagraph with spline interpolators for vectorized ETC.

        The yippy Coronagraph provides pre-computed radial performance curves:
        - throughput_interp(sep_lod) -> core throughput
        - raw_contrast_interp(sep_lod) -> stellar contrast
        - occ_trans_interp(sep_lod) -> sky transmission
        - core_area_interp(sep_lod) -> omega in (λ/D)²

        These enable fully vectorized count rate calculations.
        """
        if self._yippy_coronagraph is not None:
            return self._yippy_coronagraph

        # Get coronagraph path from config
        coro_path = self.det_params.get("coronagraph_path")
        if coro_path is None and self.coronagraph_path is not None:
            coro_path = self.coronagraph_path

        if coro_path is None:
            raise ValueError(
                "No coronagraph path specified. Set coronagraph_path or "
                "provide coronagraph1 in .ayo config."
            )

        self.vprint(f"Loading yippy Coronagraph from: {coro_path}")
        self._yippy_coronagraph = Coronagraph(coro_path)

        # Store key parameters
        self._coro_IWA_lod = float(self._yippy_coronagraph.IWA.value)
        self._coro_OWA_lod = float(self._yippy_coronagraph.OWA.value)

        self.vprint(
            f"  IWA={self._coro_IWA_lod:.2f} λ/D, OWA={self._coro_OWA_lod:.2f} λ/D"
        )

        return self._yippy_coronagraph

    def _calc_inttime_vectorized(self, sInd, dMag_grid, WA_grid):
        """Calculate integration times using vectorized pyEDITH count rates.

        Uses yippy coronagraph interpolators for throughput/Istar/omega at all
        separations, then applies pyEDITH's count rate formulas in numpy.

        Args:
            sInd: Star index in TargetList
            dMag_grid: Array of delta magnitudes [n_orbits, n_phases]
            WA_grid: Array of working angles in arcsec [n_orbits, n_phases]

        Returns:
            inttime_grid: Array of integration times in days [n_orbits, n_phases]
        """
        # Load coronagraph interpolators
        coro = self._load_yippy_coronagraph()

        # Ensure pyEDITH observatory is initialized
        if not self._observatory_initialized:
            self._init_star_for_vectorized_etc(sInd)

        # Get pyEDITH objects
        scene = self._base_scene
        obs = self._base_observation
        observatory = self._pyedith_observatory

        # Get observation parameters
        params = self.det_params
        wavelength_um = params.get("wavelength", 0.5)
        if hasattr(wavelength_um, "__len__"):
            wavelength_um = float(np.mean(wavelength_um))
        diameter_m = params.get("diameter", 6.0)

        # λ/D in arcsec
        lod_arcsec = (wavelength_um * 1e-6 / diameter_m) * 206265.0

        # Convert WA to λ/D for coronagraph lookup
        sep_lod = WA_grid / lod_arcsec  # [n_orbits, n_phases]
        shape = sep_lod.shape
        sep_flat = sep_lod.flatten()

        # Get coronagraph performance at all separations (vectorized!)
        throughput = np.clip(coro.throughput_interp(sep_flat), 0, 1).reshape(shape)
        Istar_diffraction = np.maximum(
            coro.core_intensity_interp(sep_flat), 1e-20
        ).reshape(shape)
        # Note: skytrans (occ_trans) not used - we use core_throughput for zodi per AYO

        # Omega (aperture area) - use configured photometric aperture radius
        # AYO uses photap_rad = 0.85 λ/D, giving omega = π × 0.85² = 2.27 (λ/D)²
        # yippy's core_area default is 0.7 λ/D = 1.54 (λ/D)², causing 1.47x diff
        photap_rad = params.get("photometric_aperture_radius", 0.85)  # λ/D
        omega = np.full(shape, np.pi * photap_rad**2)

        # Use yippy's core_mean_intensity directly (validated against AYO)
        # Note: Deep think suggested using contrast floor, but our CRbs validation
        # showed 1.0000 ratio with yippy's Istar. The 1e-10 is the RAW coronagraph
        # contrast, but yippy's core_mean_intensity already accounts for aperture
        # integration. The throughput difference (1.14x) explains remaining variance.
        Istar = Istar_diffraction

        # Planet flux ratio from dMag
        Fp_over_Fs = 10 ** (-0.4 * dMag_grid)  # [n_orbits, n_phases]

        # Get fluxes from pyEDITH scene (use first wavelength for simplicity)
        # These are per-star constants
        F0 = float(scene.F0[0].value) if hasattr(scene, "F0") else 1e10
        Fs_over_F0 = (
            float(scene.Fs_over_F0[0].value) if hasattr(scene, "Fs_over_F0") else 1.0
        )
        Fzodi = (
            float(scene.Fzodi_list[0].value) if hasattr(scene, "Fzodi_list") else 1e-8
        )

        # Observatory parameters
        area_cm2 = float(observatory.telescope.Area.to(u.cm**2).value)
        total_throughput = float(observatory.total_throughput[0].value)
        # delta_wavelength comes from observation, not observatory
        if hasattr(obs, "delta_wavelength") and obs.delta_wavelength is not None:
            dlambda_nm = float(obs.delta_wavelength[0].to(u.nm).value)
        else:
            # Fallback: compute from wavelength range
            wavelength_nm = params.get("wavelength", 500) * 1e3  # um -> nm
            if hasattr(wavelength_nm, "__len__"):
                dlambda_nm = float(wavelength_nm[-1] - wavelength_nm[0])
            else:
                dlambda_nm = float(wavelength_nm * 0.2)  # ~20% bandwidth estimate
        nchannels = observatory.coronagraph.nchannels
        pixscale = float(observatory.coronagraph.pixscale.value)  # λ/D per pixel

        # Observation parameters
        SNR = float(np.mean(obs.SNR.value)) if hasattr(obs.SNR, "value") else 7.0
        noisefloor_PPF = params.get("noisefloor_PPF", 30.0)
        # Note: CRb_multiplier is deprecated - AYO uses fixed 2× for ADI noise
        toverhead_multi = float(observatory.telescope.toverhead_multi)
        toverhead_fixed = float(observatory.telescope.toverhead_fixed.to(u.s).value)

        # Common flux factor
        flux_factor = (
            F0 * Fs_over_F0 * area_cm2 * total_throughput * dlambda_nm * nchannels
        )

        # Planet count rate: CRp = flux × Fp/Fs × throughput [electrons/s]
        CRp = flux_factor * Fp_over_Fs * throughput

        # Stellar leakage: CRbs = F0 × Fs/F0 × Istar × A × throughput × Δλ × omega_core
        # Uses omega (core_area in λ/D²), NOT 1/pixscale²
        CRbs = flux_factor * Istar * omega

        # CRbz = Fzodi × omega_arcsec² × A × (T_core × QE × dQE) × Δλ
        # IMPORTANT: AYO uses core_throughput for zodi, NOT optical throughput!
        # This matches EXOSIMS use_core_thruput_for_ez=True behavior
        omega_arcsec2 = omega * (lod_arcsec**2)  # (λ/D)² to arcsec²
        # Get core throughput-based total (T_core × QE × dQE)
        QE = float(observatory.detector.QE[0].value)
        dQE = float(observatory.detector.dQE[0].value)
        zodi_throughput = throughput * QE * dQE  # throughput here is T_core
        CRbz = (
            Fzodi * omega_arcsec2 * area_cm2 * zodi_throughput * dlambda_nm * nchannels
        )

        # Exozodi: CRbez = F_exozodi × omega_arcsec² × A × zodi_throughput × Δλ
        # Scales with 1/dist² (exozodi is farther than local)
        dist_pc = float(self.TargetList.dist[sInd].value)
        nexozodis = params.get("nexozodis", 3.0)  # Default 3 zodis
        # F_exozodi = Fzodi × nexozodis (relative to local zodi)
        F_exozodi = Fzodi * nexozodis
        # Exozodi illumination scales with 1/r² where r is in AU projected
        # WA_grid is in arcsec, convert to AU: AU = arcsec × dist_pc
        WA_au = WA_grid * dist_pc  # AU (approximate for small angles)
        # Avoid division by zero
        WA_au_safe = np.maximum(WA_au, 0.01)
        CRbez = (
            F_exozodi
            * omega_arcsec2
            * area_cm2
            * zodi_throughput
            * dlambda_nm
            * nchannels
            / (WA_au_safe**2)
        )

        # Detector noise: CRbd = (DC + RN²/tread + CIC/t_photon) × npix
        det_DC = params.get("det_DC", 0.0003)  # Dark current e-/pix/s
        det_RN = params.get("det_RN", 0.0)  # Read noise e-/pix (PC mode = 0)
        det_tread = params.get("det_tread", 10.0)  # Read time (s)
        det_CIC = params.get("det_CIC", 0.0013)  # Clock induced charge
        npix_multiplier = params.get("npix_multiplier", 1.0)
        # npix = npix_multiplier × omega × (1/pixscale²) × nchannels
        npix = npix_multiplier * omega / (pixscale**2) * nchannels
        # For now use fixed t_photon estimate (will refine with iteration if needed)
        t_photon_count = 1.0 / (6.73 * np.maximum(CRp / npix, 1e-10))
        CRbd = (det_DC + det_RN**2 / det_tread + det_CIC / t_photon_count) * npix

        # Noise floor: CRnf = CRbs / PPF (NOT scaled by SNR - that's in denominator)
        # The noisefloor term represents systematic speckle fluctuations
        # Reference: AYO C code line 522: tempCRnffactor = SNR * CRbsfactor * noisefloor
        # But noisefloor_interp = Istar/PPF, so CRnf = SNR × CRbs × (1/PPF) × omega
        CRnf = SNR * flux_factor * (Istar / noisefloor_PPF) * omega

        # Total background: CRbs + CRbz + CRbez + CRbd
        CRb = CRbs + CRbz + CRbez + CRbd

        # AYO Exposure time formula (accounts for ADI noise with 2× background factor):
        # cp = (CRp + 2×CRb) / (CRp² - CRnf²)  [CRnf already includes SNR!]
        # t = SNR² × cp × toverhead_multi + toverhead_fixed
        # Reference: AYO C code line 569: cp = (CRp + twoCRb) / (CRp*CRp - SNRCRpfloor*SNRCRpfloor)
        # Note: SNRCRpfloor = CRnoisefloor which already has SNR factored in
        numerator = CRp + 2 * CRb
        # CRnf already has omega factored in from its calculation above
        denominator = CRp**2 - CRnf**2  # NO extra SNR² here - it's already in CRnf

        # DEBUG: Print diagnostic info on first call
        if not hasattr(self, "_etc_debug_done"):
            self._etc_debug_done = True
            valid_mask = (sep_lod >= self._coro_IWA_lod) & (
                sep_lod <= self._coro_OWA_lod
            )
            valid_crp = CRp[valid_mask]
            valid_crnf_omega = (CRnf * omega)[valid_mask]  # Both are 2D now
            valid_denom = denominator[valid_mask]
            valid_tp = throughput[valid_mask]
            valid_Istar = Istar[valid_mask]
            self.vprint("  ETC Debug:")
            self.vprint(f"    F0={F0:.2e}, Fs/F0={Fs_over_F0:.2e}")
            self.vprint(f"    area={area_cm2:.2e} cm², dlambda={dlambda_nm:.1f} nm")
            self.vprint(f"    total_throughput={total_throughput:.4f}")
            self.vprint(
                f"    (components: optics={float(observatory.optics_throughput[0].value):.3f}, "
                f"QE={float(observatory.detector.QE[0].value):.3f}, "
                f"dQE={float(observatory.detector.dQE[0].value):.3f})"
            )
            self.vprint(
                f"    core_throughput range: {np.min(valid_tp):.4f} "
                f"to {np.max(valid_tp):.4f}"
            )
            self.vprint(
                f"    Istar range: {np.min(valid_Istar):.4e} "
                f"to {np.max(valid_Istar):.4e}"
            )
            self.vprint(f"    noisefloor_PPF={noisefloor_PPF:.0f}, SNR={SNR:.0f}")
            self.vprint(f"    flux_factor={flux_factor:.2e}")
            if len(valid_crp) > 0:
                self.vprint(
                    f"    CRp range: {np.min(valid_crp):.2e} to {np.max(valid_crp):.2e}"
                )
                self.vprint(
                    f"    CRnf×omega range: {np.min(valid_crnf_omega):.2e} "
                    f"to {np.max(valid_crnf_omega):.2e}"
                )
                self.vprint(
                    f"    denom (CRp²-CRnf²) range: {np.min(valid_denom):.2e} "
                    f"to {np.max(valid_denom):.2e}"
                )
                pos_denom = np.sum(valid_denom > 0)
                self.vprint(f"    Positive denominator: {pos_denom}/{len(valid_denom)}")

        with np.errstate(invalid="ignore", divide="ignore"):
            cp = numerator / denominator  # seconds per electron
            inttime_sec = SNR**2 * cp * toverhead_multi + toverhead_fixed

            # Handle noise floor limit (denominator <= 0)
            inttime_sec = np.where(denominator > 0, inttime_sec, np.inf)

            # Handle negative times
            inttime_sec = np.where(inttime_sec > 0, inttime_sec, np.inf)

        # Mark outside OWA as infinite
        inttime_sec = np.where(sep_lod <= self._coro_OWA_lod, inttime_sec, np.inf)

        # Convert to days
        inttime_days = inttime_sec / 86400.0

        # Apply exposure time limit from config
        td_limit = params.get("td_limit", 10.0)  # days
        inttime_days = np.where(inttime_days <= td_limit, inttime_days, np.inf)

        # Apply nrolls multiplier (for 360° sky coverage)
        # Reference: AYO C code line 586: if(nrolls != 1) temptp *= nrolls
        nrolls = params.get("nrolls", 1)  # Default 1 (coronagraph header default)
        if nrolls != 1:
            # Only multiply finite times (not inf)
            inttime_days = np.where(
                np.isfinite(inttime_days), inttime_days * nrolls, inttime_days
            )

        return inttime_days

    def _init_star_for_vectorized_etc(self, sInd):
        """Initialize pyEDITH objects for a specific star.

        The observatory should already be initialized via _get_completeness_fast
        before this is called. This method re-initializes with the specific star.
        """
        TL = self.TargetList

        # Get wavelength info from det_params
        wavelength = self.det_params.get("wavelength")
        if hasattr(wavelength, "__len__"):
            nlambda = len(wavelength)
        else:
            nlambda = 1
        self._pyedith_nlambda = nlambda

        # Get star parameters
        vmag = float(TL.Vmag[sInd])
        dist_pc = float(TL.dist[sInd].to(u.pc).value)
        L_star = TL.L[sInd]
        L_val = L_star.value if hasattr(L_star, "value") else float(L_star)

        # Get coordinates
        coords = TL.coords[sInd]
        ra_deg = coords.ra.deg
        dec_deg = coords.dec.deg

        # Get stellar properties
        stellar_radius = 1.0  # Default solar radius
        if hasattr(TL, "stellar_radius") and TL.stellar_radius is not None:
            sr = TL.stellar_radius[sInd]
            stellar_radius = sr.value if hasattr(sr, "value") else float(sr)
        stellar_temp = 5778.0  # Default solar temp
        if hasattr(TL, "stellar_temp") and TL.stellar_temp is not None:
            st = TL.stellar_temp[sInd]
            stellar_temp = st.value if hasattr(st, "value") else float(st)

        # Merge with detection params for full config
        star_params = {
            "distance": dist_pc,
            "Lstar": L_val,
            "vmag": vmag,
            "magV": vmag,
            "mag": np.full(nlambda, vmag),
            "ra": ra_deg,
            "dec": dec_deg,
            "separation": 0.1,
            "delta_mag": np.full(nlambda, 25.0),
            "stellar_radius": stellar_radius,
            "stellar_temperature": stellar_temp,
            "nzodis": self.det_params.get("nzodis", 3.0),
        }

        # Initialize observatory with star params
        self._init_observatory_once(star_params)

    def _get_grid(self, sInd):
        """Get or build the integration time grid for a star (cached).

        Args:
            sInd: Star index in TargetList

        Returns:
            dict with keys: "inttime", "dMag", "WA", "planets"
        """
        if sInd in self._grid_cache:
            return self._grid_cache[sInd]

        # Generate planets and phase grid
        planets = self._create_orbix_planets(sInd)
        dMag, WA = self._calc_dmag_wa_phases(planets, self.n_phases)

        # Calculate vectorized integration times
        inttime = self._calc_inttime_vectorized(sInd, dMag, WA)

        # Cache and return
        self._grid_cache[sInd] = {
            "inttime": inttime,
            "dMag": dMag,
            "WA": WA,
            "planets": planets,
        }

        return self._grid_cache[sInd]

    def _compute_inttime_scale(self):
        """Compute scale factor to convert real time (days) to relative units.

        Uses median of AYO exposure times and median of finite grid times
        to establish the mapping.
        """
        # Get reference exposure time from AYO data
        if self.ayo_df is not None and len(self.ayo_df) > 0:
            ref_time_days = float(self.ayo_df["Exp Time (days)"].median())
        else:
            ref_time_days = 0.2  # Default ~5 hours

        # Collect finite integration times from first few cached grids
        all_finite_times = []
        for sInd, grid in list(self._grid_cache.items())[:5]:
            finite = grid["inttime"][np.isfinite(grid["inttime"])]
            all_finite_times.extend(finite.tolist())

        if len(all_finite_times) == 0:
            # No finite times yet - use a default scale
            self._inttime_scale = 5000.0  # Empirical default
            self.vprint(f"  Calibration: using default scale={self._inttime_scale:.0f}")
            return

        # Median of all finite relative times
        median_rel = np.median(all_finite_times)

        # Scale factor: median_rel / ref_time gives the mapping
        # So: int_time_days * scale = int_time_relative
        self._inttime_scale = median_rel / ref_time_days

        self.vprint(
            f"  Calibration: scale={self._inttime_scale:.0f} "
            f"(median_rel={median_rel:.0f}, ref={ref_time_days:.3f} days)"
        )

    def _calc_completeness_from_grid(self, sInd, int_time, detected_mask=None):
        """Calculate completeness from cached integration time grid - O(n_orbits).

        This is instant once the grid is built - just a threshold comparison.

        Args:
            sInd: Star index
            int_time: Available integration time (days)
            detected_mask: Optional boolean array [n_orbits] of previously
                          detected planets (True = already detected, exclude)

        Returns:
            Completeness value (fraction of undetected orbits detectable)
        """
        grid = self._get_grid(sInd)
        inttime = grid["inttime"]  # [n_orbits, n_phases]

        # For each orbit, find minimum integration time across all phases
        # (orbit is detectable if ANY phase is detectable)
        min_inttime_per_orbit = np.nanmin(inttime, axis=1)

        # Detectable if min_inttime <= available time (in days)
        detectable = min_inttime_per_orbit <= int_time

        # Apply detected_mask to exclude already-found planets
        if detected_mask is not None:
            detectable = detectable & ~detected_mask
            n_available = np.sum(~detected_mask)
        else:
            n_available = len(detectable)

        if n_available == 0:
            return 0.0

        # Completeness = fraction of orbits that are detectable
        # Scale by eta_earth since orbits represent the planet population
        completeness = self.eta_earth * np.sum(detectable) / n_available

        return completeness

    def _mark_detected(self, sInd, int_time):
        """Mark planets as detected for this visit (for revisit tracking).

        Updates _detected_planets[sInd] to track which orbits have been
        detected across visits.

        Args:
            sInd: Star index
            int_time: Integration time used for this visit (days)
        """
        grid = self._get_grid(sInd)
        min_inttime = np.nanmin(grid["inttime"], axis=1)
        newly_detected = min_inttime <= int_time

        if sInd in self._detected_planets:
            self._detected_planets[sInd] |= newly_detected
        else:
            self._detected_planets[sInd] = newly_detected.copy()

    def get_inttime_curve(self, sInd, int_times=None):
        """Get completeness vs integration time curve for debugging.

        Args:
            sInd: Star index
            int_times: Array of integration times to evaluate (days).
                      Defaults to logspace(0.001, 10, 50).

        Returns:
            dict with "int_times" and "completeness" arrays
        """
        if int_times is None:
            int_times = np.logspace(-3, 1, 50)  # 0.001 to 10 days

        grid = self._get_grid(sInd)
        min_inttime = np.nanmin(grid["inttime"], axis=1)

        completeness = []
        for t in int_times:
            detectable = np.sum(min_inttime <= t)
            c = self.eta_earth * detectable / len(min_inttime)
            completeness.append(c)

        return {
            "int_times": int_times,
            "completeness": np.array(completeness),
            "n_orbits": len(min_inttime),
        }

    def _get_completeness(self, sInd, int_time, is_detection=True):
        """Calculate completeness for given integration time.

        AYO formula: (η_⊕ / n_orbits) × count(orbits where min_phase(t) < int_time)

        Args:
            sInd: Star index
            int_time: Available integration time [days]
            is_detection: If True, use detection grid; else use char grid

        Returns:
            Completeness value
        """
        # Get or build grid
        cache = self.det_inttime_grids if is_detection else self.char_inttime_grids

        if sInd not in cache:
            cache[sInd] = self._build_inttime_grid(sInd, is_detection)

        grid = cache[sInd]  # [n_orbits, n_phases]

        # For each orbit, get minimum int time across phases
        min_inttime_per_orbit = np.min(grid, axis=1)

        # Count orbits that are detectable
        n_detectable = np.sum(min_inttime_per_orbit < int_time)

        # Scale by eta_earth
        completeness = (self.eta_earth / self.n_orbits) * n_detectable

        return completeness

    def run_sim(self, debug_limit=None):
        """Run the pyEDITH-based yield calculation.

        Args:
            debug_limit: If set, only process first N unique stars (for faster
                debugging)
        """
        if self.det_params is None:
            raise ValueError(
                "No pyEDITH configuration loaded. "
                "Specify ayo_config_file in initialization."
            )

        if self.ayo_df is None:
            raise ValueError(
                "No AYO observations loaded. "
                "Specify ayo_observations_file in initialization."
            )

        # Initialize results
        results = {
            "observations": [],
            "per_star_totals": {},
            "total_ayo_yield": 0.0,
            "total_pyedith_yield": 0.0,
            "unmatched_stars": 0,
        }

        # Sort by visit time
        sorted_df = self.ayo_df.sort_values("Visit dt (years)")

        # Apply debug limit if specified
        if debug_limit is not None:
            unique_hips = sorted_df["HIP"].unique()[:debug_limit]
            sorted_df = sorted_df[sorted_df["HIP"].isin(unique_hips)]
            self.vprint(f"\nDEBUG MODE: Processing only {len(unique_hips)} stars")

        self.vprint("\nCalculating completeness using pyEDITH ETC...")
        self.vprint(
            f"  n_orbits={self.n_orbits}, n_phases={self.n_phases}, "
            f"n_samples={self.n_samples}"
        )

        for idx, row in tqdm(
            sorted_df.iterrows(), total=len(sorted_df), desc="Processing observations"
        ):
            hip = row["HIP"]
            int_time_days = row["Exp Time (days)"]
            ayo_yield = row["exoEarth candidate yield"]
            visit_num = row.get("Visit #", 1)

            # Skip invalid entries
            if hip < 0 or int_time_days <= 0:
                continue

            # Match to TargetList
            if hip not in self.star_mapping:
                results["unmatched_stars"] += 1
                continue

            sInd = self.star_mapping[hip]

            # Calculate completeness using vectorized grid method
            detected_mask = self._detected_planets.get(sInd)
            pyedith_comp = self._calc_completeness_from_grid(
                sInd, int_time_days, detected_mask=detected_mask
            )

            # Mark any detectable planets as detected for revisit tracking
            self._mark_detected(sInd, int_time_days)

            # Store observation result
            obs_result = {
                "HIP": hip,
                "sInd": sInd,
                "visit": visit_num,
                "int_time_days": int_time_days,
                "ayo_yield": ayo_yield,
                "pyedith_yield": pyedith_comp,
            }
            results["observations"].append(obs_result)

            # Accumulate totals
            results["total_ayo_yield"] += ayo_yield
            results["total_pyedith_yield"] += pyedith_comp

            # Per-star totals
            if hip not in results["per_star_totals"]:
                results["per_star_totals"][hip] = {
                    "ayo_total": 0.0,
                    "pyedith_total": 0.0,
                }
            results["per_star_totals"][hip]["ayo_total"] += ayo_yield
            results["per_star_totals"][hip]["pyedith_total"] += pyedith_comp

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """Print comparison summary."""
        print("\n" + "=" * 70)
        print("AYO vs PyEDITH Yield Comparison Summary")
        print("=" * 70)
        print(f"Total observations processed: {len(results['observations'])}")
        print(f"Unique stars matched: {len(results['per_star_totals'])}")
        print(f"Unmatched observations: {results['unmatched_stars']}")

        ayo_total = results["total_ayo_yield"]
        pyedith_total = results["total_pyedith_yield"]
        ratio = ayo_total / pyedith_total if pyedith_total > 0 else float("inf")

        print(f"\n{'Metric':<30} {'AYO':<15} {'PyEDITH':<15} {'Ratio':<10}")
        print("-" * 70)
        print(
            f"{'Total Yield':<30} {ayo_total:<15.4f} {pyedith_total:<15.4f} {ratio:.2f}x"
        )
        print("=" * 70)
