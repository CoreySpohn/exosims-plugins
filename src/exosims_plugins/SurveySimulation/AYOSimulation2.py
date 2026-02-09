"""PyEDITHSimulation - Uses pyEDITH ETC for AYO-style yield calculation.

This module implements AYO's completeness methodology using pyEDITH for
integration time calculations, enabling direct comparison with EXOSIMS-based
calculations (AYOSimulation).

Key features:
1. Loads AYO's .ayo configuration file for ETC parameters
2. Loads AYO's observation schedule from CSV
3. Generates orbit×phase integration time grids using pyEDITH ETC
4. Calculates completeness and characterization times
5. Implements optimized photometric aperture selection matching AYO algorithm

Comparison with AYOSimulation:
- AYOSimulation uses EXOSIMS OpticalSystem.calc_intTime()
- PyEDITHSimulation uses pyEDITH calculate_exposure_time_or_snr()
"""

import copy
from pathlib import Path

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pandas as pd
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from orbix.kepler.shortcuts import get_grid_solver

# pyEDITH imports
from pyEDITH import (
    AstrophysicalScene,
    Observation,
    ObservatoryBuilder,
    calculate_exposure_time_or_snr,
    parse_input,
)
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from yippy import Coronagraph

# Get Kepler solver
solver = get_grid_solver(E=False, trig=True, jit=True)


class AYOSimulation2(SurveySimulation):
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
        noise_floor_csv=None,
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
        self.noise_floor_csv = noise_floor_csv
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
            # Load per-star detection wavelengths from target_list.csv
            self._load_star_detection_wavelengths()
        else:
            self.ayo_df = None
            self.star_mapping = {}
            self.hip_to_det_wavelength = {}

        # Cache for integration time grids per star
        self.det_inttime_grids = {}  # {sInd: array[n_orbits, n_phases]}
        self.char_inttime_grids = {}  # For characterization mode

        # Pre-create pyEDITH observatory (shared across all ETC calls)
        self._pyedith_observatory = None
        self._pyedith_nlambda = None
        self._observatory_initialized = False
        self._base_observation = None
        self._base_scene = None

        # Vectorized ETC: yippy coronagraph with spline interpolators
        self._yippy_coronagraph = None  # Loaded on first use
        self._aperture_interpolators = None  # For dynamic aperture selection
        self._noise_floor_interp = None  # Loaded from CSV if provided

        # Vectorized grid cache: {sInd: {"inttime": grid, "dMag": grid, "WA": grid}}
        self._grid_cache = {}

        # Detected planets tracking for revisits: {sInd: boolean array [n_orbits]}
        self._detected_planets = {}

        # Planets cache for time-based calculations: {sInd: Planets object}
        self._planets_cache = {}

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
            "psf_trunc_ratio": "psf_trunc_ratio",  # Include AYO optimization list
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

    def _load_star_detection_wavelengths(self):
        """Load per-star detection wavelengths from target_list.csv.

        The target_list.csv file (in same directory as observations.csv) contains
        a 'Detection Wavelength (microns)' column specifying the optimal detection
        wavelength for each star. This varies from ~0.4 to ~1.0 microns based on
        stellar type.

        Creates:
            self.hip_to_det_wavelength: dict mapping HIP number to wavelength (um)
        """
        import os

        # Find target_list.csv in same directory as observations
        obs_dir = os.path.dirname(self.ayo_observations_file)
        target_list_path = os.path.join(obs_dir, "target_list.csv")

        self.hip_to_det_wavelength = {}

        if not os.path.exists(target_list_path):
            self.vprint(f"Warning: target_list.csv not found at {target_list_path}")
            self.vprint("  Using default wavelength for all stars")
            return

        try:
            target_df = pd.read_csv(target_list_path)

            # Check for wavelength column
            wl_col = "Detection Wavelength (microns)"
            if wl_col not in target_df.columns:
                # Try alternative column names
                alt_names = [
                    " Detection Wavelength (microns)",
                    "Detection Wavelength",
                ]
                for alt in alt_names:
                    if alt in target_df.columns:
                        wl_col = alt
                        break
                else:
                    self.vprint("Warning: No wavelength column found in target_list")
                    return

            # Create HIP -> wavelength mapping
            for _, row in target_df.iterrows():
                hip = row.get("HIP")
                if pd.isna(hip) or hip < 0:
                    continue
                hip = int(hip)
                wavelength = row.get(wl_col)
                if pd.notna(wavelength):
                    self.hip_to_det_wavelength[hip] = float(wavelength)

            self.vprint(
                f"Loaded detection wavelengths for {len(self.hip_to_det_wavelength)} "
                f"stars from target_list.csv"
            )

            # Report wavelength range
            if self.hip_to_det_wavelength:
                wls = list(self.hip_to_det_wavelength.values())
                self.vprint(
                    f"  Wavelength range: {min(wls):.3f} - {max(wls):.3f} microns"
                )

        except Exception as e:
            self.vprint(f"Warning: Failed to load target_list.csv: {e}")
            self.hip_to_det_wavelength = {}

    def _create_orbix_planets(self, sInd):
        """Create an orbix Planets object for a given star.

        Uses the EXOSIMS PlanetPopulation to generate orbital elements,
        then constructs an orbix Planets object for JIT-compiled propagation.

        Args:
            sInd: Star index in TargetList

        Returns:
            planets: orbix.system.Planets object
        """
        from orbix.system.planets import Planets

        TL = self.TargetList
        PPop = self.Completeness.PlanetPopulation

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
            Ms=Ms_jnp,
            dist=dist_jnp,
            a=a_jnp,
            e=e_jnp,
            W=W_jnp,
            i=i_jnp,
            w=w_jnp,
            M0=M0_jnp,
            t0=t0_jnp,
            Mp=Mp_jnp,
            Rp=Rp_jnp,
            p=p_jnp,
        )

        return planets

    def _calc_dmag_wa_phases(self, planets, n_phases):
        """Calculate dMag and WA on a uniform Mean Anomaly grid.

        This samples Mean Anomaly uniformly over [0, 2π] for EVERY orbit,
        ensuring uniform phase coverage regardless of orbital period.

        Args:
            planets: orbix.system.Planets object
            n_phases: Number of phase samples per orbit

        Returns:
            dMag: array [n_orbits, n_phases]
            WA: array [n_orbits, n_phases] in arcsec
        """
        # Create uniform Mean Anomaly grid [0, 2π]
        M_grid = jnp.linspace(0, 2 * jnp.pi, n_phases)

        # Get orbital parameters
        n_arr = np.array(planets.n)  # Mean motion [n_orbits]
        M0_arr = np.array(planets.M0)  # Initial mean anomaly [n_orbits]
        t0_arr = np.array(planets.t0)  # Epoch [n_orbits]
        n_orbits = len(n_arr)

        # Calculate times to reach each Mean Anomaly for each orbit
        # M(t) = n*(t - t0) + M0  =>  t = (M - M0)/n + t0
        t_grid = np.zeros((n_orbits, n_phases))
        for j in range(n_phases):
            M_target = float(M_grid[j])
            t_grid[:, j] = (M_target - M0_arr) / n_arr + t0_arr

        # Initialize output arrays
        WA = np.zeros((n_orbits, n_phases))
        dMag_np = np.zeros((n_orbits, n_phases))

        # Propagate all orbits at their specific times for each phase
        for j in range(n_phases):
            t_column = jnp.array(t_grid[:, j])  # [n_orbits] times
            alpha_j, dMag_j = planets.j_alpha_dMag(solver, t_column)
            # Extract diagonal: orbit i at its specific time
            WA[:, j] = np.diag(np.array(alpha_j))
            dMag_np[:, j] = np.diag(np.array(dMag_j))

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
        # Use AYO-compatible settings:
        # - aperture_radius_lod=0.85: photometric aperture size
        # - contrast_floor=1e-10: engineering stability floor
        # - use_inscribed_diameter=True: match AYO's λ/D definition
        self._yippy_coronagraph = Coronagraph(
            coro_path,
            aperture_radius_lod=0.85,
            contrast_floor=1e-10,
            use_inscribed_diameter=True,
        )

        # Store key parameters
        self._coro_IWA_lod = float(self._yippy_coronagraph.IWA.value)
        self._coro_OWA_lod = float(self._yippy_coronagraph.OWA.value)

        self.vprint(
            f"  IWA={self._coro_IWA_lod:.2f} λ/D, OWA={self._coro_OWA_lod:.2f} λ/D"
        )

        # Load empirical noise floor from AYO CSV if provided
        self._noise_floor_interp = None
        if self.noise_floor_csv is not None:
            self._load_noise_floor_csv(self.noise_floor_csv)

        return self._yippy_coronagraph

    def _load_noise_floor_csv(self, csv_path):
        """Load empirical noise floor curve from AYO coronagraph CSV.

        This loads the 'Noise floor (point source 1-sigma)' column from the
        AYO coronagraph output CSV. This value represents the engineering
        stability limit (speckle noise) that cannot be removed, as opposed to
        the theoretical diffraction limit from yippy.

        The CSV noise floor is ALREADY calculated as Contrast / PPF, so we use
        it directly without further division.
        """
        from scipy.interpolate import interp1d

        csv_path = Path(csv_path)
        if not csv_path.exists():
            self.vprint(f"  Warning: Noise floor CSV not found: {csv_path}")
            return

        self.vprint(f"  Loading empirical noise floor from: {csv_path}")
        df = pd.read_csv(csv_path)

        # The CSV has columns like 'Sep (l/D)' and
        # 'Noise floor (point source 1-sigma)'
        sep_col = "Sep (l/D)"
        nf_col = "Noise floor (point source 1-sigma)"

        if sep_col not in df.columns or nf_col not in df.columns:
            self.vprint("  Warning: Expected columns not found in CSV")
            self.vprint(f"    Available columns: {list(df.columns)}")
            return

        seps = df[sep_col].values
        noise_floor = df[nf_col].values

        # Create interpolator
        self._noise_floor_interp = interp1d(
            seps,
            noise_floor,
            kind="linear",
            bounds_error=False,
            fill_value=(noise_floor[0], noise_floor[-1]),  # Extrapolate
        )

        self.vprint(
            f"  Noise floor at 8 λ/D: {float(self._noise_floor_interp(8.0)):.2e}"
        )

    def _init_multiaperture_curves(self, coro):
        """Pre-compute throughput interpolators for aperture optimization.

        This mimics AYO's behavior of testing multiple PSF truncation ratios (aperture sizes)
        to find the optimal balance between planet signal (throughput) and background noise (omega).
        """
        if (
            hasattr(self, "_aperture_interpolators")
            and self._aperture_interpolators is not None
        ):
            return

        # Get radii list from params (AYO psf_trunc_ratio) or default
        radii = self.det_params.get("psf_trunc_ratio")
        if radii is None:
            # Check raw params (parse_input puts arrays here)
            radii = self.pyedith_raw_params.get("psf_trunc_ratio")

        # Handle scalars or missing values
        if radii is None:
            # Default optimization range matching AYO's psf_trunc_ratio array
            # 20 values from 0.05 to 1.0 in 0.05 steps
            radii = [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ]
        elif np.isscalar(radii):
            radii = [float(radii)]
        else:
            # Ensure it's a list/array of floats
            radii = [float(r) for r in radii]

        self.vprint(
            f"Initializing aperture optimization over {len(radii)} radii: {radii}"
        )

        self._aperture_interpolators = {}

        # Compute curves for each radius
        # We use yippy's _compute_performance_metrics to avoid overhead of saving/plotting
        for r in tqdm(radii, desc="Pre-computing aperture curves", leave=False):
            # AYO uses fixed circular aperture integration (no Gaussian fit)
            # This matches omega = pi * r^2
            curves = coro._compute_performance_metrics(
                aperture_radius_lod=r,
                fit_gaussian_for_core_area=False,
                oversample=2,
                compute_throughput=True,
                compute_contrast=False,  # We use Istar independent of aperture
                compute_core_area=False,  # Known pi*r^2
            )

            sep = curves["separations"]
            tp = curves["throughput"]

            self._aperture_interpolators[r] = {
                "throughput": make_interp_spline(sep, tp, k=3),
                "omega": np.pi * r**2,
            }

        # Ensure base Istar interpolator exists (independent of aperture size)
        if not hasattr(coro, "core_intensity_interp") or not hasattr(
            coro, "occ_trans_interp"
        ):
            coro.compute_all_performance_curves(aperture_radius_lod=0.7, plot=False)

        self.vprint("Aperture pre-computation complete.")

    def calc_count_rates(
        self,
        sep_lod,
        dMag,
        dist_pc,
        wavelength_um=None,
        fZ_override=None,
        aperture_radius_lod=0.7,
        sInd=None,
    ):
        """Calculate all count rate components for exposure time calculation.

        This method computes all the individual count rate contributions used in
        the AYO exposure time formula. It is designed for easy debugging and
        comparison against AYO reference values.

        Args:
            sep_lod: Separation in λ/D units (scalar or array)
            dMag: Delta magnitude (scalar or array, same shape as sep_lod)
            dist_pc: Distance to star in parsecs
            wavelength_um: Observation wavelength in microns (optional, uses det_params default)
            fZ_override: Optional zodiacal light override (1/arcsec²)
            aperture_radius_lod: Photometric aperture radius in λ/D (default 0.7)
            sInd: Star index for initialization (optional, uses 0 if not set)

        Returns:
            dict with all count rate components and intermediate values:
                - CRp: Planet count rate (electrons/s)
                - CRbs: Stellar leakage count rate
                - CRbz: Zodi background count rate
                - CRbez: Exozodi background count rate
                - CRbd: Detector noise count rate
                - CRnf: Noise floor count rate
                - CRb: Total background count rate (CRbs + CRbz + CRbez + CRbd)
                - throughput: Core throughput (T_core)
                - omega: Photometric aperture solid angle in (λ/D)²
                - Istar: Stellar intensity at separation
                - SkyTrans: Sky transmission at separation
                - noise_floor_density: Noise floor contrast at separation
                - flux_factor: Common flux factor for calculations
                - F0: Zero-magnitude flux
                - Fs_over_F0: Stellar flux ratio (10^(-0.4*mag))
                - Fp_over_Fs: Planet/star flux ratio
                - lod_arcsec: λ/D in arcseconds
                - sep_arcsec: Separation in arcseconds
                - SNR: Target signal-to-noise ratio
                - params: Dictionary of all parameters used
        """
        # Load coronagraph
        coro = self._load_yippy_coronagraph()

        # Ensure pyEDITH observatory is initialized
        if not self._observatory_initialized:
            self._init_star_for_vectorized_etc(sInd if sInd is not None else 0)

        # Get pyEDITH objects
        scene = self._base_scene
        obs = self._base_observation
        observatory = self._pyedith_observatory

        # Get observation parameters
        params = self.det_params
        if wavelength_um is None:
            wavelength_um = params.get("wavelength", 0.5)
            if hasattr(wavelength_um, "__len__"):
                wavelength_um = float(np.mean(wavelength_um))
        diameter_m = params.get("diameter", 6.0)

        # λ/D in arcsec
        lod_arcsec = (wavelength_um * 1e-6 / diameter_m) * 206265.0
        sep_arcsec = sep_lod * lod_arcsec

        # Ensure arrays
        sep_lod = np.atleast_1d(sep_lod)
        dMag = np.atleast_1d(dMag)
        shape = sep_lod.shape
        sep_flat = sep_lod.flatten()

        # Planet flux ratio from dMag
        Fp_over_Fs = 10 ** (-0.4 * dMag)

        # Get fluxes from pyEDITH scene
        F0 = float(scene.F0[0].value) if hasattr(scene, "F0") else 1e10
        Fs_over_F0 = (
            float(scene.Fs_over_F0[0].value) if hasattr(scene, "Fs_over_F0") else 1.0
        )

        # Zodiacal light
        if fZ_override is not None:
            Fzodi = (
                float(fZ_override.value)
                if hasattr(fZ_override, "value")
                else float(fZ_override)
            )
        else:
            Fzodi = (
                float(scene.Fzodi_list[0].value)
                if hasattr(scene, "Fzodi_list")
                else 1e-8
            )

        # Observatory parameters
        area_cm2 = float(observatory.telescope.Area.to(u.cm**2).value)
        total_throughput = float(observatory.total_throughput[0].value)

        if hasattr(obs, "delta_wavelength") and obs.delta_wavelength is not None:
            dlambda_nm = float(obs.delta_wavelength[0].to(u.nm).value)
        else:
            wavelength_nm = params.get("wavelength", 500) * 1e3
            if hasattr(wavelength_nm, "__len__"):
                dlambda_nm = float(wavelength_nm[-1] - wavelength_nm[0])
            else:
                dlambda_nm = float(wavelength_nm * 0.2)

        nchannels = observatory.coronagraph.nchannels
        pixscale = float(observatory.coronagraph.pixscale.value)

        # Observation parameters
        SNR = float(np.mean(obs.SNR.value)) if hasattr(obs.SNR, "value") else 7.0
        noisefloor_PPF = params.get("noisefloor_PPF", 30.0)

        # Detector params
        det_DC = params.get("det_DC", 0.0003)
        det_RN = params.get("det_RN", 0.0)
        det_tread = params.get("det_tread", 10.0)
        det_CIC = params.get("det_CIC", 0.0013)
        npix_multiplier = params.get("npix_multiplier", 1.0)
        QE = float(observatory.detector.QE[0].value)
        dQE = float(observatory.detector.dQE[0].value)

        # Flux factor common to all calculations
        flux_factor = (
            F0 * Fs_over_F0 * area_cm2 * total_throughput * dlambda_nm * nchannels
        )

        # Exozodi setup
        nexozodis = params.get("nexozodis", 3.0)
        F_exozodi = Fzodi * nexozodis
        WA_au = sep_arcsec * dist_pc
        WA_au_safe = np.maximum(WA_au, 0.01)

        # Coronagraph performance at separation
        Istar = np.maximum(coro.core_intensity_interp(sep_flat), 1e-20).reshape(shape)
        SkyTrans = coro.occ_trans_interp(sep_flat).reshape(shape)

        # Get throughput for the specified aperture
        if (
            hasattr(self, "_aperture_interpolators")
            and self._aperture_interpolators is not None
            and aperture_radius_lod in self._aperture_interpolators
        ):
            data = self._aperture_interpolators[aperture_radius_lod]
            throughput = np.clip(data["throughput"](sep_flat), 0, 1).reshape(shape)
            omega = np.full(shape, data["omega"])
        else:
            # Use yippy's default interpolators
            throughput = np.clip(coro.throughput_interp(sep_flat), 0, 1).reshape(shape)
            omega = np.full(shape, np.pi * aperture_radius_lod**2)

        # Noise floor
        if (
            hasattr(self, "_noise_floor_interp")
            and self._noise_floor_interp is not None
        ):
            noise_floor_density = np.maximum(
                self._noise_floor_interp(sep_flat), 1e-15
            ).reshape(shape)
            use_csv_noisefloor = True
        else:
            ppf = 30.0
            noise_floor_density = np.maximum(
                coro.noise_floor(sep_flat, ppf=ppf), 1e-15
            ).reshape(shape)
            use_csv_noisefloor = False

        # === Count Rate Calculations ===

        # Planet count rate: CRp = flux_factor × Fp/Fs × throughput
        CRp = flux_factor * Fp_over_Fs * throughput

        # Stellar leakage: CRbs = flux_factor × Istar × omega
        CRbs = flux_factor * Istar * omega

        # Zodi background
        omega_arcsec2 = omega * (lod_arcsec**2)
        zodi_throughput = SkyTrans * QE * dQE
        CRbz = (
            Fzodi
            * omega_arcsec2
            * area_cm2
            * zodi_throughput
            * dlambda_nm
            * nchannels
        )

        # Exozodi background
        CRbez = (
            F_exozodi
            * omega_arcsec2
            * area_cm2
            * zodi_throughput
            * dlambda_nm
            * nchannels
            / (WA_au_safe**2)
        )

        # Detector noise
        npix = npix_multiplier * omega / (pixscale**2) * nchannels
        t_photon_count = 1.0 / (6.73 * np.maximum(CRp / npix, 1e-10))
        CRbd = (det_DC + det_RN**2 / det_tread + det_CIC / t_photon_count) * npix

        # Noise floor
        if use_csv_noisefloor:
            CRnf = SNR * flux_factor * noise_floor_density * omega
        else:
            CRnf = SNR * flux_factor * (noise_floor_density / noisefloor_PPF) * omega

        # Total background
        CRb = CRbs + CRbz + CRbez + CRbd

        # Build params dict for reference
        all_params = {
            "wavelength_um": wavelength_um,
            "diameter_m": diameter_m,
            "lod_arcsec": lod_arcsec,
            "area_cm2": area_cm2,
            "total_throughput": total_throughput,
            "dlambda_nm": dlambda_nm,
            "nchannels": nchannels,
            "pixscale": pixscale,
            "SNR": SNR,
            "noisefloor_PPF": noisefloor_PPF,
            "det_DC": det_DC,
            "det_RN": det_RN,
            "det_tread": det_tread,
            "det_CIC": det_CIC,
            "npix_multiplier": npix_multiplier,
            "QE": QE,
            "dQE": dQE,
            "nexozodis": nexozodis,
            "Fzodi": Fzodi,
            "F_exozodi": F_exozodi,
            "aperture_radius_lod": aperture_radius_lod,
            "use_csv_noisefloor": use_csv_noisefloor,
        }

        return {
            # Count rates
            "CRp": CRp,
            "CRbs": CRbs,
            "CRbz": CRbz,
            "CRbez": CRbez,
            "CRbd": CRbd,
            "CRnf": CRnf,
            "CRb": CRb,
            # Intermediate values
            "throughput": throughput,
            "omega": omega,
            "omega_arcsec2": omega_arcsec2,
            "Istar": Istar,
            "SkyTrans": SkyTrans,
            "noise_floor_density": noise_floor_density,
            "npix": npix,
            # Flux values
            "flux_factor": flux_factor,
            "F0": F0,
            "Fs_over_F0": Fs_over_F0,
            "Fp_over_Fs": Fp_over_Fs,
            # Geometry
            "lod_arcsec": lod_arcsec,
            "sep_lod": sep_lod,
            "sep_arcsec": sep_arcsec,
            "dist_pc": dist_pc,
            "WA_au": WA_au,
            # Observation
            "SNR": SNR,
            "params": all_params,
        }

    def calc_inttime(self, count_rates, toverhead_multi=None, toverhead_fixed=None):
        """Calculate integration time from count rates.

        Uses the standard AYO exposure time formula:
            cp = (CRp + 2*CRb) / (CRp² - CRnf²)
            t = SNR² × cp × toverhead_multi + toverhead_fixed

        Args:
            count_rates: Dictionary from calc_count_rates()
            toverhead_multi: Overhead multiplier (optional, uses observatory default)
            toverhead_fixed: Fixed overhead in seconds (optional, uses observatory default)

        Returns:
            dict with:
                - inttime_sec: Integration time in seconds
                - inttime_days: Integration time in days
                - cp: Noise coefficient
                - numerator: CRp + 2*CRb
                - denominator: CRp² - CRnf²
        """
        CRp = count_rates["CRp"]
        CRb = count_rates["CRb"]
        CRnf = count_rates["CRnf"]
        SNR = count_rates["SNR"]

        # Get overhead values from observatory if not provided
        if toverhead_multi is None:
            toverhead_multi = float(self._pyedith_observatory.telescope.toverhead_multi)
        if toverhead_fixed is None:
            toverhead_fixed = float(
                self._pyedith_observatory.telescope.toverhead_fixed.to(u.s).value
            )

        # Exposure time calculation
        numerator = CRp + 2 * CRb
        denominator = CRp**2 - CRnf**2

        with np.errstate(invalid="ignore", divide="ignore"):
            cp = numerator / denominator
            inttime_sec = SNR**2 * cp * toverhead_multi + toverhead_fixed

            # Filter invalid values
            inttime_sec = np.where(denominator > 0, inttime_sec, np.inf)
            inttime_sec = np.where(inttime_sec > 0, inttime_sec, np.inf)

        return {
            "inttime_sec": inttime_sec,
            "inttime_days": inttime_sec / 86400.0,
            "cp": cp,
            "numerator": numerator,
            "denominator": denominator,
            "toverhead_multi": toverhead_multi,
            "toverhead_fixed": toverhead_fixed,
        }

    def _calc_inttime_vectorized(
        self, sInd, dMag_grid, WA_grid, fZ_override=None, wavelength_override=None
    ):
        """Calculate integration times using vectorized pyEDITH count rates.

        Implements AYO's dynamic aperture optimization:
        Iterates over multiple aperture sizes (psf_trunc_ratio) and selects the
        radius that minimizes exposure time for each planet position.

        Args:
            sInd: Star index in TargetList
            dMag_grid: Array of delta magnitudes [n_orbits, n_phases]
            WA_grid: Array of working angles in arcsec [n_orbits, n_phases]
            fZ_override: Optional per-star zodiacal light value (1/arcsec²) from
                        EXOSIMS valfZmin. If provided, overrides pyEDITH's
                        RA/Dec-based zodi calculation.
            wavelength_override: Optional per-star detection wavelength in microns.
                        If provided, overrides the default wavelength from det_params.

        Returns:
            inttime_grid: Array of integration times in days [n_orbits, n_phases]
        """
        # Load coronagraph and pre-compute aperture curves
        coro = self._load_yippy_coronagraph()
        self._init_multiaperture_curves(coro)

        # Ensure pyEDITH observatory is initialized
        if not self._observatory_initialized:
            self._init_star_for_vectorized_etc(sInd)

        # Get pyEDITH objects
        scene = self._base_scene
        obs = self._base_observation
        observatory = self._pyedith_observatory

        # Get observation parameters - use per-star wavelength if provided
        params = self.det_params
        if wavelength_override is not None:
            wavelength_um = float(wavelength_override)
        else:
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

        # Planet flux ratio from dMag
        Fp_over_Fs = 10 ** (-0.4 * dMag_grid)  # [n_orbits, n_phases]

        # Get fluxes from pyEDITH scene (use first wavelength for simplicity)
        F0 = float(scene.F0[0].value) if hasattr(scene, "F0") else 1e10
        Fs_over_F0 = (
            float(scene.Fs_over_F0[0].value) if hasattr(scene, "Fs_over_F0") else 1.0
        )
        # Use EXOSIMS per-star fZmin if provided (more accurate than RA/Dec-based)
        if fZ_override is not None:
            # EXOSIMS fZ is in 1/arcsec² but represents same 10^(-0.4*magZ) ratio
            Fzodi = (
                float(fZ_override.value)
                if hasattr(fZ_override, "value")
                else float(fZ_override)
            )
        else:
            Fzodi = (
                float(scene.Fzodi_list[0].value)
                if hasattr(scene, "Fzodi_list")
                else 1e-8
            )

        # Observatory parameters
        area_cm2 = float(observatory.telescope.Area.to(u.cm**2).value)
        total_throughput = float(observatory.total_throughput[0].value)
        # delta_wavelength comes from observation, not observatory
        if hasattr(obs, "delta_wavelength") and obs.delta_wavelength is not None:
            dlambda_nm = float(obs.delta_wavelength[0].to(u.nm).value)
        else:
            wavelength_nm = params.get("wavelength", 500) * 1e3  # um -> nm
            if hasattr(wavelength_nm, "__len__"):
                dlambda_nm = float(wavelength_nm[-1] - wavelength_nm[0])
            else:
                dlambda_nm = float(wavelength_nm * 0.2)
        nchannels = observatory.coronagraph.nchannels
        pixscale = float(observatory.coronagraph.pixscale.value)  # λ/D per pixel

        # Observation parameters
        SNR = float(np.mean(obs.SNR.value)) if hasattr(obs.SNR, "value") else 7.0
        noisefloor_PPF = params.get("noisefloor_PPF", 30.0)
        toverhead_multi = float(observatory.telescope.toverhead_multi)
        toverhead_fixed = float(observatory.telescope.toverhead_fixed.to(u.s).value)
        td_limit = params.get("td_limit", 10.0)  # days
        nrolls = params.get("nrolls", 1)

        # Detector params
        det_DC = params.get("det_DC", 0.0003)
        det_RN = params.get("det_RN", 0.0)
        det_tread = params.get("det_tread", 10.0)
        det_CIC = params.get("det_CIC", 0.0013)
        npix_multiplier = params.get("npix_multiplier", 1.0)
        QE = float(observatory.detector.QE[0].value)
        dQE = float(observatory.detector.dQE[0].value)

        # Flux factor common to all calculations
        flux_factor = (
            F0 * Fs_over_F0 * area_cm2 * total_throughput * dlambda_nm * nchannels
        )

        # Exozodi setup
        dist_pc = float(self.TargetList.dist[sInd].value)
        nexozodis = params.get("nexozodis", 3.0)
        F_exozodi = Fzodi * nexozodis
        WA_au = WA_grid * dist_pc
        WA_au_safe = np.maximum(WA_au, 0.01)

        # Istar (Stellar intensity profile) is constant across apertures
        # Matches AYO C code where Istar_interp is computed once per star/position
        Istar = np.maximum(coro.core_intensity_interp(sep_flat), 1e-20).reshape(shape)

        # Occulter Transmission (SkyTrans) is constant across apertures
        # Used for Zodi background calculation
        SkyTrans = coro.occ_trans_interp(sep_flat).reshape(shape)

        # Noise floor for CRnf calculation
        # If we have a CSV-based noise floor (from AYO output), use it directly.
        # Otherwise, use yippy's noise_floor method with the same ppf=30.
        ppf = 30.0  # Post-processing factor
        if (
            hasattr(self, "_noise_floor_interp")
            and self._noise_floor_interp is not None
        ):
            # Use empirical noise floor from AYO CSV (already = Contrast/PPF)
            noise_floor_density = np.maximum(
                self._noise_floor_interp(sep_flat), 1e-15
            ).reshape(shape)
            use_csv_noisefloor = True
        else:
            # Use yippy's noise_floor method (computes Contrast/PPF correctly)
            noise_floor_density = np.maximum(
                coro.noise_floor(sep_flat, ppf=ppf), 1e-15
            ).reshape(shape)
            use_csv_noisefloor = False

        # Initialize array for optimal integration times
        best_inttime_days = np.full(shape, np.inf)

        # --- Dynamic Aperture Optimization Loop ---
        # Iterate over all pre-computed apertures and find min time per pixel
        radii_list = sorted(self._aperture_interpolators.keys())

        for r in radii_list:
            data = self._aperture_interpolators[r]

            # 1. Throughput for this aperture
            throughput = np.clip(data["throughput"](sep_flat), 0, 1).reshape(shape)

            # 2. Omega (area) for this aperture
            omega = np.full(shape, data["omega"])

            # Planet count rate: CRp = flux × Fp/Fs × throughput
            CRp = flux_factor * Fp_over_Fs * throughput

            # Stellar leakage: CRbs = flux × Istar × omega
            CRbs = flux_factor * Istar * omega

            # Zodi: CRbz = Fzodi × omega_arcsec² × A × (SkyTrans × QE × dQE) × Δλ
            # AYO uses skytrans for zodi (diffuse background), not throughput (point source)
            # Reference: AYO C code line 561: CRbz = tempCRbzfactor * omega_lod[index2]
            # where tempCRbzfactor includes skytrans[index]
            omega_arcsec2 = omega * (lod_arcsec**2)
            zodi_throughput = SkyTrans * QE * dQE
            CRbz = (
                Fzodi
                * omega_arcsec2
                * area_cm2
                * zodi_throughput
                * dlambda_nm
                * nchannels
            )

            # Exozodi: scales with 1/dist²
            # AYO C code line 563: CRbez = tempCRbezfactor * omega_lod[index2]
            # where tempCRbezfactor includes skytrans[index]
            CRbez = (
                F_exozodi
                * omega_arcsec2
                * area_cm2
                * zodi_throughput
                * dlambda_nm
                * nchannels
                / (WA_au_safe**2)
            )

            # Detector noise
            npix = npix_multiplier * omega / (pixscale**2) * nchannels
            t_photon_count = 1.0 / (6.73 * np.maximum(CRp / npix, 1e-10))
            CRbd = (det_DC + det_RN**2 / det_tread + det_CIC / t_photon_count) * npix

            # Noise floor: CRnf = SNR * flux * noisefloor_density * omega
            # If using CSV, the value is ALREADY Contrast/PPF, so no PPF division.
            # If using yippy fallback, divide by PPF.
            if use_csv_noisefloor:
                CRnf = SNR * flux_factor * noise_floor_density * omega
            else:
                CRnf = (
                    SNR * flux_factor * (noise_floor_density / noisefloor_PPF) * omega
                )

            # Total background
            CRb = CRbs + CRbz + CRbez + CRbd

            # Exposure time
            numerator = CRp + 2 * CRb
            denominator = CRp**2 - CRnf**2

            with np.errstate(invalid="ignore", divide="ignore"):
                cp = numerator / denominator
                inttime_sec = SNR**2 * cp * toverhead_multi + toverhead_fixed

                # Filter invalid/infinite/negative times
                inttime_sec = np.where(denominator > 0, inttime_sec, np.inf)
                inttime_sec = np.where(inttime_sec > 0, inttime_sec, np.inf)

                # Convert to days
                current_inttime_days = inttime_sec / 86400.0

            # Debug print for first aperture (to verify reasonable values)
            if not hasattr(self, "_etc_debug_done") and r == radii_list[0]:
                self._etc_debug_done = True
                valid_mask = (sep_lod >= self._coro_IWA_lod) & (
                    sep_lod <= self._coro_OWA_lod
                )
                valid_crp = CRp[valid_mask]
                self.vprint("  ETC Debug (Dynamic Aperture):")
                self.vprint(f"    wavelength_um={wavelength_um:.4f}")
                self.vprint(f"    psf_trunc_ratio radii={radii_list}")
                self.vprint(f"    Testing aperture r={r:.2f} λ/D")
                self.vprint(f"    F0={F0:.2e}, Fs/F0={Fs_over_F0:.2e}")
                self.vprint(f"    flux_factor={flux_factor:.2e}")
                if len(valid_crp) > 0:
                    self.vprint(
                        f"    CRp range: {np.min(valid_crp):.2e} to {np.max(valid_crp):.2e}"
                    )

            # Update optimal time
            best_inttime_days = np.minimum(best_inttime_days, current_inttime_days)

        # Final filtering for OWA and Limits
        best_inttime_days = np.where(
            sep_lod <= self._coro_OWA_lod, best_inttime_days, np.inf
        )
        best_inttime_days = np.where(
            best_inttime_days <= td_limit, best_inttime_days, np.inf
        )

        if nrolls != 1:
            best_inttime_days = np.where(
                np.isfinite(best_inttime_days),
                best_inttime_days * nrolls,
                best_inttime_days,
            )

        return best_inttime_days

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

        # Get per-star zodiacal light override from EXOSIMS
        fZ_min = self.valfZmin[sInd] if hasattr(self, "valfZmin") else None

        # Get per-star detection wavelength from target_list.csv
        # Need to look up HIP for this sInd via reverse mapping
        wavelength_um = None
        if hasattr(self, "hip_to_det_wavelength") and self.hip_to_det_wavelength:
            # Reverse lookup: find HIP for this sInd
            for hip, mapped_sInd in self.star_mapping.items():
                if mapped_sInd == sInd:
                    wavelength_um = self.hip_to_det_wavelength.get(hip)
                    break

        # Calculate vectorized integration times
        inttime = self._calc_inttime_vectorized(
            sInd,
            dMag,
            WA,
            fZ_override=fZ_min,
            wavelength_override=wavelength_um,
        )

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

    def _calc_completeness_at_time(
        self, sInd, obs_time_days, int_time_days, detected_mask=None
    ):
        """Calculate completeness at a specific observation time using precomputed grid.

        This propagates all orbits to the given observation time to determine each
        orbit's current phase (via WA), then looks up the precomputed integration time
        from the phase grid. This is efficient and matches AYO's approach.

        Args:
            sInd: Star index
            obs_time_days: Observation time in days from t=0 (first visit)
            int_time_days: Available integration time in days
            detected_mask: Optional boolean array [n_orbits] of previously
                          detected planets (True = already detected, exclude)

        Returns:
            tuple: (completeness_value, detectable_mask)
                   detectable_mask is boolean array [n_orbits] of which orbits
                   were detectable at this time (for tracking)
        """
        # Get the precomputed grid (builds if not cached)
        grid = self._get_grid(sInd)
        inttime_grid = grid["inttime"]  # [n_orbits, n_phases]
        planets = grid["planets"]

        n_orbits = inttime_grid.shape[0]
        n_phases = inttime_grid.shape[1]

        # Calculate current Mean Anomaly for all orbits at obs_time_days
        # M(t) = n * (t - t0) + M0
        n_arr = np.array(planets.n)  # Mean motion [n_orbits]
        M0_arr = np.array(planets.M0)  # Initial mean anomaly [n_orbits]
        t0_arr = np.array(planets.t0)  # Epoch [n_orbits]

        M_now = n_arr * (obs_time_days - t0_arr) + M0_arr

        # Wrap to [0, 2π]
        M_now = M_now % (2 * np.pi)

        # Convert Mean Anomaly to Grid Index
        # The grid was built with M in linspace(0, 2π, n_phases)
        # So phase index = floor(M / (2π) * n_phases)
        phase_indices = np.floor(M_now / (2 * np.pi) * n_phases).astype(int)
        phase_indices = np.clip(phase_indices, 0, n_phases - 1)

        # Direct O(1) Lookup using advanced indexing (eliminates WA ambiguity!)
        row_indices = np.arange(n_orbits)
        inttime_at_phase = inttime_grid[row_indices, phase_indices]

        # Determine which orbits are detectable at this time
        detectable = inttime_at_phase <= int_time_days

        # Apply detected_mask to exclude already-found planets
        # For completeness, we want: (newly detectable orbits) / (total orbits)
        # NOT: (newly detectable) / (remaining undetected)
        if detected_mask is not None:
            # Only count orbits that are detectable AND not already detected
            newly_detectable = detectable & ~detected_mask
            n_newly_detectable = np.sum(newly_detectable)
            n_already_detected = np.sum(detected_mask)
        else:
            newly_detectable = detectable
            n_newly_detectable = np.sum(detectable)
            n_already_detected = 0

        # Debug output for first few calls
        if not hasattr(self, "_debug_call_count"):
            self._debug_call_count = 0
        self._debug_call_count += 1

        if self._debug_call_count <= 6:
            n_detectable_total = np.sum(detectable)
            # Compare with best-phase detectability
            min_inttime = np.min(inttime_grid, axis=1)
            n_best_phase_detectable = np.sum(min_inttime <= int_time_days)

            # Get per-star wavelength for this star
            wavelength_um = None
            if hasattr(self, "hip_to_det_wavelength") and self.hip_to_det_wavelength:
                for hip, mapped_sInd in self.star_mapping.items():
                    if mapped_sInd == sInd:
                        wavelength_um = self.hip_to_det_wavelength.get(hip)
                        break

            self.vprint(
                f"\n  [DEBUG] _calc_completeness_at_time call #{self._debug_call_count}:"
            )
            self.vprint(
                f"    sInd={sInd}, obs_time={obs_time_days:.1f} days, "
                f"int_time={int_time_days:.3f} days"
            )
            self.vprint(
                f"    wavelength_um={wavelength_um if wavelength_um else 'default'}"
            )
            self.vprint(f"    n_orbits={n_orbits}, n_phases={n_phases}")
            self.vprint(
                f"    Detectable at current phase: {n_detectable_total}/{n_orbits} "
                f"({100*n_detectable_total/n_orbits:.1f}%)"
            )
            self.vprint(
                f"    Detectable at best phase: {n_best_phase_detectable}/{n_orbits} "
                f"({100*n_best_phase_detectable/n_orbits:.1f}%)"
            )
            self.vprint(f"    Already detected: {n_already_detected}")
            self.vprint(f"    Newly detectable: {n_newly_detectable}")
            self.vprint(f"    eta_earth={self.eta_earth}")

        if n_orbits == 0:
            return 0.0, np.zeros(n_orbits, dtype=bool)

        # Completeness = eta_earth * (newly detectable orbits / total orbits)
        # This is the marginal completeness for this visit
        completeness = self.eta_earth * n_newly_detectable / n_orbits

        return completeness, newly_detectable

    def _mark_detected_at_time(self, sInd, detectable_mask):
        """Mark planets as detected for this visit based on time-specific observability.

        Args:
            sInd: Star index
            detectable_mask: Boolean array [n_orbits] of which planets were
                            detectable at this specific observation time
        """
        if sInd in self._detected_planets:
            self._detected_planets[sInd] |= detectable_mask
        else:
            self._detected_planets[sInd] = detectable_mask.copy()

    def _mark_detected(self, sInd, int_time):
        """Mark planets as detected for this visit (legacy grid-based method).

        Updates _detected_planets[sInd] to track which orbits have been
        detected across visits. DEPRECATED: Use _mark_detected_at_time instead.

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
        self.vprint(f"  n_orbits={self.n_orbits}, n_phases={self.n_phases}")

        for idx, row in tqdm(
            sorted_df.iterrows(), total=len(sorted_df), desc="Processing observations"
        ):
            hip = row["HIP"]
            int_time_days = row["Exp Time (days)"]
            ayo_yield = row["exoEarth candidate yield"]
            visit_num = row.get("Visit #", 1)
            visit_dt_years = row.get("Visit dt (years)", 0.0)

            # Convert visit time to days (all orbits start at t=0)
            obs_time_days = visit_dt_years * 365.25

            # Skip invalid entries
            if hip < 0 or int_time_days <= 0:
                continue

            # Match to TargetList
            if hip not in self.star_mapping:
                results["unmatched_stars"] += 1
                continue

            sInd = self.star_mapping[hip]

            # Calculate completeness using time-based method
            # This propagates orbits to the actual observation time
            detected_mask = self._detected_planets.get(sInd)
            pyedith_comp, detectable_mask = self._calc_completeness_at_time(
                sInd, obs_time_days, int_time_days, detected_mask=detected_mask
            )

            # Mark any detectable planets as detected for revisit tracking
            self._mark_detected_at_time(sInd, detectable_mask)

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
