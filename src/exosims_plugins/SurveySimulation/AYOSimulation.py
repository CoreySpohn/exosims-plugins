"""AYOSimulation - A SurveySimulation module that replicates AYO's completeness-based yield.

This module uses EXOSIMS completeness calculations to reproduce AYO's yield methodology,
enabling direct comparison to understand the ~2x yield discrepancy.

Instead of running a full mission simulation with scheduling, this module:
1. Loads AYO's pre-computed observation schedule from CSV
2. Matches AYO stars to the EXOSIMS TargetList via Hipparcos ID
3. Calculates completeness for each observation using EXOSIMS methods
4. Outputs per-star and total yield for comparison with AYO
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from tqdm import tqdm


class AYOSimulation(SurveySimulation):
    """Survey simulation that replicates AYO's completeness-based yield calculation.

    This module bypasses EXOSIMS scheduling and instead uses AYO's pre-computed
    observation schedule to calculate completeness using EXOSIMS methods.

    Args:
        ayo_observations_file (str):
            Path to the AYO observations CSV file containing columns:
            - HIP: Hipparcos ID
            - Exp Time (days): Detection exposure time
            - exoEarth candidate yield: AYO's calculated yield per visit
        **specs:
            Additional specifications passed to parent class
    """

    def __init__(
        self, ayo_observations_file=None, eta_earth=0.24, n_char_phases=10, **specs
    ):
        # Store the AYO observations file path before calling parent init
        self.ayo_observations_file = ayo_observations_file
        self.eta_earth = eta_earth  # Fraction of stars with Earth-like planets
        self.n_char_phases = (
            n_char_phases  # Number of orbital phases to sample for char time
        )

        # Call parent init
        SurveySimulation.__init__(self, **specs)

        # Store in outspec for reproducibility
        self._outspec["ayo_observations_file"] = self.ayo_observations_file
        self._outspec["eta_earth"] = self.eta_earth
        self._outspec["n_char_phases"] = self.n_char_phases

        # Get detection mode
        self.detmode = list(
            filter(
                lambda mode: mode["detectionMode"],
                self.OpticalSystem.observingModes,
            )
        )[0]

        # Get characterization mode (first non-detection mode)
        char_modes = [
            m
            for m in self.OpticalSystem.observingModes
            if not m.get("detectionMode", True)
        ]
        self.charmode = char_modes[0] if char_modes else self.detmode

        # Load and process AYO observations
        if self.ayo_observations_file is not None:
            self.ayo_df = self._load_ayo_observations()
            self.star_mapping = self._match_stars_to_targetlist()
        else:
            self.ayo_df = None
            self.star_mapping = {}

    def _load_ayo_observations(self):
        """Load AYO observations from CSV file.

        Returns:
            pandas.DataFrame: AYO observations with cleaned columns
        """
        path = Path(self.ayo_observations_file)
        if not path.exists():
            raise FileNotFoundError(f"AYO observations file not found: {path}")

        self.vprint(f"Loading AYO observations from: {path}")
        df = pd.read_csv(path)

        # Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Ensure HIP is integer (handle float HIP values like "104217.")
        df["HIP"] = df["HIP"].astype(float).astype(int)

        # Get unique stars
        unique_hips = df["HIP"].unique()
        self.vprint(
            f"Loaded {len(df)} observations for {len(unique_hips)} unique stars"
        )

        return df

    def _match_stars_to_targetlist(self):
        """Match AYO stars to EXOSIMS TargetList by Hipparcos ID.

        Uses HPIC's get_name_by_hip function to match AYO HIP numbers to
        the catalog names used in TargetList.

        Returns:
            dict: Mapping from HIP ID to EXOSIMS star index
        """
        TL = self.TargetList
        SC = TL.StarCatalog

        # Build name-to-index mapping for TargetList
        name_to_sInd = {TL.Name[i]: i for i in range(TL.nStars)}

        # Use HPIC's get_name_by_hip to match AYO HIPs to catalog names
        hip_to_sInd = {}
        ayo_hips = set(self.ayo_df["HIP"].unique())

        # Check if StarCatalog has the get_name_by_hip method (HPIC)
        if hasattr(SC, "get_name_by_hip"):
            for hip in ayo_hips:
                if hip < 0:  # Skip invalid HIPs
                    continue
                catalog_name = SC.get_name_by_hip(int(hip))
                if catalog_name is not None and catalog_name in name_to_sInd:
                    hip_to_sInd[hip] = name_to_sInd[catalog_name]
        else:
            # Fallback: try to extract HIP from star names directly
            self.vprint("StarCatalog doesn't have get_name_by_hip, using fallback")
            for sInd in range(TL.nStars):
                name = TL.Name[sInd]
                hip_num = self._extract_hip_from_name(name)
                if hip_num is not None:
                    hip_to_sInd[hip_num] = sInd

        # Report matching statistics
        matched = set(hip_to_sInd.keys())
        unmatched = ayo_hips - matched - {-1}  # Exclude -1 (invalid HIPs)

        self.vprint(f"Matched {len(matched)}/{len(ayo_hips)} AYO stars to TargetList")
        if len(unmatched) > 0 and len(unmatched) < 10:
            self.vprint(f"Unmatched HIPs: {sorted(unmatched)}")
        elif len(unmatched) >= 10:
            self.vprint(f"Unmatched HIPs (first 10): {sorted(list(unmatched))[:10]}")

        return hip_to_sInd

    def _extract_hip_from_name(self, name):
        """Extract HIP number from a star name.

        Args:
            name: Star name string like "HIP 12345" or "HIP 12345 A"

        Returns:
            int or None: HIP number if found
        """
        import re

        match = re.match(r"HIP\s*(\d+)", name)
        if match:
            return int(match.group(1))
        return None

    def run_sim(self):
        """Run the AYO-style yield calculation using dynamic completeness.

        Dynamic completeness means planets detected in earlier visits are
        removed from consideration in later visits. This matches AYO's method.

        Steps:
        1. Initialize orbix completeness infrastructure
        2. For each star, track valid_orbits mask
        3. For each observation, calculate marginal completeness (new detections)
        4. Update valid_orbits mask after each observation

        Returns:
            dict: Results containing per-star completeness and yield comparison
        """
        if self.ayo_df is None:
            raise ValueError(
                "No AYO observations file specified. "
                "Set ayo_observations_file parameter."
            )

        Comp = self.Completeness

        # Check if OrbixCompleteness - must detect by class name since
        # star_ensembles/alpha_factors are only populated after orbix_setup
        use_orbix = "OrbixCompleteness" in type(Comp).__name__

        if use_orbix:
            return self._run_sim_orbix()
        else:
            return self._run_sim_simple()

    def _run_sim_orbix(self):
        """Run simulation using orbix dynamic completeness."""
        import jax.numpy as jnp
        from orbix.kepler.shortcuts import get_grid_solver

        TL = self.TargetList
        TK = self.TimeKeeping
        Comp = self.Completeness
        ZL = self.ZodiacalLight

        # Setup orbix if needed
        if not hasattr(self, "_orbix_initialized") or not self._orbix_initialized:
            self._setup_orbix()
            self._orbix_initialized = True

        results = {
            "observations": [],
            "per_star_totals": {},
            "total_completeness": 0.0,
            "total_ayo_yield": 0.0,
            "total_exosims_char_time": 0.0,
            "total_ayo_char_time": 0.0,
            "matched_stars": 0,
            "unmatched_stars": 0,
        }

        # Track valid orbits per star (copy from star ensembles)
        # valid_orbits[sInd] = boolean mask of orbits not yet detected
        valid_orbits = {}
        for sInd in range(TL.nStars):
            if sInd in Comp.star_ensembles:
                ens = Comp.star_ensembles[sInd]
                valid_orbits[sInd] = np.copy(ens.valid_orbits)
            else:
                valid_orbits[sInd] = np.ones(Comp.Nplanets, dtype=bool)

        # Sort observations by star and visit number for proper ordering
        sorted_df = self.ayo_df.sort_values(["HIP", "Visit #"])

        self.vprint("\nCalculating dynamic completeness for AYO observations...")

        for idx, row in tqdm(
            sorted_df.iterrows(), total=len(sorted_df), desc="Processing observations"
        ):
            hip = row["HIP"]
            int_time_days = row["Exp Time (days)"]
            ayo_yield = row["exoEarth candidate yield"]
            visit_num = row.get("Visit #", 1)
            # Get actual visit time from AYO (years from mission start)
            visit_dt_years = row.get("Visit dt (years)", 0.0)

            # Skip invalid entries
            if hip < 0 or int_time_days <= 0:
                continue

            # Match to TargetList
            if hip not in self.star_mapping:
                results["unmatched_stars"] += 1
                continue

            sInd = self.star_mapping[hip]
            results["matched_stars"] += 1

            # Get the dMag0Grid for this star
            mode_hex = self.detmode.get("hex", self.detmode.get("hashname"))
            dMag0Grid = self.dMag0s[mode_hex][sInd]

            # Get alpha (angular separation) and dMag for all orbits at this time
            # Use actual observation time from AYO CSV (years from mission start)
            t_norm = visit_dt_years * 365.25  # Convert years to days

            # Find closest time index
            t_ind = np.searchsorted(Comp.comp_times, t_norm, side="right") - 1
            t_ind = np.clip(t_ind, 0, len(Comp.comp_times) - 1)

            # Get orbital positions
            alpha = Comp.s[:, t_ind] * Comp.alpha_factors[sInd]
            dMag = Comp.dMag[:, t_ind]

            # Get fZ for this observation
            fZ = self.valfZmin[sInd] if hasattr(self, "valfZmin") else ZL.fZ0
            fZ_val = fZ.to_value(self.fZ_unit) if hasattr(fZ, "to_value") else float(fZ)

            # Get kEZ (exozodi coefficient)
            if hasattr(TL, "system_fbeta") and hasattr(TL, "systemInclination"):
                kEZ = TL.system_fbeta[sInd] * (
                    1 - (np.sin(TL.systemInclination[sInd]) ** 2) / 2
                )
            else:
                kEZ = 1.0

            # Get the int_time index
            int_times = np.array(dMag0Grid.int_times)
            int_time_hr = (int_time_days * u.d).to_value(u.hr)
            int_ind = np.searchsorted(int_times, int_time_hr)
            int_ind = np.clip(int_ind, 0, len(int_times) - 1)

            # Calculate which orbits are detectable at this integration time
            # alpha_dMag_mask returns True for orbits that ARE detectable
            try:
                mask = dMag0Grid.alpha_dMag_mask(
                    alpha.reshape(-1, 1),
                    dMag.reshape(-1, 1),
                    jnp.array([fZ_val]),
                    kEZ,
                )[:, 0, int_ind]
            except Exception as e:
                self.vprint(f"Error calculating mask for HIP {hip}: {e}")
                mask = np.zeros(Comp.Nplanets, dtype=bool)

            # Calculate marginal completeness = new detections only
            # These are orbits that are detectable AND still valid (not detected before)
            new_detections = mask & valid_orbits[sInd]
            marginal_comp = np.sum(new_detections) / Comp.Nplanets

            # Calculate AYO-equivalent characterization time
            # τ_{i,c} = (η_⊕ / n_sim) × Σ t_{j,c,min}
            char_time_sum, n_det_for_char, avg_char_time = self._calc_ayo_char_time(
                sInd, new_detections, fZ_val, kEZ
            )
            # Apply AYO's formula: (η_⊕ / n_sim) × Σ t_{j,c,min}
            exosims_char_time = (self.eta_earth / Comp.Nplanets) * char_time_sum

            # Get AYO spec char time from CSV (if column exists)
            ayo_spec_char_time = row.get("Spec char time (days)", 0.0)

            # Update valid orbits mask (remove detected orbits)
            valid_orbits[sInd] = valid_orbits[sInd] & ~mask

            # Store results
            obs_result = {
                "HIP": hip,
                "sInd": sInd,
                "visit": visit_num,
                "visit_dt_years": visit_dt_years,
                "t_norm_days": t_norm,
                "int_time_days": int_time_days,
                "ayo_yield": ayo_yield,
                "exosims_completeness": marginal_comp,
                "ayo_spec_char_time": ayo_spec_char_time,
                "exosims_char_time": exosims_char_time,
                "n_detectable_orbits": n_det_for_char,
                "avg_min_char_time": avg_char_time,
                "star_name": TL.Name[sInd],
            }
            results["observations"].append(obs_result)

            # Accumulate per-star totals
            if hip not in results["per_star_totals"]:
                results["per_star_totals"][hip] = {
                    "total_exosims": 0.0,
                    "total_ayo": 0.0,
                    "total_exosims_char": 0.0,
                    "total_ayo_char": 0.0,
                    "n_visits": 0,
                    "star_name": TL.Name[sInd],
                }
            results["per_star_totals"][hip]["total_exosims"] += marginal_comp
            results["per_star_totals"][hip]["total_ayo"] += ayo_yield
            results["per_star_totals"][hip]["total_exosims_char"] += exosims_char_time
            results["per_star_totals"][hip]["total_ayo_char"] += ayo_spec_char_time
            results["per_star_totals"][hip]["n_visits"] += 1

            # Accumulate grand totals
            results["total_completeness"] += marginal_comp
            results["total_ayo_yield"] += ayo_yield
            results["total_exosims_char_time"] += exosims_char_time
            results["total_ayo_char_time"] += ayo_spec_char_time

        # Print summary
        self._print_summary(results)

        return results

    def _setup_orbix(self):
        """Setup orbix infrastructure for dynamic completeness."""
        from orbix.integrations.exosims import dMag0_grid
        from orbix.kepler.shortcuts import get_grid_solver

        OS = self.OpticalSystem
        TK = self.TimeKeeping
        Comp = self.Completeness

        # Create solver
        self.solver = get_grid_solver(
            level="scalar", jit=False, kind="bilinear", E=False, trig=True
        )

        # Set up base modes required by OrbixCompleteness.orbix_setup
        # These are normally set by OrbixScheduler but we need them for star ensembles
        self.base_det_mode = self.detmode
        # Find characterization mode (spectral mode)
        char_modes = [
            m for m in OS.observingModes if m.get("detectionMode", True) is False
        ]
        self.base_char_mode = char_modes[0] if char_modes else self.detmode

        # Setup integration times
        t0 = self.OpticalSystem.intCutoff / 100  # min int time
        tf = OS.intCutoff
        self.n_int_times = getattr(self, "n_int_times", 20)
        int_times = (
            np.logspace(
                np.log10(t0.to_value(u.hr)),
                np.log10(tf.to_value(u.hr)),
                self.n_int_times,
            )
            << u.hr
        )

        # Use fixed nEZ value
        SU = self.SimulatedUniverse
        nEZ_val = getattr(SU, "fixed_nEZ_val", 3.0)
        nEZ_range = np.array([nEZ_val])

        # Generate dMag0 grids for each mode
        self.dMag0s = {}
        for mode in OS.observingModes:
            snr_margin = getattr(self, "snr_margin_det", 1.0)
            self.dMag0s[mode["hex"]] = dMag0_grid(
                self, mode, int_times, nEZ_range, n_kEZs=3, snr_margin=snr_margin
            )

        # Setup completeness orbix
        Comp.orbix_setup(self.solver, self)

        self.vprint("Orbix setup completed for AYOSimulation")

    def _calc_ayo_char_time(self, sInd, detectable_mask, fZ, kEZ):
        """Calculate AYO-equivalent characterization time for an observation.

        Implements AYO's equation 6:
            τ_{i,c} = (η_⊕ / n_sim) × Σ t_{j,c,min}

        where t_{j,c,min} is the minimum characterization time for orbit j.

        This vectorized implementation:
        1. For each detectable orbit, finds the brightest (minimum dMag) epoch
           that is also within IWA-OWA
        2. Calculates char time at that optimal epoch using vectorized calc_intTime

        Args:
            sInd: Star index
            detectable_mask: Boolean array of shape (Nplanets,) indicating which
                orbits are detectable at this observation
            fZ: Zodiacal light value for this star
            kEZ: Exozodiacal light multiplier

        Returns:
            tuple: (char_time_sum, n_detectable, avg_char_time)
                - char_time_sum: Sum of minimum char times for detectable orbits (days)
                - n_detectable: Number of detectable orbits
                - avg_char_time: Average minimum char time (days)
        """
        TL = self.TargetList
        OS = self.OpticalSystem
        Comp = self.Completeness

        n_detectable = int(np.sum(detectable_mask))
        if n_detectable == 0:
            return 0.0, 0, 0.0

        # Get indices of detectable orbits
        det_indices = np.where(detectable_mask)[0]

        mode = self.charmode
        IWA = mode["IWA"].to(u.arcsec).value  # arcsec
        OWA = mode["OWA"].to(u.arcsec).value  # arcsec
        alpha_factor = Comp.alpha_factors[sInd]

        # Get s and dMag for detectable orbits: shape (n_detectable, n_times)
        s_det = Comp.s[det_indices, :]  # AU
        dMag_det = Comp.dMag[det_indices, :]

        # Convert s to angular separation (arcsec)
        alpha_det = s_det * alpha_factor  # shape (n_detectable, n_times)

        # Soft IWA: only check OWA limit, throughput naturally handles small separations
        # iwa_owa_mask = (alpha_det >= IWA) & (alpha_det <= OWA)  # OLD: hard IWA
        owa_mask = alpha_det <= OWA  # NEW: soft IWA - only OWA limit

        # For each orbit, find the time index with minimum dMag that is within OWA
        # Set dMag to inf where outside OWA so argmin ignores those
        dMag_masked = np.where(owa_mask, dMag_det, np.inf)

        # Find index of minimum dMag for each orbit
        best_t_indices = np.argmin(dMag_masked, axis=1)  # shape (n_detectable,)

        # Get the dMag and alpha at best time for each orbit
        best_dMag = dMag_det[np.arange(n_detectable), best_t_indices]
        best_alpha = alpha_det[np.arange(n_detectable), best_t_indices]

        # Filter out orbits where no valid time exists (all times outside IWA-OWA)
        valid_mask = np.isfinite(dMag_masked[np.arange(n_detectable), best_t_indices])
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            # All detectable orbits are outside IWA-OWA at all times
            return 0.0, n_detectable, 0.0

        # Get valid dMag and WA arrays
        valid_dMag = best_dMag[valid_mask]
        valid_WA = best_alpha[valid_mask] * u.arcsec

        # Prepare fZ and fEZ for vectorized calc_intTime
        fZ_val = fZ if not hasattr(fZ, "value") else fZ
        mode_hash = mode.get("hashname", mode.get("hex"))
        if hasattr(TL, "JEZ0") and isinstance(TL.JEZ0, dict):
            fEZ = TL.JEZ0[mode_hash][sInd] * kEZ
        else:
            fEZ = self.ZodiacalLight.fEZ0 * kEZ

        # Create arrays of sInd for vectorized call
        sInds_arr = np.full(n_valid, sInd, dtype=int)

        # Vectorized calc_intTime call
        try:
            char_int_times = OS.calc_intTime(
                TL,
                sInds_arr,
                fZ_val,
                fEZ,
                valid_dMag,
                valid_WA,
                mode,
            )
            char_times_days = char_int_times.to(u.d).value

            # Filter valid times (< intCutoff)
            intCutoff_d = OS.intCutoff.to(u.d).value
            char_times_days = np.where(
                char_times_days < intCutoff_d, char_times_days, intCutoff_d
            )
        except Exception:
            # Fallback: use intCutoff for all
            intCutoff_d = OS.intCutoff.to(u.d).value
            char_times_days = np.full(n_valid, intCutoff_d)

        # Sum of minimum char times (these are already minimums per orbit)
        char_time_sum = float(np.sum(char_times_days))
        avg_char_time = char_time_sum / n_detectable if n_detectable > 0 else 0.0

        return char_time_sum, n_detectable, avg_char_time

    def _run_sim_simple(self):
        """Fallback for non-orbix completeness - not currently supported.

        For proper dynamic completeness calculation, OrbixCompleteness is required.
        """
        raise NotImplementedError(
            "AYOSimulation requires OrbixCompleteness for dynamic completeness. "
            "Please configure your EXOSIMS spec to use "
            "'modules': {'Completeness': "
            "'exosims_plugins.Completeness.OrbixCompleteness'}"
        )

    def _print_summary(self, results):
        """Print a summary of the yield comparison."""
        self.vprint("\n" + "=" * 70)
        self.vprint("AYO vs EXOSIMS Yield Comparison Summary")
        self.vprint("=" * 70)

        n_obs = len(results["observations"])
        n_stars = len(results["per_star_totals"])

        self.vprint(f"Total observations processed: {n_obs}")
        self.vprint(f"Unique stars matched: {n_stars}")
        self.vprint(f"Unmatched observations: {results['unmatched_stars']}")

        self.vprint(f"\n{'Metric':<30} {'AYO':<15} {'EXOSIMS':<15} {'Ratio':<10}")
        self.vprint("-" * 70)

        ayo_total = results["total_ayo_yield"]
        exo_total = results["total_completeness"]
        ratio = ayo_total / exo_total if exo_total > 0 else float("inf")

        self.vprint(
            f"{'Total Yield':<30} {ayo_total:<15.4f} {exo_total:<15.4f} {ratio:.2f}x"
        )

        # Char time comparison
        ayo_char = results.get("total_ayo_char_time", 0.0)
        exo_char = results.get("total_exosims_char_time", 0.0)
        char_ratio = ayo_char / exo_char if exo_char > 0 else float("inf")

        self.vprint(
            f"{'Total Char Time (days)':<30} {ayo_char:<15.4f} {exo_char:<15.4f} {char_ratio:.2f}x"
        )

        # Calculate observation-level statistics
        obs_data = results["observations"]
        if obs_data:
            ayo_vals = np.array([o["ayo_yield"] for o in obs_data])
            exo_vals = np.array([o["exosims_completeness"] for o in obs_data])

            # Per-observation ratios (where EXOSIMS > 0)
            valid_mask = exo_vals > 0
            if np.any(valid_mask):
                ratios = ayo_vals[valid_mask] / exo_vals[valid_mask]
                self.vprint(f"\n{'Per-Observation Statistics:'}")
                self.vprint(f"  Mean ratio (AYO/EXOSIMS): {np.mean(ratios):.3f}")
                self.vprint(f"  Median ratio: {np.median(ratios):.3f}")
                self.vprint(f"  Std dev: {np.std(ratios):.3f}")
                self.vprint(f"  Min ratio: {np.min(ratios):.3f}")
                self.vprint(f"  Max ratio: {np.max(ratios):.3f}")

            # Correlation
            if len(ayo_vals) > 1:
                corr = np.corrcoef(ayo_vals, exo_vals)[0, 1]
                self.vprint(f"  Correlation coefficient: {corr:.4f}")

        # Show top 10 stars by AYO yield
        self.vprint(f"\n{'Top 10 Stars by AYO Yield:'}")
        self.vprint(
            f"{'HIP':<10} {'Name':<20} {'AYO':<10} {'EXOSIMS':<10} {'Ratio':<10}"
        )
        self.vprint("-" * 60)

        sorted_stars = sorted(
            results["per_star_totals"].items(),
            key=lambda x: x[1]["total_ayo"],
            reverse=True,
        )[:10]

        for hip, data in sorted_stars:
            ayo = data["total_ayo"]
            exo = data["total_exosims"]
            r = ayo / exo if exo > 0 else float("inf")
            name = data["star_name"][:18]
            self.vprint(f"{hip:<10} {name:<20} {ayo:<10.4f} {exo:<10.4f} {r:.2f}x")

        # Show sample observation-by-observation comparison (first 10)
        self.vprint(f"\n{'Sample Observation-by-Observation (first 10):'}")
        self.vprint(
            f"{'HIP':<10} {'Visit':<6} {'t(yr)':<8} "
            f"{'AYO':<12} {'EXOSIMS':<12} {'Ratio':<8}"
        )
        self.vprint("-" * 70)

        for obs in results["observations"][:10]:
            hip = obs["HIP"]
            visit = obs["visit"]
            t_yr = obs.get("visit_dt_years", 0.0)
            ayo = obs["ayo_yield"]
            exo = obs["exosims_completeness"]
            r = ayo / exo if exo > 0 else float("inf")
            self.vprint(
                f"{hip:<10} {visit:<6} {t_yr:<8.3f} "
                f"{ayo:<12.6f} {exo:<12.6f} {r:.2f}x"
            )

        self.vprint("=" * 70)

        # Create detailed results DataFrame for export
        results["observations_df"] = pd.DataFrame(results["observations"])

    def get_comparison_df(self, results):
        """Get a DataFrame with observation-by-observation comparison.

        Args:
            results: Results dict from run_sim()

        Returns:
            pd.DataFrame: Detailed comparison data
        """
        if "observations_df" in results:
            return results["observations_df"]
        return pd.DataFrame(results["observations"])
