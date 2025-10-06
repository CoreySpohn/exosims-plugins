"""Scheduler that uses orbix to calculate the detection probability.

Basically a full rewrite of the SurveySimulation class.
"""

import os
import time
import warnings
from bisect import insort
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional

import astropy.constants as const
import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from intervaltree import Interval, IntervalTree
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch
from orbix.integrations.exosims import dMag0_grid
from orbix.kepler.shortcuts import get_grid_solver

warnings.filterwarnings("ignore", category=UserWarning, module="erfa")


@dataclass
class Target:
    """Information on what we're pointing at."""

    kind: str  # "star", "planet", "reference star"
    sInd: int
    pInd: Optional[int] = None
    revisitNumber: Optional[int] = None
    extra: Optional[dict] = None

    @classmethod
    def star(cls, sInd):
        """Create a star target."""
        return cls("star", sInd=sInd)

    @classmethod
    def planet(cls, sInd, pInd):
        """Create a planet target."""
        return cls("planet", sInd=sInd, pInd=pInd)


@dataclass(slots=True)
class ObservationResult:
    """Output of do_detection / do_characterization.

    `data` holds the numerics, `meta` holds identifiers needed for DRM.
    """

    data: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- DRM serialization (one public entry‑point) ----------------
    def to_drm(self):
        """Convert the observation result to a DRM."""
        if self.meta.get("purpose") == "detection":
            return self._det_drm()
        elif self.meta.get("purpose") == "characterization":
            return self._char_drm()
        else:
            return {}

    # Mirroring the detection DRM format
    def _det_drm(self):
        """Convert the observation result to a DRM for detection."""
        mode = self.data["det_mode"]
        return {
            "star_ind": self.meta["star_ind"],
            "star_name": self.meta["star_name"],
            "arrival_time": self.meta["arrival_time"],
            "OB_nb": self.meta["OB_nb"],
            "ObsNum": self.meta["ObsNum"],
            "plan_inds": self.meta["plan_inds"],
            "det_time": self.data["det_time"],
            "det_status": self.data["det_status"],
            "det_SNR": self.data["det_SNR"],
            "det_fZ": self.data["det_fZ"],
            "det_JEZ": self.data.get("det_JEZ"),
            "det_dMag": self.data.get("det_dMag"),
            "det_WA": self.data.get("det_WA"),
            "det_params": self.data["det_params"],
            "det_mode": mode,
            "det_comp": self.data["det_comp"],
        }

    # Mirroring the characterization DRM format
    def _char_drm(self):
        """Convert the observation result to a DRM for characterization."""
        return {
            "star_ind": self.meta["star_ind"],
            "star_name": self.meta["star_name"],
            "arrival_time": self.meta["arrival_time"],
            "OB_nb": self.meta["OB_nb"],
            "ObsNum": self.meta["ObsNum"],
            "plan_inds": self.meta["plan_inds"],
            "char_info": self.data["char_info"],
        }


@dataclass
class ScheduleAction:
    """Works like a reservation: tells the scheduler what, when, and for how long."""

    # timing
    start: float  # absolute MJD
    duration: float  # calendar time (intTime+overheads)

    # planning information
    purpose: Literal["detection", "characterization", "wait"]
    # Target/observation information
    target: Optional[Any] = None
    mode: Optional[dict] = None
    int_time: Optional[float] = None
    # Whether we're doing a blind detection
    blind: bool = False
    comp: float = 0
    pdet: float = 0
    result: Optional[ObservationResult] = None
    # Predicted values for debugging
    predicted: Optional[dict] = None

    def __post_init__(self):
        """Fill derived attributes after dataclass sets the fields."""
        self.end = self.start + self.duration

    def __lt__(self, other):
        """Compare based on start time only."""
        return self.start < other.start

    def shift(self, dt: u.Quantity):
        """Shift the reservation by a given number of days."""
        return replace(self, start=self.start + dt)

    # check if this reservation overlaps with another
    def overlaps(self, other: "ScheduleAction"):
        """Check if this reservation overlaps with another."""
        return (self.start < other.end) and (self.end > other.start)

    # req + dt
    def __add__(self, dt: u.Quantity):
        """Allows things like `req + 3` (always in days!)."""
        return self.shift(dt)

    # for dt + req
    __radd__ = __add__


@dataclass
class Epoch:
    """Per-epoch timing information for *all* stars."""

    # "Now" e.g. TK.currentTimeNorm.to_value(u.day)
    now_norm: float

    # ndarray(nStars) of absolute start times (MJD, not including settling + det OH)
    start_times_mjd: np.ndarray

    # scalar overheads in days
    det_oh: float
    char_oh: float

    # derived values
    now_det: float = field(init=False)
    now_char: float = field(init=False)
    start_det: np.ndarray = field(init=False)
    start_char: np.ndarray = field(init=False)

    # ------------------------------------------------------------------
    def __post_init__(self):
        """Fill derived attributes after dataclass sets the fields."""
        # same length as start_times_mjd but in mission‑elapsed days

        # scalar "current time" for each mode
        self.now_det = self.now_norm + self.det_oh
        self.now_char = self.now_norm + self.char_oh

        # derived values
        self.start_det = self.start_times_mjd + self.det_oh
        self.start_char = self.start_times_mjd + self.char_oh


@dataclass
class PlanetTrack:
    """Per-planet tracking of expected orbital information."""

    sInd: int
    pInd: int
    # index into err_progression
    stage: int
    last_obs_mjd: float
    det_successes: int = 0
    char_successes: int = 0
    det_failures: int = 0
    char_failures: int = 0

    def ready_for_char(self, thresh):
        """Helper to check if the planet track is ready for characterization."""
        return self.det_successes >= thresh

    def bump_detection(self):
        """Call once each time this planet is positively detected."""
        self.det_successes += 1
        self.stage += 1

    def bump_char(self):
        """Call once each time this planet is positively characterized."""
        self.char_successes += 1
        self.stage += 1

    def bump_det_fail(self):
        """Add to detection failures."""
        self.det_failures += 1

    def bump_char_fail(self):
        """Add to characterization failures."""
        self.char_failures += 1


class OrbixScheduler(SurveySimulation):
    """A scheduler that uses orbix to calculate the detection probability."""

    def __init__(
        self,
        n_det_remove=3,
        err_progression=[0.2, 0.1, 0.025],
        followup_wait_d=15,
        followup_horizon_d=500,
        pdet_threshold=0.75,
        required_det_successes=3,
        required_char_successes=1,
        max_char_failures=3,
        max_det_failures=6,
        min_comp_div_int_time=0.5,
        min_int_time_hr=1,
        n_int_times=100,
        debug_plots=False,
        final_plot=False,
        plot_format="png",
        plot_dir=".",
        max_requeue_attempts=3,
        *args,
        **kwargs,
    ):
        """Initialize the OrbixScheduler."""
        super().__init__(*args, **kwargs)

        # Add all parameters to _outspec for proper reset functionality
        self._outspec["n_det_remove"] = n_det_remove
        self._outspec["err_progression"] = err_progression
        self._outspec["followup_wait_d"] = followup_wait_d
        self._outspec["followup_horizon_d"] = followup_horizon_d
        self._outspec["pdet_threshold"] = pdet_threshold
        self._outspec["required_det_successes"] = required_det_successes
        self._outspec["required_char_successes"] = required_char_successes
        self._outspec["max_char_failures"] = max_char_failures
        self._outspec["max_det_failures"] = max_det_failures
        self._outspec["min_comp_div_int_time"] = min_comp_div_int_time
        self._outspec["min_int_time_hr"] = min_int_time_hr
        self._outspec["n_int_times"] = n_int_times
        self._outspec["debug_plots"] = debug_plots
        self._outspec["final_plot"] = final_plot
        self._outspec["plot_format"] = plot_format
        self._outspec["plot_dir"] = plot_dir
        self._outspec["max_requeue_attempts"] = max_requeue_attempts

        self.plot_format = plot_format
        self.plot_dir = plot_dir
        self.n_det_remove = n_det_remove
        self.err_progression = err_progression
        # This is in days
        self.followup_wait_d = followup_wait_d
        self.followup_horizon_d = followup_horizon_d
        # This is the threshold to consider pdet values for
        self.required_det_successes = required_det_successes
        self.required_char_successes = required_char_successes
        self.pdet_threshold = pdet_threshold
        self.max_char_failures = max_char_failures
        self.max_det_failures = max_det_failures
        self.min_comp_div_int_time = min_comp_div_int_time
        self.min_int_time = min_int_time_hr * u.hr
        self.min_int_time_d = self.min_int_time.to_value(u.day)
        self.n_int_times = n_int_times
        self.debug_plots = debug_plots
        self.final_plot = final_plot
        self.plot_format = plot_format
        self.plot_dir = plot_dir
        self.max_requeue_attempts = max_requeue_attempts

    def initializeStorageArrays(self):
        """Initialize all storage arrays based on # of stars and targets."""
        self.DRM = []
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        TL = self.TargetList
        allModes = OS.observingModes
        num_char_modes = len(
            list(filter(lambda mode: "spec" in mode["inst"]["name"], allModes))
        )
        self.fullSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.partialSpectra = np.zeros((num_char_modes, SU.nPlans), dtype=int)
        self.propagTimes = np.zeros(self.TargetList.nStars) << u.d
        self.lastObsTimes = np.zeros(self.TargetList.nStars) << u.d
        # contains the number of times each star was visited
        self.starVisits = np.zeros(self.TargetList.nStars, dtype=int)
        self.det_starVisits = np.zeros(self.TargetList.nStars, dtype=int)
        self.char_starVisits = np.zeros(self.TargetList.nStars, dtype=int)
        self.starRevisit = np.array([])
        self.starExtended = np.array([], dtype=int)
        self.lastDetected = np.empty((self.TargetList.nStars, 4), dtype=object)

        self.ignore_stars = []
        # Number of detections by star index
        self.sInd_detcounts = np.zeros(TL.nStars, dtype=int)
        self.sInd_dettimes = {}

        # Create schedule/history observation lists
        # This format ensures that the schedule is always sorted
        self.schedule: list[ScheduleAction] = []
        self._itr = IntervalTree()
        self.history: list[ScheduleAction] = []

        # Create the planet tracks
        # Indexed by (sInd, pInd)
        self.planet_tracks = {}
        self.retired_tracks = {}

        self.to_requeue = []
        self.to_requeue_later = {}

        # Track all planets detected throughout the mission
        self._all_detected_planets = set()

        # Track planets that have completed full characterization
        self._completed_planets = set()

        # Initialize mission statistics tracking
        self._prev_mission_stats = None

        self.promotion_times = {}

        SU.orbix_planets = {}

        # Flag to track if orbix has been set up to prevent multiple setups
        # on reset_sim
        self._orbix_initialized = False

        # Retry tracking for deferred scheduling
        self.track_retry_counts = {}

        # choose observing modes selected for detection (default marked with a flag)
        self.select_default_observing_modes()

    def setup_orbix(self):
        """Create necessary Orbix objects."""
        OS, SU = self.OpticalSystem, self.SimulatedUniverse
        t0 = self.min_int_time
        tf = OS.intCutoff
        int_times = (
            np.logspace(
                np.log10(t0.to_value(u.hr)),
                np.log10(tf.to_value(u.hr)),
                self.n_int_times,
            )
            << u.hr
        )
        # Right now this is fixed to the exact value of all debris disks
        nEZ_range = np.array([SU.fixed_nEZ_val])
        self.dMag0s = {}
        self.solver = get_grid_solver(
            level="planet", jit=False, kind="bilinear", E=False, trig=True
        )
        for mode in OS.observingModes:
            self.dMag0s[mode["hex"]] = dMag0_grid(
                self, mode, int_times, nEZ_range, n_kEZs=3
            )

        # Set up orbix completeness
        self.Completeness.orbix_setup(self.solver, self)

    def run_sim(self) -> None:
        """Top-level simulation loop."""
        TK = self.TimeKeeping
        # Compute the kEZ value for each star system using their system inclination
        # and the fbeta values for the system
        self.exact_kEZs = self.TargetList.system_fbeta * (
            1 - (np.sin(self.TargetList.systemInclination) ** 2) / 2
        )

        # Initialize all the orbix things only once per instance
        if not self._orbix_initialized:
            self.setup_orbix()
            self._orbix_initialized = True

        # begin survey, and loop until mission is finished
        t0 = time.time()
        self.ObsNum = 0

        while True:
            # Next target just adds an observation to the schedule (if necessary)
            self.next_action()

            if len(self.schedule) == 0:
                # No more observations could be scheduled
                break

            # Get the next action from the schedule
            action = self.schedule.pop(0)
            _interval = Interval(
                float(action.start),
                float(action.end),
                (action.purpose, action.target),
            )
            self._itr.remove(_interval)

            # Advance to the action's start time
            if TK.currentTimeAbs.mjd < action.start:
                TK.advanceToAbsTime(
                    Time(action.start, format="mjd"), addExoplanetObsTime=False
                )

            # Check if the action would violate the mission duration
            if TK.currentTimeAbs.mjd + action.duration > TK.missionFinishAbs.mjd:
                break

            # Check if the action would violate the allowed observation time
            if (
                TK.exoplanetObsTime.to_value(u.day) + action.duration
                >= TK.allocated_time_d
            ):
                break

            # Perform the action
            if action.purpose == "detection":
                # Get the result of the detection and add it to the observation
                action.result = self.do_detection(action)
            elif action.purpose == "characterization":
                action.result = self.do_characterization(action)
            elif action.purpose == "general_astrophysics":
                # NOTE: I maintain that this is funny
                # action.result = self.do_worthless_observation(action)
                pass
            if action.result is False:
                # If the action failed to allocate time, break out of the loop
                break

            # Update time to end of the action
            if (action.end - TK.currentTimeAbs.mjd) > 1e-6:
                TK.advanceToAbsTime(
                    Time(action.end, format="mjd"), addExoplanetObsTime=False
                )

            # Respond to the action's result
            self.respond(action)
            self.log_obs(action)

        # Mission is complete
        dtsim = (time.time() - t0) * u.s
        log_end = (
            "Mission complete: no more time available.\n"
            + "Simulation duration: %s.\n" % dtsim.astype("int")
            + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
        )
        self.logger.info(log_end)
        self.vprint(log_end)

        self.calc_mission_stats()
        # Generate final schedule plot after mission completes
        if self.debug_plots or self.final_plot:
            self.plot_final_schedule()
        self.cleanup_memory()

    def respond(self, action):
        """Processes the observation result and responds accordingly.

        Is called after every observation. Modifies the schedule and history.

        Args:
            action (ScheduleAction):
                The action to respond to.
        """
        completed_track_stage = None

        # Determine the stage of the completed observation
        if action.target and action.target.kind == "planet":
            key = (action.target.sInd, action.target.pInd)
            track = self.planet_tracks.get(key) or self.retired_tracks.get(key)
            if track:
                completed_track_stage = track.stage

        if (
            action.purpose == "detection"
            and not (action.result.data["det_status"] == 1).any()
        ):
            # No detection, completeness based response
            self._process_nondetection(action)
        elif action.purpose == "detection":
            self._process_detection(action)
        elif action.purpose == "characterization":
            self._process_characterization(action)

        # After processing the current observation, check for tracks to retry
        if completed_track_stage is not None:
            self._process_requeue_later(completed_track_stage)

        # After responding to results, prune any now-redundant scheduled actions
        # (e.g., pending detections for planets that have reached detection threshold)
        self._prune_redundant_scheduled_actions()

    def _prune_redundant_scheduled_actions(self) -> None:
        """Remove scheduled observations that are no longer necessary.

        - Drop any pending detections for planets that already reached the
          detection threshold (i.e., now ready for characterization).
        - Drop any pending observations for retired/completed tracks.
        - Deduplicate multiple scheduled actions of the same purpose for the
          same planet by keeping the earliest and removing the rest.
        """
        if not self.schedule:
            return

        to_remove = []
        seen = {}
        for act in self.schedule:
            tgt = getattr(act, "target", None)
            if tgt is None or getattr(tgt, "kind", None) != "planet":
                continue

            key = (tgt.sInd, tgt.pInd)
            track = self.planet_tracks.get(key)

            # Remove any actions for retired tracks
            if key in self.retired_tracks:
                to_remove.append(act)
                continue

            # If we still have an active track, check thresholds
            if track is not None:
                # If planet is ready for characterization, drop pending detections
                if act.purpose == "detection" and track.ready_for_char(
                    self.required_det_successes
                ):
                    to_remove.append(act)
                    continue
                # If fully characterized, drop any pending characterizations
                if (
                    act.purpose == "characterization"
                    and track.char_successes >= self.required_char_successes
                ):
                    to_remove.append(act)
                    continue

            # Deduplicate same-purpose actions per planet (keep earliest only)
            dkey = (tgt.sInd, tgt.pInd, act.purpose)
            if dkey in seen:
                to_remove.append(act)
            else:
                seen[dkey] = act

        for act in to_remove:
            self._remove_action(act)

    def _process_nondetection(self, action: ScheduleAction) -> None:
        sInd = action.target.sInd
        TK, Comp = self.TimeKeeping, self.Completeness
        # There are two cases here, a blind target or a known target
        if action.target.kind == "star":
            ens = Comp.star_ensembles.get(sInd)
            # Blind target
            # Update the ensemble's mask and fraction of valid orbits
            ens.n_failures += 1
            if ens.n_failures > self.nVisitsMax:
                # give up on this star
                self.ignore_stars.append(sInd)
                return

            _dMag0Grid = self.dMag0s[action.result.data["det_mode"]["hex"]][sInd]
            # Get closest time to the observation in our pre-computed s/dMag values
            t_ind = (
                np.searchsorted(
                    Comp.comp_times,
                    TK.currentTimeNorm.to_value(u.day),
                    side="right",
                )
                - 1
            )
            alphas = Comp.s[:, t_ind].reshape(-1, 1) * Comp.alpha_factors[sInd]
            dMags = Comp.dMag[:, t_ind].reshape(-1, 1)
            fZ = jnp.array([action.result.data["det_fZ"].to_value(self.fZ_unit)])
            kEZ = self.exact_kEZs[sInd]
            mask = _dMag0Grid.alpha_dMag_mask(alphas, dMags, fZ, kEZ)

            # Get the closest int_time to the observation's int_time
            int_time_ind = np.searchsorted(_dMag0Grid.int_times, action.int_time)

            # Get the mask for the used int_time
            mask = mask[:, 0, int_time_ind]

            # Update the ensemble's mask and fraction of valid orbits
            ens.valid_orbits &= ~mask
            ens.f_valid = ens.valid_orbits.sum() / ens.n_orbits

            # Forecast the dynamic completeness values for 1000 days
            dtimes = TK.currentTimeNorm.to_value(u.day) + jnp.linspace(
                0, 1000, 100, endpoint=False
            )
            fZMap = self.ZodiacalLight.fZMap[self.base_det_mode["syst"]["name"]]
            fZ_inds = (
                np.searchsorted(
                    self.koTimes_mjd, dtimes + TK.missionStart.mjd, side="right"
                )
                - 1
            )
            fZ = jnp.float32(fZMap[sInd, fZ_inds])
            comp_t_inds = (
                np.searchsorted(
                    Comp.comp_times,
                    dtimes,
                    side="right",
                )
                - 1
            )
            alphas = Comp.s[:, comp_t_inds] * Comp.alpha_factors[sInd]
            dMags = Comp.dMag[:, comp_t_inds]
            # This calculates the dynamic completeness as a function of
            # integration time
            dyn_comp_div_intTime = np.array(
                _dMag0Grid.dyn_comp_vec(
                    self.solver, alphas, dMags, fZ, kEZ, ens.valid_orbits
                )
            )
            # For each obs time, get the maximum dynamic completeness value
            # and the integration time that gives it
            max_dyn_comp_inds = np.argmax(dyn_comp_div_intTime, axis=1)
            ens.best_int_times = np.array(_dMag0Grid.int_times)[max_dyn_comp_inds]
            ens.best_comp_div_intTime = np.array(dyn_comp_div_intTime)[
                np.arange(dyn_comp_div_intTime.shape[0]), max_dyn_comp_inds
            ]
            ens.times = np.array(dtimes)
            # Get the first time where the comp/int_time is > 0.5 of the max
            # This is just a simple way to avoid making a bunch of immediate
            # revisit observations
            max_remaining = ens.best_comp_div_intTime.max()
            remaining_times = ens.times[
                ens.best_comp_div_intTime > 0.75 * max_remaining
            ]
            if remaining_times.size:
                first_available_time = remaining_times[0]
                ens.next_available_time = first_available_time
            else:
                ens.next_available_time = np.inf
        else:
            # Known target, update the planet tracks and queue follow-up
            for track in list(self._tracks_for_star(sInd)):
                track.bump_det_fail()
                track.last_obs_mjd = TK.currentTimeAbs.mjd

                if track.det_failures > self.max_det_failures:
                    # give up on this planet
                    del self.planet_tracks[(track.sInd, track.pInd)]
                    self.logger.info(
                        f"Dropping planet ({track.sInd},{track.pInd})"
                        f" after {track.det_failures} failed re-detections."
                    )
                    continue

                # Otherwise update the orbit and schedule another attempt
                self._update_orbit(track)
                self._queue_followup(track, action)

    def _process_detection(self, action: ScheduleAction) -> None:
        """Create tracks for new detections and queue their follow-up."""
        sInd = action.target.sInd
        detected = action.result.data["det_status"]

        plan_inds = action.result.meta["plan_inds"][detected == 1]

        _time = self.TimeKeeping.currentTimeAbs.mjd
        for pInd in plan_inds:
            key = (sInd, int(pInd))
            if key in self.retired_tracks:
                continue
            # if pInd == 109:
            #     # Get the history of this star
            #     all_obs = [obs for obs in self.history if obs.target.sInd == sInd]
            #     all_obs.append(action)
            #     if len(all_obs) >= 2:
            #         breakpoint()

            # Add to the set of all detected planets
            self._all_detected_planets.add(key)

            track = self.planet_tracks.get(key)
            if track is None:
                track = self.planet_tracks[key] = PlanetTrack(
                    sInd=sInd, pInd=int(pInd), stage=0, last_obs_mjd=_time
                )

            track.last_obs_mjd = _time
            # Improve orbital estimate with error progression
            self._update_orbit(track)
            track.bump_detection()
            if self._check_characterizable(track):
                self._queue_followup(track, action)
            else:
                # Planet will never be characterizable
                # :(
                self._retire_track(track)

    def _check_characterizable(self, track):
        """Check if the planet is characterizable at any time in the future."""
        # Check if this planet would be characterizable using char mode
        char_dMag0Grid = self.dMag0s[self.base_char_mode["hex"]][track.sInd]

        # Check if characterizable at any integration time
        planets = self.SimulatedUniverse.orbix_planets[track.pInd]

        # Times for the remaining mission
        times = jnp.arange(
            self.TimeKeeping.currentTimeNorm.to_value(u.day),
            self.TimeKeeping.missionLife_d,
            1,
        )
        time_inds = (
            np.searchsorted(
                self.koTimes_mjd,
                times + self.TimeKeeping.missionStart.mjd,
                side="right",
            )
            - 1
        )
        fZ = self.ZodiacalLight.fZMap[self.base_char_mode["syst"]["name"]][
            track.sInd, time_inds
        ]
        kEZ = self.exact_kEZs[track.sInd]
        char_pdet = char_dMag0Grid.pdet_planets(self.solver, times, planets, fZ, kEZ)
        is_characterizable = jnp.any(char_pdet > 0)
        return is_characterizable

    def _process_characterization(self, action: ScheduleAction) -> None:
        """Advance or retry tracks depending on success."""
        sInd = action.target.sInd
        # Was this characterization successful?
        # Last char_info entry has 'char_status' array in the same order
        char_status = action.result.data["char_info"][-1]["char_status"]
        plan_inds = action.result.meta["plan_inds"]

        for pInd, status in zip(plan_inds, char_status):
            key = (sInd, int(pInd))
            if key not in self.planet_tracks:
                continue

            track = self.planet_tracks[key]
            track.last_obs_mjd = self.TimeKeeping.currentTimeAbs.mjd

            self._update_orbit(track)
            if status == 1:
                # Successful characterization
                track.bump_char()
                # track.char_failures = 0

                self.logger.info(
                    "CHAR DEBUG: Successful characterization for planet "
                    f"({sInd},{pInd}). "
                    "Successes now "
                    f"{track.char_successes}/{self.required_char_successes}"
                )

                if track.char_successes < self.required_char_successes:
                    self._update_orbit(track)
                    self._queue_followup(track, action)
                else:
                    # Planet is fully characterized
                    if not hasattr(self, "_completed_planets"):
                        self._completed_planets = set()

                    # Add detailed logging for characterization completion
                    self.logger.info(
                        f"CHAR DEBUG: Planet ({sInd},{pInd}) has been fully "
                        "characterized with "
                        f"{track.char_successes}/{self.required_char_successes} "
                        "successful characterizations. Removing from active tracking."
                    )

                    # Add to completed set
                    self._completed_planets.add(key)

                    # Remove from planet_tracks to prevent further scheduling
                    self._retire_track(track)

                    # Check if this was the last planet for this star
                    remaining_planets_for_star = [
                        t for t in self.planet_tracks.values() if t.sInd == sInd
                    ]

                    # Only add to ignore_stars if this was the last planet for this star
                    # AND if we want to ignore stars after characterization
                    if not remaining_planets_for_star and hasattr(self, "ignore_stars"):
                        self.logger.info(
                            f"CHAR DEBUG: No more planets to track for star {sInd}. "
                            "Adding to ignore list."
                        )
                        if sInd not in self.ignore_stars:
                            self.ignore_stars.append(sInd)
            else:
                # Failed characterization
                track.bump_char_fail()

                if track.char_failures < self.max_char_failures:
                    self._queue_followup(track, action)
                else:
                    # Remove the planet from tracking after too many failures
                    self._retire_track(track)

    def _update_orbit(self, track):
        """Update the orbit of the given planet track."""
        stage = np.clip(track.stage, 0, len(self.err_progression) - 1)
        self.SimulatedUniverse.create_orbix_planets(
            track.pInd,
            self.TimeKeeping,
            self,
            err=self.err_progression[stage],
            norb=500,
        )

    def _queue_followup(self, track, prev_action):
        """Create a follow up observation for the given planet track."""
        # Use the planet's generated orbix planets to forecast the best time to observe
        _do_char = track.ready_for_char(self.required_det_successes)
        if _do_char:
            purpose = "characterization"
            mode = self.base_char_mode
        else:
            purpose = "detection"
            mode = self.base_det_mode

        # Check if there's already a scheduled follow-up for this planet and cancel it
        # Also cancel any already-scheduled detections if we just reached the
        # detection threshold and are switching to characterization.
        _to_remove = []
        for _action in self.schedule:
            if (
                _action.target is not None
                and getattr(_action.target, "kind", None) == "planet"
                and _action.target.sInd == track.sInd
                and _action.target.pInd == track.pInd
            ):
                # Remove any same-purpose duplicates
                if _action.purpose == purpose:
                    _to_remove.append(_action)
                # If promoting to characterization, remove pending detections
                if _do_char and _action.purpose == "detection":
                    _to_remove.append(_action)
        for _action in _to_remove:
            self._remove_action(_action)

        # Define a sequence of progressively lower thresholds to try
        threshold_multipliers = [1.0, 0.9, 0.5, 0.25]
        if _do_char:
            # More aggressive thresholds for characterization
            horizon_multipliers = [10, 10, 10, 10]
        else:
            # Standard thresholds for detection
            horizon_multipliers = [1, 3, 5, 10]
        t_start, int_time, max_pdet = None, None, 0
        predicted_values = None

        # Try each threshold until we find a valid observation or exhaust all options
        for threshold_multiplier, horizon_multiplier in zip(
            threshold_multipliers, horizon_multipliers
        ):
            current_threshold = self.pdet_threshold * threshold_multiplier
            current_horizon = self.followup_horizon_d * horizon_multiplier
            t_start, int_time, max_pdet, pdet_val, predicted_values = (
                self._calc_optimal_followup(
                    track, current_threshold, mode, current_horizon
                )
            )
            if t_start is not None:
                # Found a valid observation, break out of the loop
                break

        # If all thresholds failed, retire the planet
        if t_start is None:
            self._queue_for_later_retry(track)
            return

        oh = (mode["syst"]["ohTime"] + self.Observatory.settlingTime).to_value(u.day)

        action = ScheduleAction(
            start=t_start,
            duration=int_time + oh,
            purpose=purpose,
            target=Target.planet(track.sInd, track.pInd),
            mode=mode,
            int_time=int_time,
            blind=False,
            pdet=float(pdet_val),
            predicted=predicted_values,
        )
        self._add_action(action)
        while self.to_requeue:
            act = self.to_requeue.pop(0)
            if act.target and act.target.kind == "planet":
                tr = self.planet_tracks.get((act.target.sInd, act.target.pInd))
                if tr is not None:
                    self._queue_followup(tr, act)

    def _calc_optimal_followup(self, track, threshold, mode, horizon):
        """Calculate the optimal follow-up time for the given planet track."""
        is_char = mode == self.base_char_mode

        TK, ZL, TL = self.TimeKeeping, self.ZodiacalLight, self.TargetList
        _dMag0Grid = self.dMag0s[mode["hex"]][track.sInd]
        planets = self.SimulatedUniverse.orbix_planets[track.pInd]
        # Get the best time to observe
        current_time = TK.currentTimeNorm.to_value(u.day)
        max_end_time = TK.missionLife_d - _dMag0Grid.int_times[0]
        t0 = current_time + self.followup_wait_d
        tf = np.min([t0 + horizon, max_end_time])
        prop_times = jnp.linspace(t0, tf, 250, endpoint=False)
        prop_times_mjd = prop_times + TK.missionStart.mjd
        # Get the fZ values from fZMap
        # fZ_inds = np.searchsorted(self.koTimes_mjd, prop_times_mjd, side="right") - 1
        # fZMap = ZL.fZMap[self.base_det_mode["syst"]["name"]]
        # fZ = fZMap[track.sInd, fZ_inds]
        fZ = ZL.fZ(
            self.Observatory,
            TL,
            np.array([track.sInd]),
            Time(prop_times_mjd, format="mjd"),
            mode,
        ).to_value(self.fZ_unit)
        if len(fZ) == 1:
            # This edge case occurs when the start/end times are essentially the same
            # and fZ decides there's only one unique fZ value to calculate
            # and return. But we need the array sizes to match
            fZ = np.repeat(fZ, len(prop_times))
        kEZ = self.exact_kEZs[track.sInd]
        # Shape is (n_times, n_int_times)
        _int_times = _dMag0Grid.int_times
        _pdet = np.array(
            _dMag0Grid.pdet_planets(
                self.solver,
                prop_times,
                planets,
                jnp.array(fZ),
                jnp.array(kEZ),
            )
        )
        max_pdet = np.max(_pdet)
        # If max_pdet is below threshold, fail early
        if max_pdet < threshold:
            if is_char:
                # Create debug plot for failed scheduling attempt
                if hasattr(self, "debug_plots") and self.debug_plots:
                    self._plot_scheduling_attempt(
                        track,
                        prop_times,
                        _int_times,
                        _pdet,
                        threshold,
                        mode,
                        ko_status=None,
                        schedule_conflicts=None,
                        time_limits=None,
                        mission_limits=None,
                        result="threshold_failure",
                    )

            return None, None, max_pdet, None, None

        pdet = _pdet.copy()
        pdet[pdet < threshold] = 0

        # Keepout filter
        overhead_days = (
            mode["syst"]["ohTime"] + self.Observatory.settlingTime
        ).to_value(u.day)
        ko_mask = np.zeros((len(prop_times_mjd), len(_int_times)), dtype=bool)
        mission_finish_mjd = (
            self.TimeKeeping.missionFinishAbs.mjd
        )  # Get mission end for boundary

        # Iterate over each candidate follow-up start time generated for this planet
        for i, obs_start_mjd_val in enumerate(prop_times_mjd):
            # For this obs_start_mjd_val, find which integration times are observable
            ko_mask[i, :] = self._get_observable_int_time_mask_for_start_time(
                sInd=track.sInd,
                mode_name=mode["syst"]["name"],
                obs_start_mjd=obs_start_mjd_val,
                sorted_int_times_days=_int_times,
                overhead_days=overhead_days,
                mission_finish_mjd=mission_finish_mjd,
            )

        # After the loop, ko_mask is now accurately populated for the
        # specific prop_times_mjd
        pdet[~ko_mask] = 0.0

        # Mask observations that conflict with the existing schedule
        mission_start_mjd = TK.missionStart.mjd
        oh = (mode["syst"]["ohTime"] + self.Observatory.settlingTime).to_value(u.day)

        # build candidate start / end grids  (n_t, n_int)
        # absolute MJD
        cand_starts = prop_times[:, None] + mission_start_mjd
        cand_ends = cand_starts + _int_times[None, :] + oh

        # Get scheduled blocks and check for conflicts
        sched_conflicts_mask = np.zeros_like(pdet, dtype=bool)
        if self.schedule:
            sched_starts = np.fromiter(
                (a.start for a in self.schedule if a.start >= TK.currentTimeAbs.mjd),
                dtype=float,
            )
            sched_ends = np.fromiter(
                (a.end for a in self.schedule if a.start >= TK.currentTimeAbs.mjd),
                dtype=float,
            )

            if sched_starts.size:
                # broadcast and compare
                overlaps = (
                    (cand_starts[..., None] < sched_ends)
                    & (cand_ends[..., None] > sched_starts)
                ).any(axis=2)
                sched_conflicts_mask = overlaps

                # collapse along the scheduled action axis
                pdet[overlaps] = 0.0

        # Mask out observations that would use too much integration time
        available = self._available_obs_time()
        fits_time = _int_times <= available
        fits_time_mask = np.tile(fits_time, (pdet.shape[0], 1))

        pdet[:, ~fits_time] = 0

        # Mask out observations that would end after the mission ends
        mission_life = TK.missionLife.to_value(u.d)
        fits_mission = (prop_times[:, None] + _int_times[None, :] + oh) < mission_life

        pdet[~fits_mission] = 0

        # Check that any pdet is above threshold
        if np.all(pdet == 0):
            if is_char:
                # Create debug plot for failed scheduling attempt
                if hasattr(self, "debug_plots") and self.debug_plots:
                    self._plot_scheduling_attempt(
                        track,
                        prop_times,
                        _int_times,
                        _pdet,
                        threshold,
                        mode,
                        ko_status=ko_mask,
                        schedule_conflicts=sched_conflicts_mask,
                        time_limits=fits_time_mask,
                        mission_limits=fits_mission,
                        result="constraints_failure",
                    )

            # Second pass:  look for potentially good windows even if
            # they collide with the current schedule, then see whether
            # we can bump the conflicting blocks.
            pdet_wo_sched = _pdet.copy()
            # Threshold filter
            pdet_wo_sched[pdet_wo_sched < threshold] = 0
            # Keepout filter
            pdet_wo_sched[~ko_mask] = 0
            # still respect mission end and int-time limits
            pdet_wo_sched[~fits_mission] = 0
            pdet_wo_sched[:, ~fits_time] = 0

            if np.all(pdet_wo_sched == 0):
                # No valid windows found even without overlaps
                return None, None, max_pdet, None, None

            # Pick the highest pdet/intTime window
            pdet_div_int = pdet_wo_sched / _int_times
            row, col = jnp.unravel_index(jnp.argmax(pdet_div_int), pdet_div_int.shape)
            cand_start = cand_starts[row, col]
            cand_end = cand_ends[row, col]
            pdet_val = pdet_wo_sched[row, col]

            # Calculate predicted values at the selected time
            selected_time_norm = prop_times[row]
            predicted_values = self._calculate_predicted_values(
                track, selected_time_norm, fZ[row], mode
            )

            # Is everyone we collide with lower priority?
            ok = self._bump_lower_priority_blocks(cand_start, cand_end, track)
            if ok:
                # put its planet/star back in the queue
                if is_char:
                    # Create debug plot for successful bump scheduling
                    if hasattr(self, "debug_plots") and self.debug_plots:
                        # Create an empty schedule conflict mask
                        empty_sched_conflicts_mask = np.zeros_like(sched_conflicts_mask)

                        self._plot_scheduling_attempt(
                            track,
                            prop_times,
                            _int_times,
                            _pdet,
                            threshold,
                            mode,
                            ko_status=ko_mask,
                            schedule_conflicts=empty_sched_conflicts_mask,
                            time_limits=fits_time_mask,
                            mission_limits=fits_mission,
                            result="bump_success",
                            selected_time=prop_times[row],
                            selected_int=_int_times[col],
                        )

                return cand_start, _int_times[col], max_pdet, pdet_val, predicted_values

            # still no luck
            if is_char:
                # Create debug plot for failed bump attempt
                if hasattr(self, "debug_plots") and self.debug_plots:
                    self._plot_scheduling_attempt(
                        track,
                        prop_times,
                        _int_times,
                        _pdet,
                        threshold,
                        mode,
                        ko_status=ko_mask,
                        schedule_conflicts=sched_conflicts_mask,
                        time_limits=fits_time_mask,
                        mission_limits=fits_mission,
                        result="bump_failure",
                        selected_time=prop_times[row],
                        selected_int=_int_times[col],
                    )

            return None, None, max_pdet, None, None

        pdet_div_int_time = pdet / _int_times

        # Add a penalty for the time until the observation to prioritize
        # fast follow-ups
        time_penalty = (prop_times - current_time) / 100000000
        pdet_div_int_time = pdet_div_int_time - time_penalty[:, None]

        # Get the best time/int_time
        row, col = jnp.unravel_index(
            jnp.argmax(pdet_div_int_time), pdet_div_int_time.shape
        )
        # If we've failed characterization(s) of this planet then we increase
        # the integration time by 2*n_failures rows in the pdet_div_int_time
        # array (if it's an acceptable follow-up)
        if is_char and track.char_failures > 0:
            # Check that we can increase the integration time by 2*n_failures rows
            backdowns = np.arange(1, 2 * track.char_failures + 1)
            for backdown in reversed(backdowns):
                valid_int_time = col + backdown < pdet_div_int_time.shape[1]
                valid_observation = pdet_div_int_time[row, col + backdown] > 0
                if valid_int_time and valid_observation:
                    col += backdown
                    break

        t_start = prop_times[row] + TK.missionStart.mjd
        int_time = _int_times[col]
        pdet_val = pdet_div_int_time[row, col] * int_time

        # Calculate predicted values at the selected time
        selected_time_norm = prop_times[row]
        predicted_values = self._calculate_predicted_values(
            track, selected_time_norm, fZ[row], mode
        )

        if is_char:
            # Create debug plot for successful scheduling
            if hasattr(self, "debug_plots") and self.debug_plots:
                self._plot_scheduling_attempt(
                    track,
                    prop_times,
                    _int_times,
                    _pdet,
                    threshold,
                    mode,
                    ko_status=ko_mask,
                    schedule_conflicts=sched_conflicts_mask,
                    time_limits=fits_time_mask,
                    mission_limits=fits_mission,
                    result="success",
                    selected_time=prop_times[row],
                    selected_int=int_time,
                )

        return t_start, int_time, max_pdet, pdet_val, predicted_values

    def _calculate_predicted_values(self, track, time_norm, fZ_val, mode):
        """Calculate predicted observation values at a given time for debugging.

        Args:
            track (PlanetTrack): The planet track
            time_norm (float): Time in mission-elapsed days
            fZ_val (float): Zodiacal light brightness at this time
            mode (dict): Observation mode

        Returns:
            dict: Dictionary with predicted fZ, WA, d, and dMag values
        """
        planets = self.SimulatedUniverse.orbix_planets[track.pInd]

        # Get predicted orbital position and separation/magnitude
        jax_time = jnp.array([time_norm])

        # Get position in AU
        pos_AU = planets.j_prop_AU(self.solver, jax_time)  # Shape: (n_orbits, 3)

        # Get alpha (separation) and dMag
        alphas, dMags = planets.j_alpha_dMag(self.solver, jax_time)
        alphas_rad = (alphas * u.arcsec).to_value(u.rad)

        # Calculate distance (magnitude of position vector)
        distances_AU = jnp.sqrt(jnp.sum(pos_AU**2, axis=1))

        # Take the median values across all orbit realizations for the prediction
        predicted_WA = float(jnp.median(alphas))  # arcsec
        predicted_d = float(jnp.median(distances_AU))  # AU
        predicted_separation = float(
            jnp.median(alphas_rad) * self.TargetList.dist[track.sInd].to_value(u.AU)
        )  # AU
        predicted_dMag = float(jnp.median(dMags))
        predicted_fZ = float(fZ_val)  # Already a scalar

        # Calculate predicted JEZ using the same method as actual observations
        # Get the base JEZ for this star and mode
        JEZ0 = self.TargetList.JEZ0[mode["hex"]][track.sInd]

        # Apply the 1/r^2 scaling using predicted orbital distance
        # This matches how SimulatedUniverse.scale_JEZ works
        predicted_JEZ = 3 * JEZ0 * (1 / predicted_separation) ** 2

        return {
            "predicted_fZ": predicted_fZ,
            "predicted_WA": predicted_WA,
            "predicted_d": predicted_d,
            "predicted_dMag": predicted_dMag,
            "predicted_JEZ": predicted_JEZ,
            "time_norm": float(time_norm),
        }

    def _bump_lower_priority_blocks(self, start: float, end: float, track: PlanetTrack):
        """Try to clear the time window [start, end).

        Return True iff  **all** overlapping blocks had lower priority
        and were successfully removed / rescheduled.
        """
        overlaps = list(self._itr.overlap(start, end))
        if not overlaps:
            return True  # nothing to bump

        to_requeue: list[ScheduleAction] = []
        for iv in overlaps:
            purpose, tgt = iv.data
            act = next(
                a for a in self.schedule if a.start == iv.begin and a.end == iv.end
            )

            key = (act.target.sInd, act.target.pInd)
            act_track = self.planet_tracks.get(key)
            if act_track is None:
                # Check whether the planet has been retired
                act_track = self.retired_tracks.get(key)
                to_requeue.append(act)
                continue
            if act_track.stage >= track.stage:
                # This slot is already occupied by something at least as
                # important, don't bump it
                return False

            to_requeue.append(act)

        for act in to_requeue:
            self.to_requeue.append(act)
            self._remove_action(act)

        return True

    def _tracks_for_star(self, sInd: int):
        """Iterator over all PlanetTracks belonging to one star."""
        return (t for t in self.planet_tracks.values() if t.sInd == sInd)

    def _star_ready_for_char(self, sInd: int) -> bool:
        """True if any of the star's planet's is at the characterization threshold."""
        return any(
            t.ready_for_char(self.required_det_successes)
            for t in self._tracks_for_star(sInd)
        )

    def _add_action(self, action):
        """Helper to add an action to the schedule, I keep forgetting the syntax."""
        _interval = Interval(
            float(action.start), float(action.end), (action.purpose, action.target)
        )

        self._itr.add(_interval)
        insort(self.schedule, action)

    def _remove_action(self, action):
        _interval = Interval(
            float(action.start), float(action.end), (action.purpose, action.target)
        )
        self._itr.remove(_interval)
        self.schedule.remove(action)

    def _retire_track(self, track):
        key = (track.sInd, track.pInd)
        self.retired_tracks[key] = track
        del self.planet_tracks[key]
        self.ignore_stars.append(track.sInd)

        # Clean up retry tracking
        if key in self.track_retry_counts:
            del self.track_retry_counts[key]

        # Remove from any requeue_later lists
        for stage_list in self.to_requeue_later.values():
            if track in stage_list:
                stage_list.remove(track)

        for action in self.schedule:
            # Check the schedule and remove any lingering actions
            if action.target.sInd == track.sInd and action.target.pInd == track.pInd:
                self._remove_action(action)

        # Clean up orbix planets for this retired track to save memory
        if (
            hasattr(self.SimulatedUniverse, "orbix_planets")
            and track.pInd in self.SimulatedUniverse.orbix_planets
        ):
            del self.SimulatedUniverse.orbix_planets[track.pInd]

    def next_action(self):
        """Finds index of next target star and calculates its integration time.

        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.

        Args:
            old_sInd (integer):
                Index of the previous target star
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                DRM (dict):
                    Design Reference Mission, contains the results of one complete
                    observation (detection and characterization)
                sInd (integer):
                    Index of next target star. Defaults to None.
                intTime (astropy Quantity):
                    Selected star integration time for detection in units of day.
                    Defaults to None.
                waitTime (astropy Quantity):
                    a strategically advantageous amount of time to wait in the case
                    of an occulter for slew times

        """
        TL = self.TargetList

        # If there is an observation in less than 12 hours, don't schedule any
        # more observations
        if len(self.schedule) > 0:
            if self.schedule[0].start - self.TimeKeeping.currentTimeAbs.mjd < 0.5:
                return None

        # look for available targets
        slewTimes = np.zeros(TL.nStars) << u.d

        _epoch = self.init_epoch(self.base_det_mode, self.base_char_mode)
        blind_sInds = self.get_available_targets(_epoch, self.base_det_mode)
        blind_sInds = self.pre_inttime_filter(blind_sInds, _epoch, self.base_det_mode)

        blind_intTimes = self.get_blind_inttimes(
            blind_sInds, _epoch, self.base_det_mode
        )

        blind_mask = self.post_inttime_filter(
            _epoch, blind_sInds, blind_intTimes, self.base_det_mode
        )

        # Apply the integration time masks
        blind_sInds = blind_sInds[blind_mask]
        blind_intTimes = blind_intTimes[blind_mask]

        action = self.select_target(blind_sInds, blind_intTimes, slewTimes)
        if action is not None:
            self._add_action(action)

    def init_epoch(self, det_mode, char_mode):
        """Create epoch object for next_target."""
        TK, Obs, TL = self.TimeKeeping, self.Observatory, self.TargetList
        # Create an array of the current time in mission elapsed days
        start_times_mjd = np.full(TL.nStars, TK.currentTimeAbs.mjd)
        # Observatory and coronagraph overhead times
        det_oh_time = det_mode["syst"]["ohTime"] + Obs.settlingTime
        char_oh_time = char_mode["syst"]["ohTime"] + Obs.settlingTime

        return Epoch(
            now_norm=TK.currentTimeNorm.to_value(u.day),
            start_times_mjd=start_times_mjd,
            det_oh=det_oh_time.to_value(u.day),
            char_oh=char_oh_time.to_value(u.day),
        )

    def get_available_targets(self, _epoch, det_mode):
        """Get the blind detection target indices that are not in keepout."""
        # intTimeFilterInds represents stars with integration times that are
        # - positive
        # - less than the intCutoff
        # _sInds = np.arange(self.TargetList.nStars)
        _sInds = self.Completeness.characterizable_sInds
        sInds = np.intersect1d(self.intTimeFilterInds, _sInds).astype(int)
        # Ignore stars that have been fully characterized
        sInds = np.setdiff1d(sInds, self.ignore_stars)

        # Assume observable initially
        observable_mask = np.ones(len(sInds), dtype=bool)
        for i, sInd in enumerate(sInds):
            # Use the MJD time associated with this star from the _epoch object.
            # Original code implied using _epoch.start_times_mjd (raw MJD before OH).
            check_time_mjd = _epoch.start_times_mjd[sInd]

            # Check the start time
            if not self.is_observable(
                sInd, det_mode["syst"]["name"], start_mjd=check_time_mjd
            ):
                observable_mask[i] = False
        # Get the indices of blind detection targets that are not in keepout
        # at the start of the observing window
        sInds = sInds[observable_mask]

        return sInds

    def pre_inttime_filter(self, sInds, _epoch, det_mode):
        """Detection filters that can be applied before knowing intTimes."""
        # Don't do a blind detection on a star with a PlanetTrack object already
        tracked_sInds = np.array([t.sInd for t in self.planet_tracks.values()])
        sInds = np.setdiff1d(sInds, tracked_sInds)

        # Remove stars with too many visits and no detections
        det_mask = ~(
            (self.det_starVisits[sInds] > self.n_det_remove)
            & (self.sInd_detcounts[sInds] == 0)
        )
        sInds = sInds[det_mask]
        return sInds

    def get_blind_inttimes(self, sInds, _epoch, det_mode):
        """Get the detection integration times for the given stars."""
        Comp = self.Completeness
        TK = self.TimeKeeping

        intTimes = np.zeros(len(sInds))

        for i, sInd in enumerate(sInds):
            # Get the star ensemble for this star (should always exist now)
            star_ensemble = Comp.star_ensembles.get(sInd)
            if star_ensemble is not None:
                # Find the time index closest to current time
                t_ind = (
                    np.searchsorted(
                        star_ensemble.times,
                        TK.currentTimeNorm.to_value(u.day),
                        side="right",
                    )
                    - 1
                )
                # Clamp to valid range
                t_ind = np.clip(t_ind, 0, len(star_ensemble.best_int_times) - 1)
                intTimes[i] = star_ensemble.best_int_times[t_ind]
            else:
                # Fallback to a default integration time if no ensemble exists
                # This shouldn't happen with the new system, but included for safety
                self.vprint(f"Warning: No StarEnsemble found for star {sInd}")
                intTimes[i] = self.min_int_time.to_value(u.day)

        return intTimes << u.d

    def _available_obs_time(self, ref=None, mode=None):
        """Calculates the available exoplanet observation time.

        Accounts for spent time, currently scheduled (held) time, and future
        reserved time for active tracks. Returns available time in days.
        """
        TK = self.TimeKeeping

        # Time already spent on exoplanet science observations
        spent_exoplanet_time_d = TK.exoplanetObsTime.to_value(u.d)
        if mode is None:
            mode = self.base_det_mode

        obs_oh_time = (mode["syst"]["ohTime"] + self.Observatory.settlingTime).to_value(
            u.d
        )

        # Time for observations already in self.schedule (future commitments)
        # act.duration should already be in days and include overheads
        scheduled_held_d = sum(
            act.duration
            for act in self.schedule
            if (ref is None)
            or (
                act.start
                >= (ref if isinstance(ref, (float, int)) else ref.to_value(u.d))
            )
        )

        # Newly calculated reserved time for future needs of all active PlanetTracks
        future_tracks_reserved_d = self._calculate_total_reserved_time_for_tracks()

        # Total time allocated for exoplanet science in the mission
        total_allocated_exoplanet_time_d = TK.allocated_time_d

        # Net available time for new initiatives (e.g., blind searches)
        net_available_d = (
            total_allocated_exoplanet_time_d
            - spent_exoplanet_time_d
            - scheduled_held_d
            - future_tracks_reserved_d
            - obs_oh_time
        )

        return max(0.0, net_available_d)

    def _get_estimated_max_int_time_for_mode(self, mode, sInd):
        """Helper to get a pessimistic (maximum) estimate of integration time.

        For a given mode and star, returns the maximum integration time.

        Args:
            mode (dict):
                Observing mode.
            sInd (int):
                Star index.

        Returns:
            float:
                Maximum integration time in days.
        """
        # Ensure dMag0s is initialized and contains the mode and sInd
        dMag0Grid_for_star = self.dMag0s[mode["hex"]][sInd]
        if dMag0Grid_for_star.int_times.size > 0:
            return float(dMag0Grid_for_star.int_times[-1])

    def _calculate_total_reserved_time_for_tracks(self):
        """Estimates the total future observation time (integration + overheads).

        Accounts for all active PlanetTracks, excluding deferred tracks.

        Returns:
            float:
                Total reserved time in days.
        """
        total_reserved_time_d = 0.0

        # Get overhead estimates in days
        det_oh_d = (
            self.base_det_mode["syst"]["ohTime"] + self.Observatory.settlingTime
        ).to_value(u.d)
        char_oh_d = (
            self.base_char_mode["syst"]["ohTime"] + self.Observatory.settlingTime
        ).to_value(u.d)

        # Get set of deferred track keys to exclude from reservation
        deferred_track_keys = set()
        for stage_tracks in self.to_requeue_later.values():
            for track in stage_tracks:
                deferred_track_keys.add((track.sInd, track.pInd))

        for track in self.planet_tracks.values():
            key = (track.sInd, track.pInd)

            # Skip deferred tracks - they shouldn't reserve time since they
            # can't be scheduled
            if key in deferred_track_keys:
                continue

            sInd = track.sInd

            # Get pessimistic (maximum) integration time estimates for this track's star
            est_max_int_time_det_d = (
                self._get_estimated_max_int_time_for_mode(self.base_det_mode, sInd)
                * 0.25
            )
            est_max_int_time_char_d = (
                self._get_estimated_max_int_time_for_mode(self.base_char_mode, sInd)
                * 0.25
            )

            remaining_detections_needed = 0
            remaining_characterizations_needed = 0

            # Calculate remaining characterizations first, as they depend on
            # prior detections
            if track.char_successes < self.required_char_successes:
                # If not fully characterized
                if track.det_successes >= self.required_det_successes:
                    # All detections are done, only characterizations remain
                    # for this track
                    remaining_characterizations_needed = (
                        self.required_char_successes - track.char_successes
                    )
                else:
                    # Still needs detections. Assume these detections will be successful
                    # and then will require all characterization observations.
                    remaining_detections_needed = (
                        self.required_det_successes - track.det_successes
                    )
                    remaining_characterizations_needed = self.required_char_successes
                    # (or self.required_char_successes - track.char_successes
                    # if some chars were done opportunistically)
                    # For simplicity, assuming all chars are needed once dets are done:
                    remaining_characterizations_needed = (
                        self.required_char_successes - track.char_successes
                    )

            # Time for remaining detections for this track
            reserved_for_track_dets = remaining_detections_needed * (
                est_max_int_time_det_d + det_oh_d
            )

            # Time for remaining characterizations for this track
            reserved_for_track_chars = remaining_characterizations_needed * (
                est_max_int_time_char_d + char_oh_d
            )

            total_reserved_time_d += reserved_for_track_dets + reserved_for_track_chars

        return total_reserved_time_d

    def post_inttime_filter(self, _epoch, sInds, intTimes, mode):
        """Create a mask for the detection targets after knowing intTimes."""
        if len(sInds) == 0:
            return np.zeros(0, dtype=bool)
        # Remove targets with intTimes of 0
        mask = intTimes != 0 * u.d

        # Remove stars that are not observable for the entire observation window
        mask &= self.ko_observability(sInds, _epoch, intTimes, mode)

        # Remove stars that don't fit in the schedule
        mask &= self.fits_schedule(
            sInds, intTimes, _epoch.start_det[sInds], mode, "detection"
        )

        # Remove observations that would exceed the available observation time
        # or use observation time saved for a scheduled observation
        _necessary_time = (
            intTimes.to_value(u.d)
            + self.Observatory.settlingTime.to_value(u.d)
            + mode["syst"]["ohTime"].to_value(u.d)
        )
        int_time_mask = _necessary_time <= self._available_obs_time()
        mask &= int_time_mask

        return mask

    def ko_observability(self, sInds, _epoch, intTimes, mode):
        """Return a boolean array the same length as sInds.

        True means the star is outside keepout for the full observation window.
        """
        koMap = self.koMaps[mode["syst"]["name"]]
        start_mjd = _epoch.start_det[sInds]
        end_mjd = start_mjd + intTimes.to_value(u.d)
        # idx0 = index of time whose interval starts just before `start`
        idx0 = np.searchsorted(self.koTimes_mjd, start_mjd, side="right") - 1
        # idx1 = index of time whose interval starts just before `end`
        idx1 = np.searchsorted(self.koTimes_mjd, end_mjd, side="right") - 1

        # clip to avoid out of bounds
        idx0 = np.clip(idx0, 0, len(self.koTimes_mjd) - 1)
        idx1 = np.clip(idx1, 0, len(self.koTimes_mjd) - 1)

        observable = np.zeros(len(sInds), dtype=bool)
        for k, sInd in enumerate(sInds):
            # Determine if the star is observable for the entire observation window
            observable[k] = (koMap[sInd, idx0[k] : idx1[k] + 1]).all()

        return observable

    def is_observable(self, sInd, mode_name, start_mjd, end_mjd=None):
        """Checks if a target star is observable for a given mode and time.

        Args:
            sInd (int):
                The star index.
            mode_name (str):
                The name of the observing mode (key for self.ko_intervals).
                 Typically from mode['syst']['name'].
            start_mjd (float):
                The start MJD of the check.
            end_mjd (Optional[float]):
                The end MJD of the check. The interval is treated as
                [start_mjd, end_mjd). If None, a point-in-time check is
                performed at start_mjd.

        Returns:
            bool: True if the star is observable during the specified time/interval,
                  False if it is in keepout or if the mode is not found.
        """
        if mode_name not in self.ko_intervals:
            self.logger.error(
                f"Mode '{mode_name}' not found in ko_intervals for sInd={sInd}"
                ". Assuming not observable."
            )
            return False

        # Check if sInd is valid for the list of trees for that mode
        if sInd < 0 or sInd >= len(self.ko_intervals[mode_name]):
            self.logger.error(
                f"Invalid sInd {sInd} for mode '{mode_name}' in ko_intervals."
                " Assuming not observable."
            )
            return False

        star_ko_tree = self.ko_intervals[mode_name][sInd]

        if end_mjd is None:
            # Point-in-time check: Is the star in keepout AT start_mjd?
            # .at(point) returns a set of intervals containing the point.
            # If the set is non-empty, it's in keepout.
            return not bool(star_ko_tree.at(start_mjd))
        else:
            # Duration check: Is the star in keepout ANYWHERE IN [start_mjd, end_mjd)?
            if start_mjd >= end_mjd:
                self.logger.warning(
                    f"Invalid duration for observability check: start_mjd ({start_mjd})"
                    f" >= end_mjd ({end_mjd}) for sInd {sInd}, mode {mode_name}. "
                    "Assuming not observable."
                )
                return False
            # .overlaps(begin, end) returns True if any interval in the tree
            # overlaps [begin, end).
            # So, if it overlaps, it's in keepout (not observable).
            return not star_ko_tree.overlaps(start_mjd, end_mjd)

    def _get_observable_int_time_mask_for_start_time(
        self,
        sInd: int,
        mode_name: str,
        obs_start_mjd: float,
        sorted_int_times_days: np.ndarray,
        overhead_days: float,
    ) -> np.ndarray:
        """Observable integration time mask for a given start time, mode, and star.

        For a given star and observation start time, determines which integration
        times result in an observable window using a bisection search approach.

        Assumes sorted_int_times_days is sorted in ascending order.

        Args:
            sInd (int):
                Star index.
            mode_name (str):
                Observing mode name.
            obs_start_mjd (float):
                The MJD for the start of the observation.
            sorted_int_times_days (np.ndarray):
                1D array of integration times (in days), sorted ascending.
            overhead_days (float):
                Total overhead time (settling + instrument) in days.

        Returns:
            np.ndarray:
                A 1D boolean mask of the same length as sorted_int_times_days.
                True if the observation [obs_start_mjd, obs_start_mjd +
                int_time + overhead) is observable, False otherwise.
        """
        num_int_times = len(sorted_int_times_days)
        if num_int_times == 0:
            return np.array([], dtype=bool)

        # This mask will be [True, True, ..., True, False, False, ...]
        observable_mask = np.zeros(num_int_times, dtype=bool)

        # Bisection search to find the first integration time index (k)
        # such that the interval
        # [obs_start_mjd, obs_start_mjd + sorted_int_times_days[k] + overhead_days)
        # is NOT observable. All intervals with indices < k will be observable.
        low = 0
        # Exclusive upper bound for search, represents count of observable int_times
        high = num_int_times
        # Assume all are observable initially
        first_non_observable_idx = num_int_times

        while low < high:
            mid_idx = low + (high - low) // 2
            current_int_time = sorted_int_times_days[mid_idx]
            obs_end_mjd = obs_start_mjd + current_int_time + overhead_days

            if self.is_observable(sInd, mode_name, obs_start_mjd, obs_end_mjd):
                # This interval is observable. The first non-observable one
                # must be further to the right (i.e., a longer integration
                # time, or all are observable so far).
                low = mid_idx + 1
            else:
                # This interval is NOT observable. This could be the first
                # non-observable one. Record it and search in the left half
                # (including this mid_idx).
                first_non_observable_idx = mid_idx
                high = mid_idx

        # All integration times from index 0 up to first_non_observable_idx - 1
        # are observable.
        if first_non_observable_idx > 0:
            observable_mask[0:first_non_observable_idx] = True

        return observable_mask

    def _get_observable_int_time_mask_for_start_time(
        self,
        sInd: int,
        mode_name: str,
        obs_start_mjd: float,
        sorted_int_times_days: np.ndarray,
        overhead_days: float,
        mission_finish_mjd: float,
    ) -> np.ndarray:
        num_int_times = len(sorted_int_times_days)
        if num_int_times == 0:
            return np.array([], dtype=bool)

        observable_mask = np.zeros(num_int_times, dtype=bool)
        star_ko_tree = self.ko_intervals[mode_name][sInd]

        # Check if obs_start_mjd itself is in a keepout interval
        if star_ko_tree.at(obs_start_mjd):
            # All False, cannot start an observation
            return observable_mask

        # Find the start of the next keepout interval strictly after obs_start_mjd
        # Query intervals that start after obs_start_mjd or overlap with a
        # point slightly after obs_start_mjd
        # This ensures we find the *next* relevant keepout.
        potential_limiters = [
            iv.begin for iv in star_ko_tree if iv.begin > obs_start_mjd
        ]

        # Default if no future keepouts
        min_future_ko_begin_mjd = mission_finish_mjd
        if not potential_limiters:
            # No keepout starts after obs_start_mjd
            pass
        else:
            min_future_ko_begin_mjd = min(potential_limiters)

        # Max observation end time is the start of the next keepout
        max_obs_end_mjd = min_future_ko_begin_mjd

        # Max allowed total duration (int_time + overhead)
        max_total_duration_days = max_obs_end_mjd - obs_start_mjd
        if max_total_duration_days <= 0:
            # Should not happen if .at() was false and logic is right
            return observable_mask

        max_allowed_int_time_days = max_total_duration_days - overhead_days

        if max_allowed_int_time_days <= 0:
            return observable_mask

        # count of elements <= max_allowed_int_time_days
        count_observable = np.searchsorted(
            sorted_int_times_days, max_allowed_int_time_days, side="right"
        )

        observable_mask[:count_observable] = True

        return observable_mask

    def fits_schedule(self, sInds, intTimes, start_mjds, mode, purpose):
        """Returns a boolean array the same length as sInds.

        True means the observation does not conflict with scheduled observations.
        """
        oh = (mode["syst"]["ohTime"] + self.Observatory.settlingTime).to_value(u.day)
        # build ObservationRequests in a loop because intTimes is Quantity
        fits = np.ones(len(sInds), dtype=bool)
        _int_times = intTimes.to_value(u.d)
        for k, (ind, int_time, start) in enumerate(zip(sInds, _int_times, start_mjds)):
            conflict = self._itr.overlaps(start, start + int_time + oh)
            fits[k] = not conflict
        return fits

    def select_target(self, blind_sInds, blind_intTimes, slewTimes):
        """Select the next target to observe.

        Returns an Observation object with all available info filled in.
        """
        target = None
        if len(blind_sInds) > 0:
            # Choose a blind detection target
            purpose = "detection"
            target, int_time, wait_time, comp, comp_d_t = self.choose_blind_target(
                blind_sInds, blind_intTimes, slewTimes
            )
            oh = (
                self.Observatory.settlingTime + self.base_det_mode["syst"]["ohTime"]
            ).to_value(u.day)
        if target is None:
            # No targets available, return None
            return None
        if comp_d_t < self.min_comp_div_int_time:
            # Observations are not providing significant information, so we should
            # wait until the next possible observation

            # If there is an obseravtion within the minimum possible integration time
            # return None, otherwise do a short wait
            _oh = self.Observatory.settlingTime.to_value(u.day) + self.base_det_mode[
                "syst"
            ]["ohTime"].to_value(u.day)
            min_duration = 2 * (self.min_int_time_d + oh)
            current_time = self.TimeKeeping.currentTimeAbs.mjd
            if self.schedule:
                next_action = self.schedule[0]
                if next_action.start - current_time < min_duration:
                    return None
            # do a short wait
            action = ScheduleAction(
                start=current_time,
                duration=min_duration,
                purpose="general_astrophysics",
                target=None,
                mode=None,
            )
        else:
            # Gather all available info for the observation
            arrival_time = self.TimeKeeping.currentTimeAbs.mjd

            # Increment the observation number since we're scheduling an observation
            self.ObsNum += 1
            _int_time = int_time.to_value(u.day)
            action = ScheduleAction(
                start=arrival_time,
                duration=_int_time + oh,
                purpose=purpose,
                target=target,
                mode=self.base_det_mode,
                int_time=_int_time,
                blind=True,
                comp=comp,
            )
            action.mode = self.base_det_mode
        return action

    def choose_blind_target(self, sInds, intTimes, slewTimes):
        """Choose a target from the list of blind targets.

        Returns None if there are no available blind targets.

        Args:
            sInds (integer array):
                Indices of available blind targets
            intTimes (astropy Quantity array):
                Integration times for the available blind targets
            slewTimes (astropy Quantity array):
                Slew times for the available blind targets
        """
        Comp = self.Completeness
        TK = self.TimeKeeping

        if len(sInds) == 0:
            return None, None, None, 0, 0

        best_comp_div_intTime = 0
        best_sInd = None
        best_int_time = None
        best_ind = None

        # Check all available stars and find the one with highest comp/intTime
        for i, sInd in enumerate(sInds):
            star_ensemble = Comp.star_ensembles.get(sInd)
            if star_ensemble is None:
                self.vprint(f"Warning: No StarEnsemble found for star {sInd}")
                continue

            # Check if star is available (not in waiting period)
            if TK.currentTimeNorm.to_value(u.day) < star_ensemble.next_available_time:
                continue

            # Find the time index closest to current time
            t_ind = (
                np.searchsorted(
                    star_ensemble.times,
                    TK.currentTimeNorm.to_value(u.day),
                    side="right",
                )
                - 1
            )
            # Clamp to valid range
            t_ind = np.clip(t_ind, 0, len(star_ensemble.best_comp_div_intTime) - 1)

            # Get the completeness/intTime for this star at this time
            comp_div_intTime = star_ensemble.best_comp_div_intTime[t_ind]

            if comp_div_intTime > best_comp_div_intTime:
                best_comp_div_intTime = comp_div_intTime
                best_sInd = sInd
                best_int_time = star_ensemble.best_int_times[t_ind]
                best_ind = i

        if best_sInd is None:
            # No valid targets available
            return None, None, None, 0, 0

        # Get the slew time for the selected target
        slew_time = slewTimes[best_ind]

        # Calculate completeness
        comp = self.PlanetPopulation._eta * best_comp_div_intTime * best_int_time

        target = Target.star(best_sInd)
        return target, best_int_time * u.d, slew_time, comp, best_comp_div_intTime

    def observation_detection(self, sInd, intTime, mode):
        """Determines detection SNR and detection status for a given integration time.

        Also updates the lastDetected and starRevisit lists.

        Args:
            sInd (int):
                Integer index of the star of interest
            intTime (~astropy.units.Quantity(~numpy.ndarray(float))):
                Selected star integration time for detection in units of day.
                Defaults to None.
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                detected (numpy.ndarray(int)):
                    Detection status for each planet orbiting the observed target star:
                    1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
                fZ (astropy.units.Quantity(numpy.ndarray(float))):
                    Surface brightness of local zodiacal light in units of 1/arcsec2
                JEZ (astropy.units.Quantity(numpy.ndarray(float))):
                    Intensity of exo-zodiacal light in units of photons/s/m2/arcsec2
                systemParams (dict):
                    Dictionary of time-dependant planet properties averaged over the
                    duration of the integration
                SNR (numpy.darray(float)):
                    Detection signal-to-noise ratio of the observable planets
                FA (bool):
                    False alarm (false positive) boolean

        """
        ZL = self.ZodiacalLight
        PPro = self.PostProcessing
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # Save Current Time before attempting time allocation
        currentTimeNorm = TK.currentTimeNorm.copy()
        currentTimeAbs = TK.currentTimeAbs.copy()

        dt = intTime / float(self.ntFlux)
        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]

        # initialize outputs
        detected = np.array([], dtype=int)
        # write current system params by default
        systemParams = SU.dump_system_params(sInd)
        SNR = np.zeros(len(pInds))

        # if any planet, calculate SNR
        if len(pInds) > 0:
            # initialize arrays for SNR integration
            fZs = np.zeros(self.ntFlux) << self.fZ_unit
            systemParamss = np.empty(self.ntFlux, dtype="object")
            JEZs = np.zeros((self.ntFlux, len(pInds))) << self.JEZ_unit
            Ss = np.zeros((self.ntFlux, len(pInds)))
            Ns = np.zeros((self.ntFlux, len(pInds)))
            # accounts for the time since the current time
            timePlus = Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
            # integrate the signal (planet flux) and noise
            for i in range(self.ntFlux):
                # allocate first half of dt
                timePlus += dt / 2.0
                # calculate current zodiacal light brightness
                fZs[i] = ZL.fZ(
                    Obs,
                    TL,
                    np.array([sInd], ndmin=1),
                    (currentTimeAbs).reshape(1),
                    mode,
                )[0]
                # propagate the system to match up with current time
                # SU.propag_system(
                #     sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
                # )
                SU.propag_system(sInd, currentTimeNorm - self.propagTimes[sInd])
                # Calculate the exozodi intensity
                JEZs[i] = SU.scale_JEZ(sInd, mode)
                self.propagTimes[sInd] = currentTimeNorm  # + timePlus
                # save planet parameters
                systemParamss[i] = SU.dump_system_params(sInd)
                # calculate signal and noise (electron count rates)
                Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                    sInd, pInds, dt, mode, fZ=fZs[i], JEZ=JEZs[i]
                )
                # allocate second half of dt
                timePlus += dt / 2.0

            # average output parameters
            fZ = np.mean(fZs)
            JEZ = np.mean(JEZs, axis=0)
            systemParams = {
                key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                / float(self.ntFlux)
                for key in sorted(systemParamss[0])
            }
            # calculate SNR
            S = Ss.sum(0)
            N = Ns.sum(0)
            SNR[N > 0] = S[N > 0] / N[N > 0]

        # if no planet, just save zodiacal brightness in the middle of the integration
        else:
            fZ = ZL.fZ(
                Obs,
                TL,
                np.array([sInd], ndmin=1),
                (currentTimeAbs).reshape(1),
                mode,
            )[0]
            # Use the default star value if no planets
            JEZ = TL.JEZ0[mode["hex"]][sInd]

        # find out if a false positive (false alarm) or any false negative
        # (missed detections) have occurred
        FA, MD = PPro.det_occur(SNR, mode, TL, sInd, intTime)

        # populate detection status array
        # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
        if len(pInds) > 0:
            detected = (~MD).astype(int)
            WA = (
                np.array(
                    [
                        systemParamss[x]["WA"].to_value(u.arcsec)
                        for x in range(len(systemParamss))
                    ]
                )
                << u.arcsec
            )
            detected[np.all(WA < mode["IWA"], 0)] = -1
            detected[np.all(WA > mode["OWA"], 0)] = -2

        # if planets are detected, calculate the minimum apparent separation
        det = detected == 1  # If any of the planets around the star have been detected

        # populate the lastDetected array by storing det, JEZ, dMag, and WA
        self.lastDetected[sInd, :] = [
            det,
            JEZ.flatten(),
            systemParams["dMag"],
            systemParams["WA"].to("arcsec").value,
        ]

        return detected.astype(int), fZ, JEZ, systemParams, SNR, FA

    def observation_characterization(self, sInd, mode, mode_index, char_intTime=None):
        """Finds if characterizations are possible and relevant information.

        Args:
            sInd (integer):
                Integer index of the star of interest
            mode (dict):
                Selected observing mode for characterization
            mode_index (int):
                Index of the observing mode
            char_intTime (astropy Quantity):
                Selected star characterization time in units of day. Defaults to None.

        Returns:
            characterized (integer list):
                Characterization status for each planet orbiting the observed
                target star including False Alarm if any, where 1 is full spectrum,
                -1 partial spectrum, and 0 not characterized
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            systemParams (dict):
                Dictionary of time-dependant planet properties averaged over the
                duration of the integration
            SNR (float ndarray):
                Characterization signal-to-noise ratio of the observable planets.
                Defaults to None.
            intTime (astropy Quantity):
                Selected star characterization time in units of day. Defaults to None.

        """
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        JEZs = SU.scale_JEZ(sInd, mode)

        # get the detected status, and check if there was a FA
        # det = self.lastDetected[sInd,0]
        pIndsDet = np.ones(pInds.size, dtype=bool)

        # initialize outputs, and check if there's anything (planet or FA)
        # to characterize
        characterized = np.zeros(len(pIndsDet), dtype=int)
        fZ = 0.0 << self.inv_arcsec2
        JEZ = 0.0 << self.JEZ_unit
        # write current system params by default
        systemParams = SU.dump_system_params(sInd)
        SNR = np.zeros(len(pIndsDet))
        intTime = char_intTime
        if len(pIndsDet) == 0:  # nothing to characterize
            return characterized, fZ, JEZ, systemParams, SNR, intTime

        # start times
        startTime = TK.currentTimeAbs.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
        startTimeNorm = (
            TK.currentTimeNorm.copy() + mode["syst"]["ohTime"] + Obs.settlingTime
        )
        # planets to characterize
        koTimeInd = np.where(np.round(startTime.value) - self.koTimes.value == 0)[0][0]
        # wherever koMap is 1, the target is observable
        koMap = self.koMaps[mode["syst"]["name"]]
        ko_good = koMap[sInd][koTimeInd]

        # Check that the observation time is valid

        # fZ = ZL.fZ(Obs, TL, np.array([sInd], ndmin=1), startTime.reshape(1), mode)
        # WAp = TL.int_WA[sInd] * np.ones(len(pInds))
        # dMag = TL.int_dMag[sInd] * np.ones(len(pInds))

        # Determine the single total time for the observation
        # intTime is already set to the max required time from next_target
        totTime = intTime
        # end times
        endTime = startTime + totTime
        endTimeNorm = startTimeNorm + totTime

        # Check if the calculated observation time is valid
        time_is_valid = (
            (totTime > 0)
            & (totTime <= OS.intCutoff)
            & (endTimeNorm <= TK.OBendTimes[TK.OBnumber])
        )

        # If the time is invalid, no planets can be characterized

        # Otherwise, tochar retains its current state (filtered by start keepout)
        # and will be further filtered by end keepout below.

        # 3/ is target still observable at the end of any char time?
        # Check keepout at the single endTime
        endTime_val = endTime.value

        # Find index for the rounded end time
        if endTime_val > self.koTimes.value[-1]:
            koTimeInd = -1  # End time is out of bounds
        else:
            # Find index of the last index that is less than or equal to endTime_val
            koTimeInd = np.searchsorted(self.koTimes_mjd, endTime_val, side="right") - 1

        # Get keepout status. If koTimeInd is invalid or target is not observable,
        # set end_time_observable to False.
        end_time_observable = False
        if koTimeInd != -1:
            koMap = self.koMaps[mode["syst"]["name"]]
            if koMap[sInd][koTimeInd]:
                end_time_observable = True

        # 4/ if yes, perform the characterization for the maximum char time
        if ko_good and end_time_observable and time_is_valid:
            # Save Current Time before attempting time allocation
            currentTimeNorm = TK.currentTimeNorm.copy()
            currentTimeAbs = TK.currentTimeAbs.copy()

            # intTime was calculated in next_target to be the maximum required
            # time for the planets considered characterizable then.
            # We use that value directly.

            extraTime = 0
            success = TK.allocate_time(
                intTime + extraTime + mode["syst"]["ohTime"] + Obs.settlingTime, True
            )
            # allocates time
            if not (success):
                char_intTime = None
                lenChar = len(pInds)
                characterized = np.zeros(lenChar, dtype=int)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0.0 << self.inv_arcsec2
                char_JEZ = 0.0 << self.JEZ_unit
                char_systemParams = SU.dump_system_params(sInd)
                return (
                    characterized,
                    char_fZ,
                    char_JEZ,
                    char_systemParams,
                    char_SNR,
                    char_intTime,
                )

            # pIndsChar = pIndsDet[tochar]
            # log_char = "   - Charact. planet(s) %s (%s/%s detected)" % (
            #     pIndsChar,
            #     len(pIndsChar),
            #     len(pIndsDet),
            # )
            # self.logger.info(log_char)
            # self.vprint(log_char)

            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsDet
            SNRplans = np.zeros(len(planinds))
            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros(self.ntFlux) << self.inv_arcsec2
                JEZs = np.zeros((self.ntFlux, len(planinds))) << self.JEZ_unit
                systemParamss = np.empty(self.ntFlux, dtype="object")
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime / float(self.ntFlux)
                timePlus = (
                    Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
                )  # accounts for the time since the current time
                for i in range(self.ntFlux):
                    # calculate signal and noise (electron count rates)
                    # allocate first half of dt
                    timePlus += dt / 2.0
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(
                        Obs,
                        TL,
                        np.array([sInd], ndmin=1),
                        (currentTimeAbs).reshape(1),
                        mode,
                    )[0]
                    # propagate the system to match up with current time
                    # SU.propag_system(
                    #     sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
                    # )
                    SU.propag_system(sInd, currentTimeNorm - self.propagTimes[sInd])
                    self.propagTimes[sInd] = currentTimeNorm  # + timePlus
                    # Calculate the exozodi intensity
                    JEZs[i] = SU.scale_JEZ(sInd, mode, pInds=pInds)
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                        sInd, pInds, dt, mode, fZ=fZs[i], JEZ=JEZs[i]
                    )
                    # allocate second half of dt
                    timePlus += dt / 2.0

                # average output parameters
                fZ = np.mean(fZs)
                JEZ = np.mean(JEZs, axis=0)
                systemParams = {
                    key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                    / float(self.ntFlux)
                    for key in sorted(systemParamss[0])
                }
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0] / N[N > 0]
                # allocate extra time for timeMultiplier
            # if only a FA, just save zodiacal brightness in the middle of the
            # integration
            else:
                # totTime = intTime * (mode["timeMultiplier"])
                fZ = ZL.fZ(
                    Obs,
                    TL,
                    np.array([sInd], ndmin=1),
                    TK.currentTimeAbs.copy().reshape(1),
                    mode,
                )[0]
                # Use the default star value if no planets
                JEZ = TL.JEZ0[mode["hex"]][sInd]

            # calculate the false alarm SNR (if any)
            SNRfa = []

            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(pIndsDet)[0]
            SNR[SNRinds] = np.append(SNRplans, SNRfa)

            # now, store characterization status: 1 for full spectrum,
            # -1 for partial spectrum, 0 for not characterized
            char = SNR >= mode["SNR"]
            # initialize with full spectra
            characterized = char.astype(int)
            # WAchar = WAs[char] * u.arcsec
            # # find the current WAs of characterized planets
            # WAs = systemParams["WA"]
            # check for partial spectra
            # IWA_max = mode["IWA"] * (1.0 + mode["BW"] / 2.0)
            # OWA_min = mode["OWA"] * (1.0 - mode["BW"] / 2.0)
            # char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
            # characterized[char] = -1

            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized
            self.fullSpectra[mode_index][pInds[charplans == 1]] += 1
            self.partialSpectra[mode_index][pInds[charplans == -1]] += 1

            # Create debug plots if enabled and we have characterized planets
            if (
                hasattr(self, "debug_plots") and self.debug_plots
                # and np.any(characterized != 0)
            ):
                # pInds_char = pInds[characterized != 0]
                # snr_char = SNR[characterized != 0]
                pInds_char = pInds
                snr_char = SNR
                if len(pInds_char) > 0:
                    self.plot_observation_results(
                        sInd,
                        pInds_char,
                        mode,
                        intTime,
                        snr_char,
                        purpose="characterization",
                        save_dir="plots/characterization",
                    )

        return characterized.astype(int), fZ, JEZ, systemParams, SNR, intTime

    def select_default_observing_modes(self):
        """Identify default detection and characterization observing modes."""
        OS = self.OpticalSystem
        allModes = OS.observingModes
        det_modes = list(filter(lambda mode: "imag" in mode["inst"]["name"], allModes))
        base_det_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(
            filter(lambda mode: "spec" in mode["inst"]["name"], allModes)
        )
        if np.any(spectroModes):
            char_modes = spectroModes
        # if no spectro mode, default char mode is first observing mode
        else:
            char_modes = [allModes[0]]

        self.det_modes = det_modes
        self.base_det_mode = base_det_mode
        self.char_modes = char_modes
        self.base_char_mode = char_modes[0]

    def do_detection(self, action):
        """Tasks related to a detection observation."""
        SU, TL, TK, Comp, Obs = (
            self.SimulatedUniverse,
            self.TargetList,
            self.TimeKeeping,
            self.Completeness,
            self.Observatory,
        )
        _meta = self._build_meta(action)
        _drm_entry = {}
        # Allocate Time
        _int_time = action.int_time.copy() << u.d
        extraTime = _int_time * (action.mode["timeMultiplier"] - 1.0)
        # calculates extraTime
        success = TK.allocate_time(
            _int_time + extraTime + Obs.settlingTime + action.mode["syst"]["ohTime"],
            True,
        )
        if not success:
            _tmp_time = (
                _int_time + extraTime + Obs.settlingTime + action.mode["syst"]["ohTime"]
            )
            self.logger.warning(
                f"Could not allocate observation detection time ({_tmp_time})."
            )
            return False

        # update visited list for selected star
        self.starVisits[action.target.sInd] += 1
        self.det_starVisits[action.target.sInd] += 1
        # PERFORM DETECTION
        detected, det_fZ, det_JEZ, det_systemParams, det_SNR, FA = (
            self.observation_detection(action.target.sInd, _int_time, action.mode)
        )

        if np.any(detected > 0):
            self.sInd_detcounts[action.target.sInd] += 1
            self.sInd_dettimes[action.target.sInd] = (
                self.sInd_dettimes.get(action.target.sInd) or []
            ) + [TK.currentTimeNorm.copy().to("day")]

        _drm_entry["det_time"] = action.int_time

        # populate the DRM with detection results
        _drm_entry["det_status"] = detected
        _drm_entry["det_fZ"] = det_fZ.to(self.fZ_unit)
        _drm_entry["det_params"] = det_systemParams
        _drm_entry["det_SNR"] = det_SNR
        if np.any(_meta["plan_inds"]):
            _drm_entry["det_JEZ"] = det_JEZ
            _drm_entry["det_dMag"] = SU.dMag[_meta["plan_inds"]].tolist()
            _drm_entry["det_WA"] = SU.WA[_meta["plan_inds"]].to(u.mas).value.tolist()

        if action.int_time is not None:
            det_comp = Comp.comp_per_intTime(
                action.int_time << u.d,
                TL,
                action.target.sInd,
                det_fZ,
                TL.JEZ0[action.mode["hex"]][action.target.sInd],
                TL.int_WA[action.target.sInd],
                action.mode,
            )[0]
            _drm_entry["det_comp"] = det_comp
        else:
            _drm_entry["det_comp"] = 0.0

        # handle case of inf OBs and missionPortion < 1
        # if np.isinf(TK.OBduration) and (TK.missionPortion < 1.0):
        #     self.arbitrary_time_advancement(TK.currentTimeAbs.mjd - action.start)
        _drm_entry["det_mode"] = dict(action.mode)
        _drm_entry["det_mode"].pop("inst", None)
        _drm_entry["det_mode"].pop("syst", None)

        return ObservationResult(data=_drm_entry, meta=_meta)

    def do_characterization(self, action):
        """Tasks related to a characterization observation."""
        TK, TL, Comp = (self.TimeKeeping, self.TargetList, self.Completeness)

        # Bookkeeping
        self.starVisits[action.target.sInd] += 1
        self.char_starVisits[action.target.sInd] += 1
        sInd = action.target.sInd

        # Execute the characterization with each mode
        # we're making an observation so increment observation number
        self.ObsNum += 1
        char_info = []
        _int_time = action.int_time.copy() << u.d
        for mode_index, char_mode in enumerate(self.char_modes):
            char_data = {}
            (
                characterized,
                char_fZ,
                char_JEZ,
                char_systemParams,
                char_SNR,
                char_intTime,
            ) = self.observation_characterization(
                sInd, char_mode, mode_index, _int_time
            )

            assert char_intTime != 0, "Integration time can't be 0."

            # populate the DRM with characterization results
            char_data["char_time"] = (
                char_intTime.to("day") if char_intTime is not None else self.zero_d
            )
            # WAS ... = characterized[:-1] if FA else characterized
            char_data["char_status"] = characterized
            # WAS ... = char_SNR[:-1] if FA else char_SNR
            char_data["char_SNR"] = char_SNR
            char_data["char_fZ"] = char_fZ.to(self.fZ_unit)
            char_data["char_JEZ"] = char_JEZ.to(self.JEZ_unit)
            char_data["char_params"] = char_systemParams

            if char_intTime is not None and np.any(characterized):
                char_comp = Comp.comp_per_intTime(
                    char_intTime,
                    TL,
                    sInd,
                    char_fZ,
                    TL.JEZ0[char_mode["hex"]][sInd],
                    TL.int_WA[sInd],
                    char_mode,
                )[0]
                char_data["char_comp"] = char_comp
            else:
                char_data["char_comp"] = 0.0
            # The FA block in coroOnlyScheduler used an old FA value so I'm ignoring it
            # populate the DRM with observation modes
            char_data["char_mode"] = dict(char_mode)
            # remove the inst and syst keys from the char_mode dict
            char_data["char_mode"].pop("inst", None)
            char_data["char_mode"].pop("syst", None)
            char_data["exoplanetObsTime"] = TK.exoplanetObsTime.copy()
            char_info.append(char_data)

        data = dict(char_info=char_info)
        meta = self._build_meta(action)

        return ObservationResult(data=data, meta=meta)

    def _build_meta(self, action):
        """Fields every DRM entry needs, regardless of purpose."""
        plan_inds = np.where(self.SimulatedUniverse.plan2star == action.target.sInd)[0]
        return dict(
            purpose=action.purpose,
            star_ind=action.target.sInd,
            star_name=self.TargetList.Name[action.target.sInd],
            arrival_time=action.start,
            OB_nb=self.TimeKeeping.OBnumber,
            ObsNum=self.ObsNum,
            plan_inds=plan_inds,
        )

    def _colorize_planet_indices(self, plan_inds, status):
        """Return a string of planet indices, green if detected, red if not.

        Args:
            plan_inds: array-like of planet indices
            status: array-like of detection status (1=detected, 0=not)

        Returns:
            str: colorized indices for terminal output
        """
        indices = []
        for idx, detected in zip(plan_inds, status):
            # Get the planet track if it exists
            key = (self.SimulatedUniverse.plan2star[idx], int(idx))
            track = self.planet_tracks.get(key)

            # Default label is just the detection status
            if detected > 0:
                label = "+"  # Detected this observation
            elif detected < 0:
                label = str(detected)  # Special status (-1 for IWA, -2 for OWA)
            else:
                label = "0"  # Not detected

            # If we have a track, use the D/C notation
            if track is not None:
                if track.char_successes > 0:
                    # Planet is in characterization phase
                    label = f"C{track.char_successes}"
                    color = "\033[92m"  # Green
                else:
                    # Planet is in detection phase
                    label = f"D{track.det_successes}"
                    color = "\033[94m"  # Blue
            else:
                # No track - use default colors based on current detection
                if detected > 0:
                    color = "\033[92m"  # Green
                else:
                    color = "\033[91m"  # Red

            indices.append(f"{color}{idx}[{label}]\033[0m")

        return "(" + ", ".join(indices) + ")"

    def log_obs(self, action):
        """Helper function to log an aligned, single-line summary of the observation.

        Also prints a second line with integration time, number of planets, and
        color-coded planet indices.
        """
        if action.purpose == "general_astrophysics":
            return
        # Start by just adding the action to our history list
        self.history.append(action)

        # Create log information
        visit_number = self.starVisits[action.target.sInd]
        det_visit_number = self.det_starVisits[action.target.sInd]
        char_visit_number = self.char_starVisits[action.target.sInd]

        # In case the action has no int_time, set the int_time_str to "-"
        int_time_str = "-"
        if hasattr(action, "int_time") and action.int_time is not None:
            int_time_str = f"{action.int_time:.2f} d"

        # Color characterizations green
        _purpose = (
            "\033[92mChar\033[0m" if action.purpose == "characterization" else "Det"
        )

        # Get planet indices and detection status
        plan_inds = (
            action.result.meta["plan_inds"]
            if len(action.result.meta["plan_inds"]) > 0
            else []
        )
        if action.purpose == "detection":
            status = (
                action.result.data["det_status"]
                if action.result.data["det_status"] is not None
                else np.zeros(len(plan_inds), dtype=int)
            )
        elif action.purpose == "characterization":
            if action.result.data.get("char_info"):
                status = action.result.data["char_info"][-1].get(
                    "char_status", np.zeros(len(plan_inds), dtype=int)
                )
            else:
                status = np.zeros(len(plan_inds), dtype=int)
        else:
            status = np.zeros(len(plan_inds), dtype=int)

        # Get and display mission statistics first
        stats, mission_header = self.get_mission_stats(action)

        # Format the observation summary
        obs_info = (
            f"Observation #{self.ObsNum:4d} | "
            f"{_purpose:<4} | "
            f"IntTime: {int_time_str} | "
            f"Star {action.target.sInd:3d} | "
            f"Visit {visit_number:2d} (det {det_visit_number:2d}, "
            f"char {char_visit_number:2d}) | "
        )

        # Get detailed planet info
        planet_details = []
        if len(plan_inds) > 0:
            for i, p_idx in enumerate(plan_inds):
                key = (self.SimulatedUniverse.plan2star[p_idx], int(p_idx))
                track = self.planet_tracks.get(key)
                if track:
                    if track.char_successes > 0:
                        stage_info = (
                            f"C{track.char_successes}/{self.required_char_successes}"
                        )
                    else:
                        stage_info = (
                            f"D{track.det_successes}/{self.required_det_successes}"
                        )
                    det_status = "+" if status[i] > 0 else "0"
                    planet_details.append(f"P{p_idx}: {stage_info} ({det_status})")

        planet_info = " | ".join(planet_details) if planet_details else "No planets"
        results_line = (
            f"Results: {self._colorize_planet_indices(plan_inds, status)} "
            f"| {planet_info}"
        )

        # Combine everything into a single custom formatted output
        # Split the mission header to inject our observation info
        header_parts = mission_header.strip().split("\n")

        # Assemble the combined output
        combined_output = "\n".join(header_parts[:-1])  # All except last line
        combined_output += f"\n{obs_info}\n{results_line}\n"
        combined_output += header_parts[-1]  # Add the closing line

        self.logger.info(combined_output)
        self.vprint(combined_output)

        # Add observation to DRM
        self.DRM.append(action.result.to_drm())

    def _format_mission_stats(self, stats, changes, action):
        """Format mission statistics into a readable summary string."""
        # Basic mission stats
        lines = [
            "\n══════════════ MISSION TRACKER ══════════════",
            (
                f"Unique stars: {stats['total_stars_visited']} | "
                f"Planets: {stats['detected_planets']} detected "
                f"| Comp: {stats['comp']:.2f}"
            ),
            (
                f"Active planet tracks: {stats['active_planets']} | "
                f"{stats['future_scheduled_observations']} scheduled "
                f"| {stats['deferred_tracks']} deferred"
            ),
            # Total number of blind observations | Number of scheduled
            # observations carried out
            (
                f"Observations: {stats['blind_observations']} blind "
                f"| {stats['past_scheduled_observations']} scheduled"
            ),
        ]

        # Combine detection and characterization stages on a single line
        stages_str = "Stages: "
        det_changes = changes.get("det_stage_changes", {}) if changes else {}
        char_changes = changes.get("char_stage_changes", {}) if changes else {}

        # First add all detection stages
        for stage in range(1, self.required_det_successes + 1):
            count = stats["detection_stages"].get(stage, 0)
            stage_diff = det_changes.get(stage, 0)

            # Format the stage count with highlighting if it changed
            if stage_diff > 0:
                # Blue for increase in detection
                stages_str += f"D{stage}: \033[94m{count}\033[0m | "
            elif stage_diff < 0:
                # Red for decrease
                stages_str += f"D{stage}: \033[91m{count}\033[0m | "
            else:
                stages_str += f"D{stage}: {count} | "

        # Then add all characterization stages
        for stage in range(1, self.required_char_successes + 1):
            stage_diff = char_changes.get(stage, 0)
            if stage == self.required_char_successes:
                count = stats["chars_completed"]
            else:
                count = stats["characterization_stages"].get(stage, 0)

            # Format the stage count with highlighting if it changed
            if stage_diff > 0:
                # Green for increase in characterization
                stages_str += f"C{stage}: \033[92m{count}\033[0m | "
            elif stage_diff < 0:
                # Red for decrease
                stages_str += f"C{stage}: \033[91m{count}\033[0m | "
            else:
                stages_str += f"C{stage}: {count} | "

        lines.append(stages_str[:-3])  # Remove the last " | "

        # Completed characterizations
        chars_completed = stats["chars_completed"]
        chars_diff = changes.get("new_chars", 0) if changes else 0

        # Highlight completed characterizations if changed
        if chars_diff > 0:
            lines.append(
                f"Completed characterizations: \033[92m{chars_completed}\033[0m"
            )
        else:
            lines.append(f"Completed characterizations: {chars_completed}")

        # Observation time stats with percentage
        lines.append(
            (
                f"Observation time: {stats['used_time']:.2f}/{stats['total_time']:.2f} "
                f"days used ({stats['percent_used']:.1f}%)"
            )
        )
        obs_year = self.TimeKeeping.currentTimeAbs.decimalyear
        end_time = self.TimeKeeping.missionFinishAbs.decimalyear
        start_time = self.TimeKeeping.missionStart.decimalyear
        fraction_remaining = (
            100 * (obs_year - start_time) / self.TimeKeeping.missionLife.to_value(u.yr)
        )
        lines.append(
            f"Mission time: {obs_year:6.2f}/{end_time:6.2f} ({fraction_remaining:.2f}%)"
        )
        lines.append("═════════════════════════════════════════════")
        return "\n".join(lines)

    def get_mission_stats(self, action=None):
        """Generate mission statistics summary, highlighting changes.

        Returns:
            dict: Current mission statistics
            str: Formatted summary string for logging
        """
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping

        # Track previous stats if we're tracking changes
        prev_stats = getattr(self, "_prev_mission_stats", None)

        # Calculate observation time stats
        # available_time = self._available_obs_time()
        available_time = TK.allocated_time_d - TK.exoplanetObsTime.to_value(u.d)
        total_time = TK.allocated_time_d
        used_time = total_time - available_time
        percent_used = (used_time / total_time) * 100 if total_time > 0 else 0

        # Calculate deferred tracks stats
        total_deferred_tracks = sum(
            len(tracks) for tracks in self.to_requeue_later.values()
        )
        deferred_by_stage = {
            stage: len(tracks) for stage, tracks in self.to_requeue_later.items()
        }

        # Initialize stats dictionary
        stats = {
            "total_planets": SU.nPlans,
            "detected_planets": len(
                self._all_detected_planets
            ),  # Use the set of all detected planets
            "comp": sum(action.comp for action in self.history),
            "active_planets": len(self.planet_tracks),
            "fraction_detected": len(self._all_detected_planets) / SU.nPlans
            if SU.nPlans > 0
            else 0,
            "total_stars_visited": np.sum(self.starVisits > 0),
            "total_observations": self.ObsNum,
            "blind_observations": sum(1 for action in self.history if action.blind),
            "future_scheduled_observations": len(self.schedule),
            "past_scheduled_observations": sum(
                1 for action in self.history if not action.blind
            ),
            "detection_stages": {i + 1: 0 for i in range(self.required_det_successes)},
            "characterization_stages": {
                i + 1: 0 for i in range(self.required_char_successes)
            },
            "chars_completed": 0,
            "available_time": available_time,
            "total_time": total_time,
            "used_time": used_time,
            "percent_used": percent_used,
            "deferred_tracks": total_deferred_tracks,
            "deferred_by_stage": deferred_by_stage,
        }

        # Count planets at each detection and characterization stage
        for track in self.planet_tracks.values():
            if track.char_successes > 0:
                # Planet is in characterization phase
                if track.char_successes <= self.required_char_successes:
                    stats["characterization_stages"][track.char_successes] += 1
            else:
                # Planet is in detection phase
                if track.det_successes <= self.required_det_successes:
                    stats["detection_stages"][track.det_successes] += 1

        # Calculate completed characterizations (planets that finished all stages)
        if hasattr(self, "_completed_planets"):
            stats["chars_completed"] = len(self._completed_planets)
        else:
            self._completed_planets = set()
            stats["chars_completed"] = 0

        # Identify changes if we have a previous state and current action
        changes = {}
        if prev_stats and action:
            changes["new_detections"] = (
                stats["detected_planets"] - prev_stats["detected_planets"]
            )
            changes["new_chars"] = (
                stats["chars_completed"] - prev_stats["chars_completed"]
            )

            # Track detection stage changes
            det_stage_changes = {}
            for stage in stats["detection_stages"]:
                diff = stats["detection_stages"][stage] - prev_stats[
                    "detection_stages"
                ].get(stage, 0)
                if diff != 0:
                    det_stage_changes[stage] = diff
            changes["det_stage_changes"] = det_stage_changes

            # Track characterization stage changes
            char_stage_changes = {}
            for stage in stats["characterization_stages"]:
                diff = stats["characterization_stages"][stage] - prev_stats[
                    "characterization_stages"
                ].get(stage, 0)
                if diff != 0:
                    char_stage_changes[stage] = diff
            changes["char_stage_changes"] = char_stage_changes

        # Store current stats for next comparison
        self._prev_mission_stats = stats.copy()

        # Format the summary string
        summary = self._format_mission_stats(
            stats, changes if prev_stats else None, action
        )

        return stats, summary

    def plot_observation_results(
        self, sInd, pInds, mode, intTime, SNR, purpose="detection", save_dir="plots"
    ):
        """Create diagnostic plots for observation results.

        Args:
            sInd (int): Star index
            pInds (array): Planet indices
            mode (dict): Observation mode
            intTime (astropy.Quantity): Integration time
            SNR (array): Signal-to-noise ratios
            purpose (str): 'detection' or 'characterization'
            save_dir (str): Directory to save plots

        Returns:
            str: Path to the saved plot file
        """
        import os

        import matplotlib.pyplot as plt

        # Filter pInds and SNR to only those present in SU.orbix_planets
        SU = self.SimulatedUniverse
        valid_indices = [i for i, pInd in enumerate(pInds) if pInd in SU.orbix_planets]
        if not valid_indices:
            return None
        filtered_pInds = [pInds[i] for i in valid_indices]
        filtered_SNR = [SNR[i] for i in valid_indices]

        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)

        TL = self.TargetList
        TK = self.TimeKeeping

        # Create subplots
        fig, axs = plt.subplots(
            nrows=len(filtered_pInds),
            ncols=3,
            figsize=(12, 5 * len(filtered_pInds)),
            squeeze=False,  # Always return 2D array
        )

        # Add supertitle with observation details
        supertitle = (
            f"{TL.Name[sInd]} - {purpose.capitalize()} "
            f"- Integration Time: {intTime.to('day').value:.2f} days, SNRs: ["
        )
        for i, _SNR in enumerate(filtered_SNR):
            supertitle += f"{_SNR:.2f}"
            if i < len(filtered_SNR) - 1:
                supertitle += ", "
        supertitle += "]"
        fig.suptitle(supertitle)

        # Get observation time and grid data
        _dMag0Grid = self.dMag0s[mode["hex"]][sInd]
        _dMag0_intTimes = _dMag0Grid.int_times
        int_time_ind = np.searchsorted(_dMag0_intTimes, intTime.to_value(u.d))
        if int_time_ind >= len(_dMag0_intTimes):
            int_time_ind = len(_dMag0_intTimes) - 1
        fZ = self.ZodiacalLight.fZMap[mode["syst"]["name"]][sInd]
        kEZ = self.exact_kEZs[sInd]
        times = self.koTimes.mjd - TK.missionStart.mjd
        time_ind = np.argmin(np.abs(times - self.propagTimes[sInd].to_value(u.d)))
        _t = jnp.array([times[time_ind]])

        for i, pInd in enumerate(filtered_pInds):
            # Get the axes for the current row
            ax0 = axs[i, 0]
            ax1 = axs[i, 1]
            ax2 = axs[i, 2]

            orbix_planets = SU.orbix_planets[pInd]

            # Get planet track info
            key = (sInd, int(pInd))
            track = self.planet_tracks.get(key)
            stage_info = ""
            if track:
                if track.char_successes > 0:
                    stage_info = (
                        f" - C{track.char_successes}/{self.required_char_successes}"
                    )
                else:
                    stage_info = (
                        f" - D{track.det_successes}/{self.required_det_successes}"
                    )

            # alpha vs dMag
            alpha, dMag = orbix_planets.j_alpha_dMag(self.solver, _t)

            pdet = _dMag0Grid.pdet_planets(self.solver, times, orbix_planets, fZ, kEZ)
            _pdet = pdet[time_ind, int_time_ind]
            ax0.plot(
                alpha,
                dMag,
                label="Orbix Planets",
                linestyle="none",
                marker=".",
                alpha=0.3,
            )
            _alpha_real_arcsec = SU.WA[pInd].to_value(u.arcsec)
            _dMag_real = SU.dMag[pInd]
            ax0.scatter(
                _alpha_real_arcsec,
                _dMag_real,
                label="EXOSIMS Planet",
                marker="x",
                color="red",
                s=100,
                zorder=5,
            )
            ax0.set_xlabel("Separation [arcsec]")
            ax0.set_ylabel("$\\Delta$mag")
            ax0.set_title(
                (
                    f"Planet {pInd}{stage_info} - pdet: {float(_pdet):.2f}"
                    f" - SNR: {filtered_SNR[i]:.2f}"
                )
            )
            ax0.set_xlim(0, 0.25)
            ax0.set_ylim(15, 40)
            ax0.axvline(
                mode["IWA"].to_value(u.arcsec),
                color="red",
                linestyle="--",
                label="IWA",
            )
            ax0.axvline(
                mode["OWA"].to_value(u.arcsec),
                color="orange",
                linestyle="--",
                label="OWA",
            )
            ax0.legend()
            ax0.grid(True, alpha=0.5)

            # plot position
            pos = orbix_planets.j_prop_AU(self.solver, _t)
            pos_real = SU.r[pInd].to_value(u.AU)

            ax1.scatter(
                pos[:, 0],
                pos[:, 1],
                label="Orbix Planets",
                alpha=0.3,
                marker=".",
            )
            ax1.scatter(
                pos_real[0],
                pos_real[1],
                label="EXOSIMS Planet",
                marker="x",
                color="red",
                s=100,
                zorder=5,  # Ensure it's visible
            )
            ax1.set_xlabel("X [AU]")
            ax1.set_ylabel("Y [AU]")
            ax1.set_xlim(-2.5, 2.5)
            ax1.set_ylim(-2.5, 2.5)
            ax1.set_title(f"X vs Y - Planet {pInd}{stage_info}")
            ax1.legend()
            ax1.grid(True, alpha=0.5)

            # pdet heatmap
            TT, IT = np.meshgrid(times, _dMag0_intTimes)  # both (Ni, Nt)

            pcm = ax2.pcolormesh(
                TT,
                IT,
                pdet.T,
                shading="auto",  # let MPL infer the correct cell edges
                cmap="viridis",
            )

            fig.colorbar(pcm, ax=ax2, label="p(det)")

            ax2.set_xlabel("Time [days]")
            ax2.set_ylabel("Integration Time [days]")
            ax2.set_xlim(times[0], times[-1])
            ax2.set_ylim(
                _dMag0_intTimes[0],
                _dMag0_intTimes[-1],
            )

            # Mark detection times with vertical lines
            if sInd in self.sInd_dettimes:
                for _time in self.sInd_dettimes[sInd]:
                    ax2.axvline(
                        _time.to_value(u.d),
                        color="blue",
                        linestyle="--",
                        label="Detection" if i == 0 else "",
                    )

            # Mark current time
            ax2.axvline(
                times[time_ind],
                color="red",
                linestyle="--",
                label="Current time" if i == 0 else "",
            )

            # Add legend for the time plot if it's the first planet
            if i == 0:
                ax2.legend()

        plt.tight_layout()

        # Save the figure
        filename = f"{save_dir}/{self.ObsNum}_{purpose}_{sInd}.png"
        # plt.show()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

        self.logger.info(f"Saved observation plot to {filename}")

        return filename

    def _plot_scheduling_attempt(
        self,
        track,
        times,
        int_times,
        pdet,
        threshold,
        mode,
        ko_status=None,  # Now expected to be a 2D mask: (n_times, n_int_times)
        schedule_conflicts=None,
        time_limits=None,
        mission_limits=None,
        result=None,
        selected_time=None,
        selected_int=None,
    ):
        """Plots a scheduling attempt.

        Create diagnostic plots for scheduling attempts to help diagnose why
        characterization observations aren't being scheduled.

        Args:
            track (PlanetTrack):
                The planet track being scheduled
            times (array):
                Array of candidate observation times
            int_times (array):
                Array of candidate integration times
            pdet (array):
                2D array of detection probabilities (times x int_times)
            threshold (float):
                Detection threshold being used
            mode (dict):
                Observation mode
            ko_status (2D array, optional):
                Boolean mask (n_times, n_int_times) indicating if the
                observation window is fully observable (True) or not (False)
            schedule_conflicts (array, optional):
                Boolean mask of scheduling conflicts
            time_limits (array, optional):
                Boolean mask of integration time limits
            mission_limits (array, optional):
                Boolean mask of mission time limits
            result (str, optional):
                Result of scheduling attempt ("success", "threshold_failure", etc.)
            selected_time (float, optional):
                Selected observation time
            selected_int (float, optional):
                Selected integration time

        Returns:
            str: Path to the saved plot file
        """
        import os

        import matplotlib.pyplot as plt

        # Create directory if it doesn't exist
        save_dir = os.path.join(self.plot_dir, "scheduling")
        os.makedirs(save_dir, exist_ok=True)

        # Get star and planet info
        sInd, pInd = track.sInd, track.pInd
        star_name = self.TargetList.Name[sInd]
        star_dist = self.TargetList.dist[sInd].to_value(u.pc)
        a = self.SimulatedUniverse.a[pInd].to_value(u.AU)  # Semi-major axis
        radius = self.SimulatedUniverse.Rp[pInd].to_value(u.earthRad)

        is_char = mode == self.base_char_mode
        observation_type = "Characterization" if is_char else "Detection"

        # Create a 1x2 grid of subplots (bottom two plots removed)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

        # Add a title with relevant information
        status_info = f"Status: {result}" if result else ""
        det_status = f"D{track.det_successes}/{self.required_det_successes}"
        char_status = f"C{track.char_successes}/{self.required_char_successes}"
        fig.suptitle(
            (
                f"{star_name} ({star_dist:.2f} pc) - Planet {pInd} ({radius:.1f} "
                f"R⊕, {a:.2f} AU) - {observation_type} Scheduling\n"
            )
            + (
                f"{det_status}, {char_status} - Max pdet: {np.max(pdet):.3f},"
                f" Threshold: {threshold:.3f} - {status_info}"
            ),
            fontsize=16,
        )

        # Plot 1: Detection probability heatmap (times x int_times)
        TT, IT = np.meshgrid(times, int_times)  # both (Ni, Nt)
        im1 = axs[0].pcolormesh(
            TT,
            IT,
            pdet.T,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=max(1.0, np.max(pdet)),
        )
        fig.colorbar(im1, ax=axs[0], label="p(det)")

        # Highlight the selected point if available
        if selected_time is not None and selected_int is not None:
            axs[0].plot(
                selected_time, selected_int, "rx", markersize=10, label="Selected"
            )
            axs[0].legend()

        axs[0].set_xlabel("Time (days from mission start)")
        axs[0].set_ylabel("Integration Time (days)")
        axs[0].set_title("Detection Probability")

        # Plot 2: Constraint mask heatmap
        # Create a composite constraint mask
        composite_mask = np.zeros_like(pdet, dtype=float)
        mask_labels = []

        # Define colors for different constraints
        constraint_colors = {
            "below_threshold": [1.0, 0.7, 0.7],  # Light red
            "keepout": [0.7, 0.7, 1.0],  # Light blue
            "schedule_conflict": [1.0, 0.7, 1.0],  # Light purple
            "time_limit": [0.7, 1.0, 0.7],  # Light green
            "mission_limit": [1.0, 1.0, 0.7],  # Light yellow
        }

        # Create a custom colormap for constraints
        cmap_colors = [(1, 1, 1)]  # Start with white for no constraints
        for color in constraint_colors.values():
            cmap_colors.append(color)
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_constraints", cmap_colors, N=len(constraint_colors) + 1
        )

        # Apply each constraint mask
        # 0 = valid, 1-5 = invalid for different reasons
        if pdet is not None:
            below_threshold = pdet < threshold
            composite_mask[below_threshold] = 1
            mask_labels.append(
                ("Below threshold", constraint_colors["below_threshold"])
            )

        if ko_status is not None:
            # ko_status is now a 2D mask: (n_times, n_int_times)
            # False means not observable for the full duration
            not_observable_mask = ~ko_status
            composite_mask[not_observable_mask] = 2
            mask_labels.append(("Keepout", constraint_colors["keepout"]))

        if schedule_conflicts is not None:
            composite_mask[schedule_conflicts] = 3
            mask_labels.append(
                (
                    "Scheduled observation conflict",
                    constraint_colors["schedule_conflict"],
                )
            )

        if time_limits is not None:
            time_exceeded = ~time_limits
            composite_mask[time_exceeded] = 4
            mask_labels.append(
                ("Exceeds available integration time", constraint_colors["time_limit"])
            )

        if mission_limits is not None:
            mission_exceeded = ~mission_limits
            composite_mask[mission_exceeded] = 5
            mask_labels.append(
                ("Exceeds mission lifetime", constraint_colors["mission_limit"])
            )

        # Create mask plot
        axs[1].pcolormesh(
            TT,
            IT,
            composite_mask.T,
            shading="auto",
            cmap=custom_cmap,
            vmin=0,
            vmax=len(constraint_colors),
        )

        # Add custom legend for mask plot
        legend_elements = [Patch(facecolor="white", edgecolor="black", label="Valid")]
        for label, color in mask_labels:
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=label)
            )
        axs[1].legend(handles=legend_elements, loc="upper right")

        # Highlight the selected point if available
        if selected_time is not None and selected_int is not None:
            axs[1].plot(selected_time, selected_int, "rx", markersize=10)

        axs[1].set_xlabel("Time (days from mission start)")
        axs[1].set_ylabel("Integration Time (days)")
        axs[1].set_title("Constraint Visualization")

        # Save the figure
        timestamp = self.TimeKeeping.currentTimeNorm.to_value(u.day)
        filename = (
            f"{save_dir}/sched_{int(timestamp)}_{observation_type}_{sInd}_"
            f"{pInd}_{result}.{self.plot_format}"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved scheduling attempt plot to {filename}")

        # Also create individual constraint visualization plot
        self._plot_constraint_visualization(
            track,
            times,
            int_times,
            pdet,
            threshold,
            mode,
            ko_status=ko_status,
            schedule_conflicts=schedule_conflicts,
            time_limits=time_limits,
            mission_limits=mission_limits,
            result=result,
            selected_time=selected_time,
            selected_int=selected_int,
        )

        return filename

    def _plot_constraint_visualization(
        self,
        track,
        times,
        int_times,
        pdet,
        threshold,
        mode,
        ko_status=None,
        schedule_conflicts=None,
        time_limits=None,
        mission_limits=None,
        result=None,
        selected_time=None,
        selected_int=None,
    ):
        """Create individual constraint visualization plots for easier analysis."""
        # Create subdirectory for constraint plots
        constraint_dir = os.path.join(self.plot_dir, "scheduling", "constraints")
        os.makedirs(constraint_dir, exist_ok=True)

        # Get star and planet info
        sInd, pInd = track.sInd, track.pInd
        star_name = self.TargetList.Name[sInd]
        star_dist = self.TargetList.dist[sInd].to_value(u.pc)
        a = self.SimulatedUniverse.a[pInd].to_value(u.AU)  # Semi-major axis
        radius = self.SimulatedUniverse.Rp[pInd].to_value(u.earthRad)

        is_char = mode == self.base_char_mode
        observation_type = "Characterization" if is_char else "Detection"

        # Create single subplot for constraint visualization
        fig, ax = plt.subplots(1, 1, figsize=(9, 7), constrained_layout=True)

        # Add title with relevant information
        fig.suptitle(
            (
                f"{star_name} ({star_dist:.2f} pc) - Planet {pInd} ({radius:.1f} R⊕, "
                f"{a:.2f} AU)\n"
            ),
            fontsize=18,
        )

        # Create constraint mask heatmap
        TT, IT = np.meshgrid(times, int_times)  # both (Ni, Nt)
        composite_mask = np.zeros_like(pdet, dtype=float)
        mask_labels = []

        # Define colors for different constraints
        constraint_colors = {
            "below_threshold": [1.0, 0.7, 0.7],  # Light red
            "keepout": [0.7, 0.7, 1.0],  # Light blue
            "schedule_conflict": [1.0, 0.7, 1.0],  # Light purple
            "time_limit": [0.7, 1.0, 0.7],  # Light green
            "mission_limit": [1.0, 1.0, 0.7],  # Light yellow
        }

        # Create a custom colormap for constraints
        cmap_colors = [(1, 1, 1)]  # Start with white for no constraints
        for color in constraint_colors.values():
            cmap_colors.append(color)
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_constraints", cmap_colors, N=len(constraint_colors) + 1
        )

        # Apply each constraint mask
        # 0 = valid, 1-5 = invalid for different reasons
        if pdet is not None:
            below_threshold = pdet < threshold
            composite_mask[below_threshold] = 1
            mask_labels.append(
                ("Below threshold", constraint_colors["below_threshold"])
            )

        if ko_status is not None:
            # ko_status is now a 2D mask: (n_times, n_int_times)
            # False means not observable for the full duration
            not_observable_mask = ~ko_status
            composite_mask[not_observable_mask] = 2
            mask_labels.append(("Field of Regard", constraint_colors["keepout"]))

        if schedule_conflicts is not None:
            composite_mask[schedule_conflicts] = 3
            mask_labels.append(
                (
                    "Scheduled observation conflict",
                    constraint_colors["schedule_conflict"],
                )
            )

        if time_limits is not None:
            time_exceeded = ~time_limits
            composite_mask[time_exceeded] = 4
            mask_labels.append(
                ("Exceeds available integration time", constraint_colors["time_limit"])
            )

        if mission_limits is not None:
            mission_exceeded = ~mission_limits
            composite_mask[mission_exceeded] = 5
            mask_labels.append(
                ("Exceeds mission lifetime", constraint_colors["mission_limit"])
            )

        # Create mask plot
        ax.pcolormesh(
            TT,
            IT,
            composite_mask.T,
            shading="auto",
            cmap=custom_cmap,
            vmin=0,
            vmax=len(constraint_colors),
        )

        # Add custom legend for mask plot
        legend_elements = [Patch(facecolor="white", edgecolor="black", label="Valid")]
        for label, color in mask_labels:
            legend_elements.append(
                Patch(facecolor=color, edgecolor="black", label=label)
            )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=14)

        # Highlight the selected point if available
        if selected_time is not None and selected_int is not None:
            ax.plot(selected_time, selected_int, "rx", markersize=15, markeredgewidth=4)
            # Add annotation for the selected point
            # ax.annotate(
            #     f"Selected\n({selected_time:.1f}d, {selected_int:.2f}d)",
            #     xy=(selected_time, selected_int),
            #     xytext=(15, 15),
            #     textcoords="offset points",
            #     bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            #     fontsize=12,
            # )

        ax.set_xlabel("Time (days from mission start)", fontsize=16)
        ax.set_ylabel("Integration Time (days)", fontsize=16)
        ax.set_title("Scheduling Constraint Visualization", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True, alpha=0.3)

        # Add additional information text box
        info_text = f"Planet Track: sInd={sInd}, pInd={pInd}\n"
        info_text += (
            f"Detection Successes: {track.det_successes}/"
            f"{self.required_det_successes}\n"
        )
        info_text += (
            f"Characterization Successes: {track.char_successes}"
            f"/{self.required_char_successes}\n"
        )
        _last_obs_mjd = track.last_obs_mjd - self.TimeKeeping.missionStart.mjd
        info_text += f"Last Observation: {_last_obs_mjd:.2f} days ago\n"

        if result == "success" or result == "bump_success":
            if selected_time is not None and selected_int is not None:
                time_idx = np.argmin(np.abs(times - selected_time))
                int_idx = np.argmin(np.abs(int_times - selected_int))
                if time_idx < pdet.shape[0] and int_idx < pdet.shape[1]:
                    info_text += f"Selected pdet: {pdet[time_idx, int_idx]:.3f}"

        # Add text box with information
        # props = dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
        # ax.text(
        #     0.02,
        #     0.02,
        #     info_text,
        #     transform=ax.transAxes,
        #     fontsize=13,
        #     verticalalignment="bottom",
        #     bbox=props,
        # )

        # Save the constraint plot
        timestamp = self.TimeKeeping.currentTimeNorm.to_value(u.day)
        filename = (
            f"{constraint_dir}/constraint_{int(timestamp)}_{observation_type}_"
            f"{sInd}_{pInd}_{result}.{self.plot_format}"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved individual constraint plot to {filename}")

        return filename

    def calc_mission_stats(self):
        """Calculate mission statistics using PlanetTrack objects.

        Maintains the same format as the reduce_DRM function from driver.py.
        """
        # Get planet classifications
        self.subtypes, self.is_earth = self.classify_planets()

        # Initialize counters for each statistic
        dets = Counter()
        chars = Counter()
        det_subtypes = Counter()
        det_earths = Counter()
        char_subtypes = Counter()
        char_earths = Counter()

        # Process active planet tracks
        for key, track in self.planet_tracks.items():
            sInd, pInd = key

            # Count detections (planets detected at least once)
            if track.det_successes > 0:
                dets[pInd] += track.det_successes
                det_subtypes[self.subtypes[pInd]] += 1
                if self.is_earth[pInd]:
                    det_earths[pInd] += track.det_successes

            # Count characterizations (planets that completed characterization)
            if track.char_successes >= self.required_char_successes:
                chars[pInd] += 1
                char_subtypes[self.subtypes[pInd]] += 1
                if self.is_earth[pInd]:
                    char_earths[pInd] += 1

        # Process retired planet tracks
        for key, track in self.retired_tracks.items():
            sInd, pInd = key

            # Count detections
            if track.det_successes > 0:
                dets[pInd] += track.det_successes
                det_subtypes[self.subtypes[pInd]] += 1
                if self.is_earth[pInd]:
                    det_earths[pInd] += track.det_successes

            # Count characterizations
            if track.char_successes >= self.required_char_successes:
                chars[pInd] += 1
                char_subtypes[self.subtypes[pInd]] += 1
                if self.is_earth[pInd]:
                    char_earths[pInd] += 1

        # Store results in the same format as reduce_DRM
        self.mission_stats = {
            "dets": dict(dets),
            "chars": dict(chars),
            "det_subtypes": dict(det_subtypes),
            "det_earths": dict(det_earths),
            "char_subtypes": dict(char_subtypes),
            "char_earths": dict(char_earths),
        }

        # Add additional debugging statistics
        self.add_extended_mission_stats()

    def add_extended_mission_stats(self):
        """Calculate and add extended debugging statistics to self.mission_stats."""
        blind_dets = 0
        followup_dets = 0
        chars_obs = 0
        blind_int_time = 0.0 * u.d
        followup_int_time = 0.0 * u.d
        char_int_time = 0.0 * u.d
        planet_obs_times = defaultdict(list)

        for action in self.history:
            if action.purpose == "detection":
                if action.blind:
                    blind_dets += 1
                    if action.int_time:
                        blind_int_time += action.int_time * u.d
                else:
                    followup_dets += 1
                    if action.int_time:
                        followup_int_time += action.int_time * u.d
                    if action.target and action.target.kind == "planet":
                        key = (action.target.sInd, action.target.pInd)
                        planet_obs_times[key].append(action.start)
            elif action.purpose == "characterization":
                chars_obs += 1
                if action.int_time:
                    char_int_time += action.int_time * u.d
                if action.target and action.target.kind == "planet":
                    key = (action.target.sInd, action.target.pInd)
                    planet_obs_times[key].append(action.start)

        # Average wait time between follow-ups
        followup_waits = []
        for p_key, times in planet_obs_times.items():
            if len(times) > 1:
                sorted_times = sorted(times)
                followup_waits.extend(np.diff(sorted_times))

        avg_followup_wait = np.mean(followup_waits) if followup_waits else 0.0

        # Retired tracks summary
        retired_total = len(self.retired_tracks)
        retired_max_det_fail = 0
        retired_max_char_fail = 0

        for track in self.retired_tracks.values():
            if track.det_failures >= self.max_det_failures:
                retired_max_det_fail += 1
            elif track.char_failures >= self.max_char_failures:
                retired_max_char_fail += 1

        retired_other = retired_total - retired_max_det_fail - retired_max_char_fail

        extended_stats = {
            "num_blind_dets_attempted": blind_dets,
            "num_followup_dets_attempted": followup_dets,
            "num_chars_attempted": chars_obs,
            "total_int_time_blind_dets_d": float(blind_int_time.to_value(u.d)),
            "total_int_time_followup_dets_d": float(followup_int_time.to_value(u.d)),
            "total_int_time_chars_d": float(char_int_time.to_value(u.d)),
            "avg_followup_wait_d": float(avg_followup_wait),
            "num_retired_tracks": retired_total,
            "retired_due_to_max_det_failures": retired_max_det_fail,
            "retired_due_to_max_char_failures": retired_max_char_fail,
            "retired_due_to_other": retired_other,
        }

        self.mission_stats.update(extended_stats)

    def classify_planets(self):
        """This determines the Kopparapu bin of the planet."""
        TL = self.TargetList
        SU = self.SimulatedUniverse
        plan2star = SU.plan2star
        Rp = SU.Rp.to_value(u.earthRad)
        a = SU.a.to_value(u.AU)
        e = SU.e

        # Calculate the luminosity of the star, assuming main-sequence
        star_lum = TL.L[plan2star] * u.Lsun

        # Find the stellar flux at the planet's location as a fraction of earth's
        earth_Lp = const.L_sun / (1 * (1 + (0.0167**2) / 2)) ** 2
        Lp = (star_lum / (a * (1 + (e**2) / 2)) ** 2 / earth_Lp).decompose().value

        # Find Planet Rp range
        Rp_bins = np.array([0.5, 1.0, 1.75, 3.5, 6.0, 14.3])
        all_Rp_types = np.array(
            [
                "Rocky",
                "Super-Earth",
                "Sub-Neptune",
                "Sub-Jovian",
                "Jovian",
            ]
        )
        L_bins = np.array(
            [
                [182, 1.0, 0.28, 0.0035],
                [187, 1.12, 0.30, 0.0030],
                [188, 1.15, 0.32, 0.0030],
                [220, 1.65, 0.45, 0.0030],
                [220, 1.65, 0.40, 0.0025],
            ]
        )
        # Find the bin of the radius
        Rp_bin = np.digitize(Rp, Rp_bins) - 1
        Rp_bin = np.clip(Rp_bin, 0, len(all_Rp_types) - 1)
        Rp_types = all_Rp_types[Rp_bin]
        # TODO Fix this to give correct when at edge cases since technically
        # they're not straight lines

        # # index of planet temp. cold,warm,hot
        all_L_types = np.array(["Hot", "Warm", "Cold"])
        specific_L_bins = L_bins[Rp_bin, :]
        L_bin = np.zeros(len(Lp))
        for i in range(len(Lp)):
            L_bin[i] = np.digitize(Lp[i], specific_L_bins[i]) - 1
        L_bin = np.clip(L_bin, 0, len(all_L_types) - 1).astype(int)
        L_types = all_L_types[L_bin]
        subtypes = [f"{L_type} {Rp_type}" for L_type, Rp_type in zip(L_types, Rp_types)]

        # Determine if the planet is Earth-like
        # Reverse luminosity scaling
        scaled_a = a / np.sqrt(star_lum.to(u.Lsun).value)

        lower_a = 0.95
        upper_a = 1.67

        lower_R = 0.8 / np.sqrt(scaled_a)
        upper_R = 1.4
        earth_a_cond = (lower_a <= scaled_a) & (scaled_a < upper_a)
        earth_Rp_cond = (lower_R <= Rp) & (Rp < upper_R)

        is_earth = earth_a_cond & earth_Rp_cond
        return subtypes, is_earth

    def _queue_for_later_retry(self, track):
        """Queue a track for later retry instead of immediately retiring it.

        Tracks are organized by stage (priority level) and have a maximum retry count.

        Args:
            track (PlanetTrack): The track that couldn't be scheduled
        """
        key = (track.sInd, track.pInd)
        retry_count = self.track_retry_counts.get(key, 0)

        if retry_count < self.max_requeue_attempts:
            # Add to requeue_later organized by stage (priority)
            if track.stage not in self.to_requeue_later:
                self.to_requeue_later[track.stage] = []

            self.to_requeue_later[track.stage].append(track)
            self.track_retry_counts[key] = retry_count + 1

            self.logger.info(
                f"REQUEUE: Queueing track ({track.sInd},{track.pInd}) for later retry "
                f"(attempt {retry_count + 1}/{self.max_requeue_attempts}),"
                f" stage {track.stage}"
            )
        else:
            # Exceeded max retries, retire the track
            self._retire_track(track)
            self.logger.info(
                f"REQUEUE: Retiring track ({track.sInd},{track.pInd}) after "
                f"{self.max_requeue_attempts} failed scheduling attempts"
            )

    def _process_requeue_later(self, completed_track_stage):
        """Process tracks queued for later retry after an observation completes.

        Retries tracks at the same or lower priority level than the completed obs.

        Args:
            completed_track_stage (int):
                The stage of the track that just completed an observation.
        """
        if not self.to_requeue_later:
            return

        tracks_to_retry = []

        # Collect tracks at same or lower priority level for retry
        for stage in list(self.to_requeue_later.keys()):
            if stage <= completed_track_stage:
                stage_tracks = self.to_requeue_later.pop(stage, [])
                tracks_to_retry.extend(stage_tracks)

        if tracks_to_retry:
            self.logger.info(
                f"REQUEUE: Processing {len(tracks_to_retry)} tracks for retry "
                f"after stage {completed_track_stage} observation"
            )

        # Try to schedule each track again
        for track in tracks_to_retry:
            key = (track.sInd, track.pInd)

            # Verify track is still valid (not retired)
            if key in self.planet_tracks:
                # Create a dummy previous action for _queue_followup
                # This is needed because _queue_followup expects a previous action
                dummy_action = ScheduleAction(
                    start=self.TimeKeeping.currentTimeAbs.mjd,
                    duration=0.1,  # Minimal duration
                    purpose="detection"
                    if track.char_successes == 0
                    else "characterization",
                    target=Target.planet(track.sInd, track.pInd),
                    mode=self.base_det_mode
                    if track.char_successes == 0
                    else self.base_char_mode,
                    int_time=0.1,
                    blind=False,
                )

                try:
                    self._queue_followup(track, dummy_action)
                    self.logger.info(
                        f"REQUEUE: Successfully rescheduled track "
                        f"({track.sInd},{track.pInd})"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"REQUEUE: Failed to reschedule track "
                        f"({track.sInd},{track.pInd}): {e}"
                    )
                    # If rescheduling fails, queue it for another retry unless
                    # we've hit max attempts
                    self._queue_for_later_retry(track)
            else:
                self.logger.info(
                    f"REQUEUE: Track ({track.sInd},{track.pInd}) no longer active,"
                    " skipping"
                )

    def _tracks_for_star(self, sInd: int):
        """Iterator over all PlanetTracks belonging to one star."""
        return (t for t in self.planet_tracks.values() if t.sInd == sInd)

    def get_deferred_retry_status(self):
        """Get detailed status of the deferred retry system for debugging.

        Returns:
            dict: Detailed information about deferred tracks and retry counts
        """
        status = {
            "total_deferred": sum(
                len(tracks) for tracks in self.to_requeue_later.values()
            ),
            "by_stage": {},
            "retry_counts": dict(self.track_retry_counts),
            "max_attempts": self.max_requeue_attempts,
        }

        for stage, tracks in self.to_requeue_later.items():
            status["by_stage"][stage] = {
                "count": len(tracks),
                "tracks": [(t.sInd, t.pInd) for t in tracks],
            }

        return status

    def log_deferred_retry_status(self):
        """Log the current status of deferred retries."""
        status = self.get_deferred_retry_status()

        if status["total_deferred"] == 0:
            self.logger.info("REQUEUE STATUS: No tracks currently deferred")
            return

        self.logger.info(
            f"REQUEUE STATUS: {status['total_deferred']} tracks deferred for retry"
        )

        for stage, info in status["by_stage"].items():
            track_list = ", ".join([f"({s},{p})" for s, p in info["tracks"]])
            self.logger.info(f"  Stage {stage}: {info['count']} tracks - {track_list}")

    def _available_obs_time(self, ref=None, mode=None):
        """Calculates the available exoplanet observation time.

        Accounting for spent time, currently scheduled (held) time, and future
        reserved time for active tracks.

        Returns available time in days.
        """
        TK = self.TimeKeeping

        # Time already spent on exoplanet science observations
        spent_exoplanet_time_d = TK.exoplanetObsTime.to_value(u.d)
        if mode is None:
            mode = self.base_det_mode

        obs_oh_time = (mode["syst"]["ohTime"] + self.Observatory.settlingTime).to_value(
            u.d
        )

        # Time for observations already in self.schedule (future commitments)
        # act.duration should already be in days and include overheads
        scheduled_held_d = sum(
            act.duration
            for act in self.schedule
            if (ref is None)
            or (
                act.start
                >= (ref if isinstance(ref, (float, int)) else ref.to_value(u.d))
            )
        )

        # Newly calculated reserved time for future needs of all active PlanetTracks
        future_tracks_reserved_d = self._calculate_total_reserved_time_for_tracks()

        # Total time allocated for exoplanet science in the mission
        total_allocated_exoplanet_time_d = TK.allocated_time_d

        # Net available time for new initiatives (e.g., blind searches)
        net_available_d = (
            total_allocated_exoplanet_time_d
            - spent_exoplanet_time_d
            - scheduled_held_d
            - future_tracks_reserved_d
            - obs_oh_time
        )

        return max(0.0, net_available_d)

    def plot_final_schedule(self):
        """Plot the full mission schedule.

        Creates a final visualization showing all detected planets and their
        observations over time. Each planet is represented as a row, with
        detection and characterization events shown as markers. Planets are
        ordered by their first detection time. Keepout periods are shown as
        black lines overlaid on each row.

        The plot will be saved to the directory specified by self.plot_dir with the
        format specified by self.plot_format.
        """
        # Create the plot directory if it doesn't exist
        if not Path(self.plot_dir).exists():
            Path(self.plot_dir).mkdir(parents=True, exist_ok=True)
        if not self.history:
            self.logger.info("No observations to plot")
            return

        # Extract all planet observations from history
        planet_observations = {}  # {(sInd, pInd): [observation_actions]}
        detection_times = {}  # {(sInd, pInd): first_detection_time}
        star_names = {}  # {(sInd, pInd): star_name}
        star_blind_observations = {}  # {(sInd): [blind_actions]}

        # Process history to find all planet observations and their first
        # detection times
        for action in self.history:
            if action.blind:
                if action.target.sInd not in star_blind_observations:
                    star_blind_observations[action.target.sInd] = []
                star_blind_observations[action.target.sInd].append(action)
        remove_from_blind = set()
        for action in self.history:
            if action.target and action.target.kind == "planet" and action.result:
                key = (action.target.sInd, action.target.pInd)

                # Store star name
                star_names[key] = self.TargetList.Name[action.target.sInd]

                # Initialize list if this is the first observation of this planet
                if key not in planet_observations:
                    if action.target.sInd in star_blind_observations:
                        # Add the blind observations for this star
                        planet_observations[key] = star_blind_observations[
                            action.target.sInd
                        ]
                        remove_from_blind.add(action.target.sInd)
                    else:
                        planet_observations[key] = []

                # Add this observation to the planet's list
                planet_observations[key].append(action)

                # If this is a detection and it's successful, track the time
                if action.purpose == "detection":
                    detected = action.result.data.get("det_status", [])
                    if np.any(detected > 0) and key not in detection_times:
                        detection_times[key] = action.start

        # Create a definitive list of all first detection times for all planets
        # from the entire mission history. This handles both blind and followup
        # detections.
        all_detection_times = {}
        for action in self.history:
            if action.purpose == "detection" and action.result:
                det_status = action.result.data.get("det_status", [])
                if np.any(det_status > 0):
                    pInds = action.result.meta.get("plan_inds", [])
                    sInd = action.target.sInd
                    for i, pInd in enumerate(pInds):
                        if det_status[i] > 0:
                            key = (sInd, pInd)
                            if key not in all_detection_times:
                                all_detection_times[key] = action.start

        # Overwrite the potentially incomplete detection_times dictionary
        detection_times = all_detection_times

        # Now, ensure all planets with a recorded detection time are in
        # planet_observations
        # This is for planets that were only detected in a blind search
        for key in detection_times:
            if key not in planet_observations:
                sInd, pInd = key
                star_names[key] = self.TargetList.Name[sInd]
                if sInd in star_blind_observations:
                    planet_observations[key] = star_blind_observations[sInd]
                    remove_from_blind.add(sInd)

        for sInd in remove_from_blind:
            del star_blind_observations[sInd]
        # Create a list of all remaining blind observations
        remaining_blind_observations = []
        remaining_blind_stars = len(star_blind_observations.keys()) - len(
            remove_from_blind
        )
        for sInd in star_blind_observations:
            remaining_blind_observations.extend(star_blind_observations[sInd])

        if not planet_observations:
            self.logger.info("No planet observations to plot")
            return

        # Sort planets so that planets around the same star appear on
        # consecutive rows. We first order stars by the earliest detection
        # of any of their planets, then order planets within a star by their
        # own first detection (tie‑break by planet index).
        # Build star -> first detection mapping
        star_first_detection = {}
        for (sInd_k, pInd_k), det_time in detection_times.items():
            if sInd_k not in star_first_detection:
                star_first_detection[sInd_k] = det_time
            else:
                star_first_detection[sInd_k] = min(
                    star_first_detection[sInd_k], det_time
                )

        # Only include planets that had at least one successful detection
        detected_planet_keys = [
            k for k in planet_observations.keys() if k in detection_times
        ]

        sorted_planet_keys = sorted(
            detected_planet_keys,
            key=lambda k: (
                star_first_detection.get(k[0], float("inf")),
                detection_times.get(k, float("inf")),
                k[1],
            ),
        )

        # Create the plot
        fig, ax = plt.subplots(
            figsize=(14, max(10, len(sorted_planet_keys) * 0.4)),
            constrained_layout=True,
        )

        # Create a mapping from planet keys to row positions
        # Shift planets up by one row so row 0 is reserved for the
        # aggregated blind observations at the bottom. The first
        # detected planet still appears at the top.
        planet_positions = {
            key: len(sorted_planet_keys) - i for i, key in enumerate(sorted_planet_keys)
        }

        # Add a row for blind observations at the bottom (y=0)
        planet_positions[(None, None)] = 0

        # Define a color scheme
        colors = {
            "detection": "#3498db",  # Blue
            "characterization": "#2ecc71",  # Green
            "failed_detection": "#e74c3c",  # Red
            "failed_characterization": "#f39c12",  # Orange
            "blind_observation": "#9b59b6",  # Purple
            "keepout": "#000000",  # Black for keepout
        }

        # Get mission time range
        mission_start = self.TimeKeeping.missionStart.mjd
        mission_end = self.TimeKeeping.missionFinishAbs.mjd

        # Set the plot limits using MJD directly
        ax.set_xlim(mission_start, mission_end)
        ax.set_ylim(-0.5, len(sorted_planet_keys) + 0.5)

        # Add grid lines
        ax.grid(True, alpha=0.3, linestyle="--")

        # Plot keepout periods for each star
        # Collect unique sInds from all planet observations
        observed_sInds = set()
        for key in sorted_planet_keys:
            sInd, pInd = key
            observed_sInds.add(sInd)

        # Use the default detection mode for keepout visualization
        # (could be made configurable if needed)
        default_mode_name = self.base_det_mode["syst"]["name"]

        if hasattr(self, "ko_intervals") and default_mode_name in self.ko_intervals:
            for key in sorted_planet_keys:
                sInd, pInd = key
                row = planet_positions[key]

                # Get the keepout intervals for this star
                if sInd < len(self.ko_intervals[default_mode_name]):
                    star_ko_tree = self.ko_intervals[default_mode_name][sInd]

                    # Plot each keepout interval as a thick black line
                    for interval in star_ko_tree:
                        start_mjd = interval.begin
                        end_mjd = interval.end

                        # Only plot intervals that overlap with the mission time range
                        if start_mjd < mission_end and end_mjd > mission_start:
                            # Clamp to mission bounds
                            plot_start = max(start_mjd, mission_start)
                            plot_end = min(end_mjd, mission_end)

                            # Plot as a thick black line
                            ax.plot(
                                [plot_start, plot_end],
                                [row, row],
                                color=colors["keepout"],
                                linewidth=6,
                                alpha=0.7,
                                solid_capstyle="butt",
                            )

        # Plot each planet's observations
        for key in sorted_planet_keys:
            row = planet_positions[key]
            sInd, pInd = key

            # Plot a horizontal line for this planet's timeline
            ax.axhline(y=row, color="gray", alpha=0.3, linestyle="-")

            for action in planet_observations[key]:
                # Determine if the observation was successful
                successful = False
                if action.blind:
                    det_status = action.result.data.get("det_status", [])
                    plan_inds = action.result.meta.get("plan_inds", [])
                    # Find the index of this specific planet in the plan_inds array
                    if len(det_status) > 0 and len(plan_inds) > 0:
                        try:
                            planet_idx_in_array = np.where(np.array(plan_inds) == pInd)[
                                0
                            ]
                            if len(planet_idx_in_array) > 0:
                                successful = det_status[planet_idx_in_array[0]] > 0
                        except (IndexError, ValueError):
                            successful = False
                    # Match styling to follow-up detection markers
                    color = (
                        colors["detection"]
                        if successful
                        else colors["failed_detection"]
                    )
                    marker = "o" if successful else "x"
                    alpha = 1.0
                else:
                    plan_inds = action.result.meta.get("plan_inds", [])
                    if action.purpose == "detection":
                        det_status = action.result.data.get("det_status", [])
                        # Find the index of this specific planet in the plan_inds array
                        if len(det_status) > 0 and len(plan_inds) > 0:
                            try:
                                planet_idx_in_array = np.where(
                                    np.array(plan_inds) == pInd
                                )[0]
                                if len(planet_idx_in_array) > 0:
                                    successful = det_status[planet_idx_in_array[0]] > 0
                            except (IndexError, ValueError):
                                successful = False
                    elif action.purpose == "characterization":
                        if action.result.data.get("char_info"):
                            char_status = action.result.data["char_info"][-1].get(
                                "char_status", []
                            )
                            # Find the index of this specific planet in the
                            # plan_inds array
                            if len(char_status) > 0 and len(plan_inds) > 0:
                                try:
                                    planet_idx_in_array = np.where(
                                        np.array(plan_inds) == pInd
                                    )[0]
                                    if len(planet_idx_in_array) > 0:
                                        successful = (
                                            char_status[planet_idx_in_array[0]] > 0
                                        )
                                except (IndexError, ValueError):
                                    successful = False

                    # Choose the right color based on observation type and success
                    if action.purpose == "detection":
                        color = (
                            colors["detection"]
                            if successful
                            else colors["failed_detection"]
                        )
                        marker = "o" if successful else "x"
                    else:  # characterization
                        color = (
                            colors["characterization"]
                            if successful
                            else colors["failed_characterization"]
                        )
                        marker = "s" if successful else "X"

                    # Add a marker for this observation
                    # alpha = action.pdet
                    alpha = 1
                    # marker = "o" if successful else "x"

                # Convert color to RGBA and apply alpha only to face color
                face_color = to_rgba(color, alpha)
                edge_color = to_rgba(color, 1.0)  # Solid edge

                # Plot markers with solid borders but transparent fills
                ax.plot(
                    action.start,
                    row,
                    marker,
                    markerfacecolor=face_color,  # Apply alpha to face color only
                    markeredgecolor=edge_color,  # Solid edge color
                    markersize=8,
                )

                # Add a line representing the duration of the observation
                if action.duration > 0:
                    ax.plot(
                        [action.start, action.start + action.duration],
                        [row, row],
                        "-",
                        color=color,
                        linewidth=4,
                        alpha=1,
                    )
        # Plot the remaining blind observations as vertical lines colored by
        # completeness
        # Compute per-star lists (kept for potential future enhancements)
        from collections import defaultdict as _dd

        star_to_blind_actions = _dd(list)
        for _act in remaining_blind_observations:
            if _act.target is not None:
                star_to_blind_actions[_act.target.sInd].append(_act)

        # Set up colormap by completeness
        import matplotlib.pyplot as _plt
        from matplotlib.colors import Normalize as _Normalize

        _cmap = _plt.get_cmap("viridis")
        _comps = [
            float(getattr(_a, "comp", 0.0) or 0.0)
            for _a in remaining_blind_observations
        ]
        _cmin = min(_comps) if _comps else 0.0
        _cmax = max(_comps) if _comps else 1.0
        if _cmax <= _cmin:
            _cmax = _cmin + 1e-6
        _norm = _Normalize(vmin=_cmin, vmax=_cmax)
        # Draw vertical lines at each blind observation time
        blind_row_y = planet_positions[(None, None)]
        for action in remaining_blind_observations:
            _cval = float(getattr(action, "comp", 0.0) or 0.0)
            _color = _cmap(_norm(_cval))
            ax.vlines(
                x=action.start,
                ymin=blind_row_y - 0.35,
                ymax=blind_row_y + 0.35,
                colors=_color,
                linewidth=1,
            )
        # Add a small horizontal colorbar for blind completeness below the plot

        # Add y-axis labels with planet information
        planet_labels = []
        prev_star = None
        star_boundaries = []  # To store rows where star systems change

        # We need to create labels in the same order as the positions (inverted)
        for i, key in enumerate(reversed(sorted_planet_keys)):
            sInd, pInd = key

            # Track star system changes for visual grouping
            current_star = self.TargetList.Name[sInd]
            if prev_star is not None and current_star != prev_star:
                # Add boundary between different stars (shifted up by 1 due to
                # blind row)
                star_boundaries.append(i + 0.5)
            prev_star = current_star

            a = self.SimulatedUniverse.a[pInd].to_value(u.AU)  # Semi-major axis
            radius = self.SimulatedUniverse.Rp[pInd].to_value(u.earthRad)
            # Format the label with planet properties
            label = f"S{sInd} - P{pInd} ({radius:.1f} R⊕, {a:.2f} AU)"

            planet_labels.append(label)

        # Build labels with blind row first (bottom), followed by planets
        blind_label = (
            f"Rest of survey ({len(remaining_blind_observations)} obs,"
            f" {remaining_blind_stars} stars)"
        )
        planet_labels = [blind_label] + planet_labels

        # Boundary between blind row (0) and first planet row (1)
        star_boundaries.append(0.5)

        ax.set_yticks(range(len(sorted_planet_keys) + 1))
        ax.set_yticklabels(planet_labels)

        # Add horizontal lines to visually separate star systems
        for boundary in star_boundaries:
            ax.axhline(y=boundary, color="black", linestyle="-", alpha=0.5, linewidth=1)

        # Format the x-axis as decimal years
        # Using astropy.time to convert between MJD and decimal years
        from astropy.time import Time

        start_year = Time(mission_start, format="mjd").decimalyear
        end_year = Time(mission_end, format="mjd").decimalyear

        # Use yearly ticks across the mission duration
        # tick_years = np.arange(np.floor(start_year), np.ceil(end_year) + 1e-9, 1.0)
        # tick_years = np.arange(np.floor(start_year), np.ceil(end_year) + 1e-9, 1.0)
        tick_years = np.arange(
            np.round(start_year, 0), np.round(end_year, 0) + 1e-9, 1.0
        )
        tick_mjd = Time(tick_years, format="decimalyear").mjd

        ax.set_xticks(tick_mjd)
        ax.set_xticklabels([f"{y:.1f}" for y in tick_years])

        # Add mission summary information as text box
        stats, _ = self.get_mission_stats()
        # summary_text = (
        #     f"Mission Summary:\n"
        #     f"Duration: {mission_duration_years:.1f} years\n"
        #     f"Total planets detected:
        #     {stats['detected_planets']}/{stats['total_planets']}\n"
        #     f"Unique stars observed: {stats['total_stars_visited']}\n"
        #     f"Planets characterized: {stats['chars_completed']}\n"
        #     f"Total observations: {stats['total_observations']}\n"
        #     f"Observation time used:
        #     {stats['used_time']:.1f}/{stats['total_time']:.1f} days
        #     ({stats['percent_used']:.1f}%)"
        # )

        # # Add the summary text box
        # props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        # ax.text(
        #     0.02,
        #     0.02,
        #     summary_text,
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     verticalalignment="bottom",
        #     bbox=props,
        # )

        # Add a legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors["detection"],
                label="Detection",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=colors["characterization"],
                label="Characterization",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color=colors["failed_detection"],
                label="Failed Detection",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color=colors["failed_characterization"],
                label="Failed Characterization",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                color=colors["keepout"],
                label="Star Unavailable by FoR",
                linewidth=6,
                alpha=0.7,
            ),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="None",
                marker="|",
                markersize=12,
                label="All other detection attempts (color = completeness)",
            ),
        ]
        # Place legend underneath the plot next to the horizontal colorbar
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=3,
            frameon=True,
        )
        _sm = _plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)
        _sm.set_array([])
        fig.colorbar(
            _sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.06,
            pad=0.01,
            aspect=40,
            label="Blind completeness",
            location="bottom",
            # loc="upper center",
            # bbox_to_anchor=(0.5, 1.1),
        )

        # Add title and labels
        # ax.set_title("Mission Observation Schedule", fontsize=16)
        ax.set_xlabel("Year", fontsize=12)
        # ax.set_ylabel("Planets", fontsize=12)

        # Save the figure
        plot_path = Path(self.plot_dir, f"mission_schedule.{self.plot_format}")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"Final schedule plot saved to {plot_path}")
        self.vprint(f"Final schedule plot saved to {plot_path}")

    def cleanup_memory(self):
        """Comprehensive memory cleanup for simulation runs.

        Call this method after each simulation run to prevent memory buildup.
        """
        # Clear JAX compilation cache - this is the biggest memory consumer
        import jax

        try:
            # Clear JAX compilation cache
            jax.clear_caches()
        except Exception as e:
            self.logger.warning(f"Could not clear JAX backends: {e}")

        # Force garbage collection
        import gc

        collected = gc.collect()
        self.logger.info(
            f"Memory cleanup completed. Garbage collected {collected} objects."
        )
