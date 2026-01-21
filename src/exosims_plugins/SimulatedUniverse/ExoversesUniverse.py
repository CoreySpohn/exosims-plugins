import astropy.units as u
import numpy as np
from astropy.time import Time

from exosims_plugins.SimulatedUniverse.OrbixUniverse import OrbixUniverse


class ExoversesUniverse(OrbixUniverse):
    def __init__(self, *args, **specs):
        super().__init__(*args, **specs)

    def gen_physical_properties(self, **specs):
        universe = specs.get("exoverses_universe")
        PPop = self.PlanetPopulation
        TL = self.TargetList

        sc_df = self.TargetList.StarCatalog.data
        ignored_systems = []
        keep_sInds = []
        system_to_HPIC_name = {}
        HPIC_name_to_exoverses_name = {}
        for i, system in enumerate(universe.systems):
            nPlanets = len(system.planets)
            # Get the HIP number of the star
            HIP = float(system.star.name.split(" ")[1])
            # Match it to the star name in the target list
            matching_hips = sc_df.loc[sc_df["hip_name"] == HIP]
            if len(matching_hips) == 0:
                print(f"Star {HIP} not found in HPIC catalog")
                ignored_systems.append(system.star.name)
                continue
            elif len(matching_hips) > 1:
                # Check if one of the stars is in the TargetList
                in_TL = matching_hips["star_name"].isin(TL.Name)
                if ~np.any(in_TL):
                    print(f"No stars found in TargetList for {system.star.name}")
                    ignored_systems.append(system.star.name)
                    continue
                # Choose the star with the closest distance to the exoverses system
                dists = np.abs(
                    matching_hips["sy_dist"].values - system.star.dist.to_value(u.pc)
                )
                closest_ind = np.argmin(dists)
                matching_hips = matching_hips.iloc[closest_ind]
                TL_name = matching_hips["star_name"]
            else:
                # Only one matching star, so use it
                TL_name = matching_hips["star_name"].item()
            # Get the sInd
            if TL_name in TL.Name:
                sInd = np.argwhere(TL.Name == TL_name)[0][0]
            else:
                print(f"Star {TL_name} not found in EXOSIMS target list")
                ignored_systems.append(system.star.name)
                continue
            system_to_HPIC_name[system.star.name] = TL_name
            HPIC_name_to_exoverses_name[TL_name] = system.star.name
            keep_sInds.append(sInd)
        # Update the target list to only be the stars we're using
        # Note that this changes the sInds so it must be done before the
        # plan2star array is created
        TL.revise_lists(np.array(keep_sInds))

        # Get the number of planets for each system in the exoverses universe
        plan2star = np.array([], dtype=int)
        for i, system in enumerate(universe.systems):
            nPlanets = len(system.planets)
            if system.star.name not in system_to_HPIC_name:
                # Star not kept
                continue
            TL_name = system_to_HPIC_name[system.star.name]
            sInd = np.argwhere(TL.Name == TL_name)[0][0]
            plan2star = np.hstack((plan2star, [sInd] * nPlanets))

        self.plan2star = plan2star.astype(int)
        self.sInds, first_inds = np.unique(self.plan2star, return_index=True)
        self.nPlans = len(self.plan2star)
        self.I = np.zeros(self.nPlans) * u.deg
        self.O = np.zeros(self.nPlans) * u.deg
        self.w = np.zeros(self.nPlans) * u.deg
        self.a = np.zeros(self.nPlans) * u.AU
        self.e = np.zeros(self.nPlans)
        self.Rp = np.zeros(self.nPlans) * u.earthRad
        self.Mp = np.zeros(self.nPlans) * u.earthMass
        self.p = np.zeros(self.nPlans)
        self.M0 = np.zeros(self.nPlans) * u.deg
        abs_planet_ind = 0
        for star_ind in self.sInds[np.argsort(first_inds)]:
            # In cases where there are stars without planets we need to work
            # key specifically on the star name to avoid incorrect indexing
            system_name = TL.Name[star_ind]
            if system_name in ignored_systems:
                continue
            system = [
                s
                for s in universe.systems
                if HPIC_name_to_exoverses_name[system_name] == s.star.name
            ][0]
            if len(system.planets) == 0:
                # No planets in the system
                print(f"No planets in system {system.star.name}")
                continue
            for planet in system.planets:
                self.I[abs_planet_ind] = planet.inc.to(u.deg)
                self.O[abs_planet_ind] = planet.W.to(u.deg)
                self.w[abs_planet_ind] = planet.w.to(u.deg)
                self.a[abs_planet_ind] = planet.a.to(u.AU)
                self.e[abs_planet_ind] = planet.e
                self.Rp[abs_planet_ind] = planet.radius.to(u.earthRad)
                self.Mp[abs_planet_ind] = planet.mass.to(u.earthMass)
                self.p[abs_planet_ind] = planet.p
                self.M0[abs_planet_ind] = planet.mean_anom(
                    Time(specs["missionStart"], format="mjd").ravel()
                ).to(u.deg)
                abs_planet_ind += 1
        self.phiIndex = np.ones(self.nPlans, dtype=int) * 2
        ZL = self.ZodiacalLight
        if self.commonSystemnEZ:
            # Assign the same nEZ to all planets in the system
            self.nEZ = ZL.gen_systemnEZ(TL.nStars)[self.plan2star]
        else:
            # Assign a unique nEZ to each planet
            self.nEZ = ZL.gen_systemnEZ(self.nPlans)
