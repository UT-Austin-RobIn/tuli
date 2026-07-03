import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robosuite.utils.observables import Observable, sensor

from .banging_env import BangingEnv, CONTACT_CUBE_TABLE


class HammeringEnv(BangingEnv):

    def __init__(
        self,
        reward_mode="target",
        target_pos=(0.0, 0.0),
        target_radius=0.03,
        target_height=0.005,
        repeat_sigma=0.05,
        **kwargs,
    ):
        assert reward_mode in ("target", "repeat"), reward_mode
        self._ham_reward_mode = reward_mode
        self._target_xy = np.array(target_pos, dtype=np.float64)
        self._target_radius = float(target_radius)
        self._target_height = float(target_height)
        self._repeat_sigma = float(repeat_sigma)

        self.target_geom_id = None
        self._last_strike_xy = np.zeros(2, dtype=np.float64)
        self._prev_strike_xy = None
        self._last_prox_factor = 0.0
        self._prox_mult      = 1.0
        self._accum_reward   = {}
        self._accum_table    = {}

        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        if self._ham_reward_mode != "target":
            return

        table_top_z = self.table_offset[2] + self.table_full_size[2] / 2.0
        tx, ty = self._target_xy
        tz = table_top_z + self._target_height

        target_body = ET.Element("body")
        target_body.set("name", "hammer_target_body")
        target_body.set("pos", f"{tx} {ty} {tz}")

        target_geom = ET.SubElement(target_body, "geom")
        target_geom.set("name", "hammer_target")
        target_geom.set("type", "cylinder")
        target_geom.set("size", f"{self._target_radius} {self._target_height}")
        target_geom.set("rgba", "0.9 0.2 0.2 1")
        target_geom.set("contype", "1")
        target_geom.set("conaffinity", "1")
        target_geom.set("group", "0")

        self.model.worldbody.append(target_body)

    def _setup_references(self):
        super()._setup_references()
        if self._ham_reward_mode == "target":
            self.target_geom_id = self.sim.model.geom_name2id("hammer_target")

    def _accumulate_contacts(self):
        reward_scan, table_scan = self._scan_ham()
        self._merge_peaks(self._accum_reward, reward_scan)
        self._merge_peaks(self._accum_table,  table_scan)

    def _scan_ham(self):
        reward_scan = {}
        table_scan  = {}
        if self.cube_geom_id is None:
            return reward_scan, table_scan

        reward_other = self.target_geom_id if self._ham_reward_mode == "target" \
            else self.table_geom_id

        contact_force = np.zeros(6, dtype=np.float64)
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            g1, g2  = contact.geom1, contact.geom2
            if self.cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_geom_id else g1
            pair  = (min(g1, g2), max(g1, g2))
            mujoco.mj_contactForce(
                self.sim.model._model, self.sim.data._data, i, contact_force
            )
            fn = float(abs(contact_force[0]))

            if reward_other is not None and other == reward_other:
                reward_scan[pair] = max(reward_scan.get(pair, 0.0), fn)
            if other == self.table_geom_id:
                table_scan[pair] = max(table_scan.get(pair, 0.0), fn)
        return reward_scan, table_scan

    def _detect_contacts(self):
        r_scan, t_scan = self._scan_ham()
        self._merge_peaks(self._accum_reward, r_scan)
        self._merge_peaks(self._accum_table,  t_scan)
        current_reward = self._accum_reward
        current_table  = self._accum_table

        for pair in set(self._outer_events) - set(current_reward):
            del self._outer_events[pair]

        self._total_outer_fn = 0.0
        self._new_contact_step = False
        for pair, fn in current_reward.items():
            if pair not in self._outer_events:
                self._outer_events[pair] = {"n_steps": 0}
                self._new_contact_step = True
            self._outer_events[pair]["n_steps"] += 1
            self._total_outer_fn += fn

        # air gate resets on ANY cube contact (table or target), not just reward contacts.
        in_contact_now = len(current_reward) > 0 or len(current_table) > 0
        if not self.use_air_gate:
            self._last_air_factor = 1.0
            if not in_contact_now:
                self._air_run += 1
            else:
                self._air_run = 0
        elif self._new_contact_step:
            self._last_air_factor = 1.0 if self._air_run >= self.min_air_steps else 0.0
            self._air_run = 0
            self._last_strike_xy = np.array(
                self.sim.data.body_xpos[self.cube_body_id][:2]
            )
        elif in_contact_now:
            self._last_air_factor = 0.0
            self._air_run = 0
        else:
            self._last_air_factor = 0.0
            self._air_run += 1

        if self._ham_reward_mode == "repeat":
            if self._new_contact_step:
                if self._prev_strike_xy is None:
                    prox = 1.0
                else:
                    d2 = float(np.sum((self._last_strike_xy - self._prev_strike_xy) ** 2))
                    prox = float(np.exp(-d2 / (2.0 * self._repeat_sigma ** 2)))
                self._last_prox_factor = prox
                self._prev_strike_xy   = self._last_strike_xy.copy()
                self._prox_mult        = prox
            else:
                self._prox_mult = 1.0
        else:
            self._last_prox_factor = 1.0 if self._new_contact_step else 0.0
            self._prox_mult        = 1.0

        self._current_contacts = {CONTACT_CUBE_TABLE: []}
        for pair, fn in current_table.items():
            self._current_contacts[CONTACT_CUBE_TABLE].append({
                "wall":         "cube",
                "pair":         pair,
                "normal_force": fn,
                "is_new":       False,
            })

        self._last_delta_fn = abs(self._total_outer_fn - self._prev_total_fn)
        self._prev_total_fn = self._total_outer_fn
        self._accum_reward  = {}
        self._accum_table   = {}

    def reward(self, action=None):
        if self._last_delta_fn < self.min_delta_fn:
            r = 0.0
        elif self.require_new_contact and not self._new_contact_step:
            r = 0.0
        else:
            r = float(self._last_delta_fn / self.reward_divisor)
        r *= self._last_air_factor
        r *= self._prox_mult
        return r

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        info["prox_factor"]  = float(self._last_prox_factor)
        info["strike_xy"]    = self._last_strike_xy.copy()
        info["reward_mode"]  = self._ham_reward_mode
        return reward, done, info

    def _setup_observables(self):
        observables = super()._setup_observables()
        modality = "hammer"

        @sensor(modality=modality)
        def hammer_target_xy(obs_cache):
            if self._ham_reward_mode == "target":
                return self._target_xy.astype(np.float32)
            if self._prev_strike_xy is None:
                return np.array(
                    self.sim.data.body_xpos[self.cube_body_id][:2],
                    dtype=np.float32,
                )
            return self._prev_strike_xy.astype(np.float32)

        observables["hammer_target_xy"] = Observable(
            name="hammer_target_xy",
            sensor=hammer_target_xy,
            sampling_rate=self.control_freq,
        )
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self._prev_strike_xy   = None
        self._last_strike_xy   = np.zeros(2, dtype=np.float64)
        self._last_prox_factor = 0.0
        self._prox_mult        = 1.0
        self._accum_reward     = {}
        self._accum_table      = {}
