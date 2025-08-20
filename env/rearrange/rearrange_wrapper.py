import numpy as np
from miniworld.envs.jeparoom import RearrangeOneRoom  # Adjust import path
from utils import aggregate_dct  # Assumes this exists like in pusht
import math
import gymnasium as gym
import random


if not hasattr(gym.wrappers.TimeLimit, "__getattr__"):
    def _forward_to_inner(self, name):
        return getattr(self.env, name)        # delegate to wrapped env
    gym.wrappers.TimeLimit.__getattr__ = _forward_to_inner

class RearrangeOneRoomWrapper(RearrangeOneRoom):
    def __init__(self, size=2, seed=0, max_entities=4, **kwargs):
        super().__init__(size=size, seed=seed, max_entities=max_entities, **kwargs)
        self.action_dim = 1
        self.proprio_dim = 1
        self.target_name = "object"
        self.ent_idx = None
        self.state_source = "entity"

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as init, one as goal.
        You need to define what "state" means for your task.
        This example uses agent (x, z, yaw) only.
        """
        pass
    
    def reset(self, *, seed=None, options=None):
        # keep seed behavior explicit & deterministic
        if seed is None:
            seed = getattr(self, "seed", None)

        if seed is not None:
            seed = int(seed)
            # your custom RNGs used by _gen_world()
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        # helpful debug line; for subprocess workers use flush or file logging
        #print(f"[Wrapper.reset] seed={seed}", flush=True)

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _resolve_target_index(self):
      cls, idx_s = self.target_name.split("_", 1)
      idx = int(idx_s)-1
      ents = getattr(self.unwrapped, "entities", [])
      if not (0 <= idx < len(ents)):
          print(f"[warn] target_name {self.target_name}: index {idx} out of range (len={len(ents)})")
      elif ents[idx].__class__.__name__ != cls:
          print(f"[warn] target_name {self.target_name}: entities[{idx}] is {ents[idx].__class__.__name__}, expected {cls}")
      self.ent_idx = idx
      


    def update_env(self, env_info):
        """Reset env using the dataset seed (prefer master_seed, fallback to seed)."""
        self.target_name = env_info.get("object")
        self.seed = env_info.get("seed")
        print(f"updated env with: {self.target_name} and {self.seed}")
        
    def eval_state(self, goal_state, cur_state, threshold: float = 0.2, yaw_threshold: float = 0.3):
        def _to_last(arr):
            a = np.asarray(arr, dtype=np.float32)
            return a[-1] if a.ndim > 1 else a

        def _pos_dist(a, b):
            return float(np.linalg.norm(a[:3] - b[:3]))

        def _yaw(a):
            return float(a[3]) if a.shape[0] >= 4 else 0.0

        def _ang_diff(a, b):
            # wrap to [-pi, pi]
            d = (a - b + np.pi) % (2 * np.pi) - np.pi
            return abs(d)

        # NEW: unify unpacking to also handle stacked arrays with leading size 2
        def _unpack(x):
            # case 1: (agent, entity) as tuple/list
            if isinstance(x, (tuple, list)) and len(x) == 2:
                return _to_last(x[0]), _to_last(x[1])
            # case 2: stacked ndarray with leading axis=2 trajectories: (2, ..., d)
            xa = np.asarray(x)
            if xa.ndim >= 2 and xa.shape[0] == 2:
                return _to_last(xa[0]), _to_last(xa[1])
            # case 3: single agent state/trajectory
            return _to_last(xa), None

        g_agent, g_entity = _unpack(goal_state)
        c_agent, c_entity = _unpack(cur_state)
        have_entity = (g_entity is not None) and (c_entity is not None)

        # agent metrics
        agent_pos = _pos_dist(g_agent, c_agent)
        agent_yaw = _ang_diff(_yaw(g_agent), _yaw(c_agent))
        agent_success = (agent_pos <= float(threshold)) and (agent_yaw <= float(yaw_threshold))

        # entity metrics (if provided)
        if have_entity:
            entity_pos = _pos_dist(g_entity, c_entity)
            entity_success = (entity_pos <= float(threshold))
        else:
            entity_pos = None
            entity_success = None

        # overall success & distance summary
        if entity_pos is not None:
            overall_success = bool(agent_success and entity_success)
            overall_distance = max(agent_pos, entity_pos)
        else:
            overall_success = bool(agent_success)
            overall_distance = agent_pos

        return {
            "success": overall_success,
            "distance": overall_distance,
            "agent_distance": agent_pos,
            "agent_yaw_error": agent_yaw,
            "entity_distance": entity_pos,
            "agent_success": agent_success,
            "entity_success": entity_success,
        }




    def prepare(self, seed, init_state):
        """
        Resets to a specific agent state
        """
        pass

    def step_multiple(self, actions):
        print("Stepping with actions:", actions)
    
    def get_agent_state(self):
        # (x, y, z, yaw)
        p = np.asarray(getattr(self.agent, "pos", (0.0, 0.0, 0.0)), dtype=np.float32)
        yaw = float(getattr(self.agent, "dir", 0.0))
        return np.array([p[0], p[1], p[2], yaw], dtype=np.float32)

    def get_entity_state(self, idx: int):
        # (x, y, z, dummy_yaw) — yaw is fixed 0.0 for entities to match agent state dim
        e = self.unwrapped.entities[idx]
        p = np.asarray(getattr(e, "pos", (np.nan, np.nan, np.nan)), dtype=np.float32)
        return np.array([p[0], p[1], p[2], 0.0], dtype=np.float32)


    def rollout(self, _seed_unused, _init_state_unused, actions):
        states_list = []

        first, _ = self.reset(seed=self.seed)
        self._resolve_target_index()
        if self.ent_idx is None:
            raise RuntimeError(f"Bad target_name '{self.target_name}'")

        def to_chw01(img):
            img = np.asarray(img)
            if img.ndim == 2:
                img = img[..., None]
            if img.shape[-1] in (1, 3, 4):
                img = img.astype(np.float32) / 255.0
                img = img.transpose(2, 0, 1)
            return img

        frames = [to_chw01(first)]

        # initial states: (agent, entity)
        states_list.append((self.get_agent_state(), self.get_entity_state(self.ent_idx)))

        A = np.asarray(actions)
        for t in range(A.shape[0]):
            a = int(A[t]) if isinstance(self.action_space, gym.spaces.Discrete) else np.asarray(A[t], dtype=np.float32)
            obs, reward, terminated, truncated, info = self.step(a)
            frames.append(to_chw01(obs))

            states_list.append((self.get_agent_state(), self.get_entity_state(self.ent_idx)))

            if terminated or truncated:
                break

        visual  = np.stack(frames, axis=0)  # (T+1, C, H, W)
        proprio = np.zeros((visual.shape[0], getattr(self, "proprio_dim", 0)), dtype=np.float32)

        # build (T+1, d) trajectories
        agent_traj  = np.stack([a for (a, e) in states_list], axis=0).astype(np.float32)
        entity_traj = np.stack([e for (a, e) in states_list], axis=0).astype(np.float32)

        # stack into (2, T+1, d): [0]=agent, [1]=entity
        rollout_states = np.stack([agent_traj, entity_traj], axis=0)

        return {"visual": visual, "proprio": proprio}, rollout_states


