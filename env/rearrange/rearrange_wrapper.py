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
        
    def eval_state(self, goal_state, cur_state, threshold: float = 0.2, yaw_threshold: float = 0.2):
        """
        Evaluate agent and entity against the goal with separate thresholds for position and yaw.

        Inputs can be either:
        - Single state arrays (3|4,) or sequences (T, 3|4)  -> treated as AGENT-only (backward compatible)
        - Tuples/lists: (agent_state, entity_state) for both goal and current.
            Each element can be (3|4,) or (T, 3|4). Uses the last frame if a sequence.

        Success rules:
        - agent:  pos_dist(agent) <= threshold  AND  yaw_err(agent) <= yaw_threshold
        - entity: pos_dist(entity) <= threshold
        - overall 'success' = agent_success AND entity_success (when entity provided)

        Returns:
        {
            "success": bool,
            "distance": float,                # overall distance summary = max(agent_pos_dist, entity_pos_dist) when both provided; otherwise the one available
            "agent_distance": float or None,  # L2 over xyz
            "agent_yaw_error": float or None, # abs wrapped to [0,pi]
            "entity_distance": float or None, # L2 over xyz
            "agent_success": bool or None,
            "entity_success": bool or None,
        }
        """

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

        # unpack inputs: support (agent, entity) tuples or single arrays
        if isinstance(goal_state, (tuple, list)) and isinstance(cur_state, (tuple, list)) and len(goal_state) == 2 and len(cur_state) == 2:
            g_agent, g_entity = _to_last(goal_state[0]), _to_last(goal_state[1])
            c_agent, c_entity = _to_last(cur_state[0]), _to_last(cur_state[1])
            have_entity = True
        else:
            g_agent, c_agent = _to_last(goal_state), _to_last(cur_state)
            g_entity = c_entity = None
            have_entity = False

        # agent metrics
        agent_pos = _pos_dist(g_agent, c_agent)
        agent_yaw = _ang_diff(_yaw(g_agent), _yaw(c_agent))
        agent_success = (agent_pos <= float(threshold)) and (agent_yaw <= float(yaw_threshold))

        # entity metrics (if provided)
        if have_entity and g_entity is not None and c_entity is not None:
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
        # (x, y, z) for the target entity — yaw is intentionally not included
        e = self.unwrapped.entities[idx]
        p = np.asarray(getattr(e, "pos", (np.nan, np.nan, np.nan)), dtype=np.float32)
        # yaw = float(getattr(e, "dir", 0.0))  # intentionally ignored for evaluation
        return np.array([p[0], p[1], p[2]], dtype=np.float32)

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

        visual  = np.stack(frames, axis=0)
        proprio = np.zeros((visual.shape[0], getattr(self, "proprio_dim", 0)), dtype=np.float32)

        # build (T+1, d) trajectories
        agent_traj  = np.stack([a for (a, e) in states_list], axis=0)
        entity_traj = np.stack([e for (a, e) in states_list], axis=0)

        # pick ONE based on flag; default to "entity"
        src = getattr(self, "state_source", "entity")
        states = agent_traj if str(src).lower().startswith("agent") else entity_traj

        return {"visual": visual, "proprio": proprio}, states
