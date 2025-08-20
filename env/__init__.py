import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id="rearrange",
    entry_point="env.rearrange.rearrange_wrapper:RearrangeOneRoomWrapper",
    kwargs={"size": 12},
    max_episode_steps=250,
)

