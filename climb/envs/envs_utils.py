from gymnasium.envs.registration import register, registry


def register_custom_envs(env_name):
    assert env_name in [
        "Climber-v0",
    ]
    if env_name not in registry:
        register(
            id=env_name,
            entry_point="climb.envs.mujoco.climber:Climber",
            max_episode_steps=1000,
            reward_threshold=200,
        )
