import numpy as np

from typing import Callable, Union, Tuple, Dict
from gymnasium import spaces

Schedule = Callable[[float], float]
def constant_fn(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: constant value
    :return: Constant schedule function.
    """

    def func(_):
        return val

    return func

def get_schedule_fn(value_schedule: Union[Schedule, float]) -> Schedule:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
    # Some types are have odd behaviors when part of a Schedule, like numpy floats
    return lambda progress_remaining: float(value_schedule(progress_remaining))

def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obses, acts, rews, nobses, terms = [], [], [], [], []
    # obses, acts, rews, nobses, terms, heights = [], [], [], [], [], []
    steps = 0
    while True:
        # if render:
        #     if 'rgb_array' in render_mode:
        #         if hasattr(env.unwrapped, sim):
        #             if 'track' in env.unwrapped.model.camera_names:
        #                 imgs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
        #             else:
        #                 imgs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])

        #     if 'human' in render_mode:
        #         env.render(mode=render_mode)
        #         time.sleep(env.model.opt.timestep)

        obses.append(obs)
        act = policy.get_action(obs)
        act = act[0]
        acts.append(act)
        nobs, rew, done, _, info = env.step(act)
        # heights.append(info['z_position'])
        nobses.append(nobs)
        rews.append(rew)
        obs = nobs.copy()
        steps += 1

        if done or steps > max_path_length:
            terms.append(1)
            break
        else:
            terms.append(0)

    return Path(obses, acts, rews, nobses, terms)
    # return Path_height(obses, heights, acts, rews, nobses, terms)


def sample_trajectories(
    env,
    policy,
    min_timesteps_per_batch,
    max_path_length,
    render=False,
    render_mode=("rgb_array"),
):
    # TODO: get this from hw1
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
        print(
            "sampled {}/{} timesteps".format(timesteps_this_batch, min_timesteps_per_batch),
            end="\r",
        )

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=("rgb_array")):
    # TODO: get this from hw1
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        print("sampled {}/ {} trajs".format(i, ntraj), end="\r")
    return paths


def Path(obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    return {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }

def Path_height(obs, height, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    return {
        "observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "height": np.array(height, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def get_pathlength(path):
    return len(path["reward"])
