import numpy as np


def sample_trajectory(
    env, policy, max_path_length, render=False, render_mode=("rgb_array")
):
    obs = env.reset()
    obses, acts, rews, nobses, terms, imgs = [], [], [], [], [], []
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
        nobs, rew, done, _ = env.step(act)
        nobses.append(nobs)
        rews.append(rew)
        obs = nobs.copy()
        steps += 1

        if done or steps > max_path_length:
            terms.append(1)
            break
        else:
            terms.append(0)

    return Path(obses, imgs, acts, rews, nobses, terms)


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
            "sampled {}/{} timesteps".format(
                timesteps_this_batch, min_timesteps_per_batch
            ),
            end="\r",
        )

    return paths, timesteps_this_batch


def sample_n_trajectories(
    env, policy, ntraj, max_path_length, render=False, render_mode=("rgb_array")
):
    # TODO: get this from hw1
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        print("sampled {}/ {} trajs".format(i, ntraj), end="\r")
    return paths


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def get_pathlength(path):
    return len(path["reward"])
