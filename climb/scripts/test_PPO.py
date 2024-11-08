import gymnasium as gym
import os
import time

from gymnasium.wrappers import RecordVideo
from climb.agents.ppo import PPOAgent
from climb.infrastructure.rl_trainer import RL_Trainer


env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = RecordVideo(env, "videos/", episode_trigger=lambda x: True)  # Save every episode

obs = env.reset()



class PPO_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        estimate_advantage_args = {
            'gamma': params['gamma'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
        }

        train_args = {
            'batch_size': params['batch_size'],
            'init_act_std': params['init_act_std'],
            'max_grad_norm': params['max_grad_norm'],
            'epoch': params['epoch'],
            'ent_coef': params['ent_coef'],
            'vf_coef': params['vf_coef'],
            'clip_range': params['clip_range']
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PPOAgent
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop()


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Humanoid-v5')
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--max_training_timesteps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--epoch', type=int, default=100)


    parser.add_argument('--eval_batch_size', '-b', type=int, default=1000) 
    parser.add_argument('--batch_size', type=int, default=1000) 

    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--init_act_std', type=float, default=0.6)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--gae_lambda', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    # for policy gradient, we made a design decision
    # to force batch_size = train_batch_size
    # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = PPO_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
