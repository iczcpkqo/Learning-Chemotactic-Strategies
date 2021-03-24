import os
from multiprocessing import cpu_count

from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from config import args
from env import Env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":

    for pattern in ['spire', 'decay']:
        args.pattern = pattern
        env_fn = lambda: Env(args)

        env = SubprocVecEnv([env_fn] * cpu_count())
        model = PPO2('MlpPolicy', env=env, seed=0, verbose=2, gamma=0.9,
                     learning_rate=3e-4, nminibatches=4, n_steps=1024*2,
                     cliprange_vf=-1, max_grad_norm=0.5,
                     tensorboard_log='log/ppo2')

        model.learn(total_timesteps=30_0000, reset_num_timesteps=False)
        model.save('model/ppo_{}'.format(args.pattern))
