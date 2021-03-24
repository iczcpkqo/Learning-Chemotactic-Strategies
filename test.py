from stable_baselines import PPO2

from config import args
from env import Env


if __name__ == "__main__":

    args.pattern = 'spire'
    model = PPO2.load('model/ppo_{}'.format(args.pattern))
    args.pygame = True

    env = Env(args)

    # Enjoy trained agent
    while True:
        obs, done = env.reset(), False

        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
