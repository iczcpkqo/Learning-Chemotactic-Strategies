import math

import numpy as np
import pygame
from stable_baselines import PPO2

from config import args
from env import Env

# a velocity based on differential odors (highly viscous medium)
def calcv(quads, vr):
    xdir = np.sign(- quads[0] + quads[1] - quads[2] + quads[3])
    ydir = np.sign(- quads[0] - quads[1] + quads[2] + quads[3])
    return scaleVect(vr / np.sqrt(2), (xdir, ydir))


def updatePosition(u, v, var, dt, rng):
    # integration position based on velocity, using a single Euler step of: d/dt position = velocity + noise
    return sumVect(u, sumVect(scaleVect(dt, v),
                              scaleVect(np.sqrt(dt * var), (rng.standard_normal(), rng.standard_normal()))))


def whatQuad(x, y, maxx, maxy):
    if abs(x) > maxx or abs(y) > maxy or x == 0 or y == 0:
        return -1
    else:
        return int(x > 0) + 2 * int(y > 0)


def wind(p, t):
    return (-1.2, -0.3)


def scaleVect(s, v):
    return (s * v[0], s * v[1])


def sumVect(u, v):
    return (u[0] + v[0], u[1] + v[1])


def intPair(p):
    return (int(p[0]), int(p[1]))


def getSpirePoint(idx, per, emit, time, center, a, b):
    # theta = per*(idx%count)/emit +  
    rng = np.random.default_rng()
    theta = per * (time - (idx / emit))
    p_spire = archimedeanSpire(center, a, b, theta)

    if center[0] - p_spire[0] == 0:
        return p_spire
    # else:
    #     r = round(random.uniform(-30,30), 2)
    #     # r = rng.poisson(60) - 30
    #     # r = 1
    #     k = (center[1]-p_spire[1]) / (center[0]-p_spire[0])
    #     alpha = math.atan(k)
    #     return (p_spire[0] + r*math.cos(alpha), p_spire[1] + r*math.sin(alpha))
    else:

        r = np.sqrt(p_spire[0] ** 2 + p_spire[1] ** 2)
        r_1 = r / 10
        r_2 = r_1 / 2
        # k = (center[1]-p_spire[1]) / (center[0]-p_spire[0])
        # alpha = math.atan(k)
        return (p_spire[0] + rng.poisson(r_1) - r_2, p_spire[1] + rng.poisson(r_1) - r_1)


def archimedeanSpire(center, a, b, theta):
    # 常数
    p = center[0]
    q = center[1]

    x_next = (a + b * theta) * math.cos(theta) + p
    y_next = (a + b * theta) * math.sin(theta) + q

    return (x_next, y_next)



 # =========== [ Parameter Menu ] ========== #
i_eat_start = 0.0   # eat在程序开始时的最小值
i_eat_end   = 1.   # eat在程序结束时的的最大值
i_eat_step  = 0.2  # eat在每个固定eat的循环完成后 每次增加的值
i_run_times = 1   # 每个固定eat的循环次数


if __name__ == "__main__":
    # current_pattern = args.pattern
    args.pattern = current_pattern = 'decay' # 'decay'  'spire'  里面挑一个
    args.useRl = False
    args.pygame = True
    switch_pattern_flag = False
    args.test = True

    eat = i_eat_start

    if current_pattern == 'spire':
        args.sx = 500
        args.sy = 300
        model = PPO2.load('model\ppo_spire')
    elif current_pattern == 'decay':
        args.sx = 100
        args.sy = 100
        model = PPO2.load('model\ppo_decay')
    else:
        args.sx = 200
        args.sy = 500
        model = PPO2.load('model\ppo_decay')

    # log 结果
    import pandas as  pd
    result = pd.DataFrame(columns=['eat', 'absorbPartsNum', 'Time-consuming'])

    env = Env(args)
    while eat <= i_eat_end:
        eat += i_eat_step
        for step in range(i_run_times):
            obs, done = env.reset(eat=eat, useRl=False), False

            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)

                """
                info 里面有 每一步的统计数据:
                'absorbPartsNum':    该步吸收离子数
                'Time-consuming':    总耗时
                """
                info.update({'eat': eat})
                result = result.append([info], ignore_index=False)
                env.render()
    result.to_csv('result.csv', index=False)