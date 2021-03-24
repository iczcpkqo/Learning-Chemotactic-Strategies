import gym
import numpy as np
import pygame
import math
# pygame config
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

## Parameter of 'theta' in the parametric equation of Archimedean spiral
## Degree of angle of change per unit of time, theta = 360 = 2 * math.pi
## It's different from theta, just per time theta
theta_per_time = 20 * (math.pi/180)

## Parameter of 'a' in the parametric equation of Archimedean spiral
## Radius = a , when theta = 0
a_arch_spire = -30

## Parameter of 'b' in the parametric equation of Archimedean spiral
## Growth of radius that from center to (x, y) of theta per unit
b_arch_spire = 5


def updatePosition(u, v: np.ndarray, var, dt):
    # integration position based on velocity, using a single Euler step of: d/dt position = velocity + noise

    velocity = dt * np.array(v)
    noise = np.sqrt(dt * var) * np.random.randn(2)
    delta_u = velocity + noise

    return u + delta_u


def whatQuad(x,y, maxx,maxy):
    if abs(x) > maxx or abs(y) > maxy or x == 0 or y == 0:
        return -1
    else:
        return int(x > 0) + 2*int(y > 0)

def calcv(quads, vr):
    xdir = np.sign(- quads[0] + quads[1] - quads[2] + quads[3])
    ydir = np.sign(- quads[0] - quads[1] + quads[2] + quads[3])
    return scaleVect(vr / np.sqrt(2), (xdir, ydir))

def scaleVect(s, v):
        return (s*v[0], s*v[1])

def getSpirePoint(idx, per, emit, time, center, a, b):
    # theta = per*(idx%count)/emit +
    rng = np.random.default_rng()
    theta = per * (time - (idx/emit))
    p_spire = archimedeanSpire(center, a, b, theta)

    if center[0]-p_spire[0] == 0:
        return p_spire
    # else:
    #     r = round(random.uniform(-30,30), 2)
    #     # r = rng.poisson(60) - 30
    #     # r = 1
    #     k = (center[1]-p_spire[1]) / (center[0]-p_spire[0])
    #     alpha = math.atan(k)
    #     return (p_spire[0] + r*math.cos(alpha), p_spire[1] + r*math.sin(alpha))
    else:

        r = np.sqrt(p_spire[0]**2 + p_spire[1]**2)
        r_1 = r/10
        r_2 = r_1/2
        # k = (center[1]-p_spire[1]) / (center[0]-p_spire[0])
        # alpha = math.atan(k)
        return (p_spire[0] + rng.poisson(r_1) - r_2, p_spire[1] + rng.poisson(r_1) - r_1)

def archimedeanSpire(center, a, b, theta):
    # 常数
    p = center[0]
    q = center[1]

    x_next = (a+b*theta)*math.cos(theta) + p
    y_next = (a+b*theta)*math.sin(theta) + q

    return x_next, y_next

class Env(gym.Env):
    # Set this in SOME subclasses
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, args):
        super(Env, self).__init__()

        self.useRl = args.useRl
        self.pattern = args.pattern

        self.timeStep = args.timeStep


        # absorption rate: chance of a particle being eaten when on agent for one unit of time
        self.eatRate = 0.

        # size of arena
        self.win_x = args.winx
        self.win_y = args.winy

        # size of agent
        self.agent_x = args.ax
        self.agent_y = args.ay

        self.test = args.test
        self.source_test_init = [args.sx, args.sy]

        # expected number of particles released per unit time
        self.releaseRate = args.emit

        # time step of simulation
        self.dt = args.dt

        # particle diffusion variance per unit time
        self.diffusionVariance = args.brown

        # initial position of agent (offset from upper left corner)
        self.u_init = np.array([args.ax0, args.ay0])

        self.u = self.u_init

        # chase velocity of agent
        self.vr = args.avel

        # Set these in ALL subclasses
        self.action_space = gym.spaces.box.Box(low= np.array([-1, -1], dtype=np.float32),
                                               high=np.array([ 1,  1], dtype=np.float32))

        self.observation_space = gym.spaces.box.Box(low= np.array([-self.agent_x / 2, -self.agent_y / 2], dtype=np.float32),
                                                    high=np.array([self.agent_x / 2,   self.agent_y / 2], dtype=np.float32))

        # display every i-th frame (default: %(default)s)
        self.frame = args.frame

        ## Initialize random number generator
        self.rng = np.random.default_rng()

        ## Initialize display (game) engine
        self.pygame = args.pygame

        if self.pygame:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.win_x, self.win_y))
            pygame.display.set_caption("Smelly the Chemotactic Agent")
            self.font = pygame.font.SysFont('Cambria', 40)


    def step(self, action: np.ndarray):

        self._updateParticles()

        dis_before = self._distance()

        # 通过rl选动作 或 原始方法计算动作
        action_exec =  action * self.vr if self.useRl else calcv(self.quads, self.vr)

        # Calculate the new coordinates of the agent
        self.u = np.clip(
            updatePosition(self.u, action_exec, 0.0, self.dt),
            a_min=0.0,
            a_max=[self.win_x - self.agent_x, self.win_y - self.agent_y]
        )

        dis_after = self._distance()

        next_obs, absorbPartsNum = self._get_obs()
        reward =  dis_before - dis_after
        done = self._terminal()
        info = {'absorbPartsNum': absorbPartsNum,
                'Time-consuming': self.t}

        # print('dis_before: {}\tdis_after: {}\treward: {}'.format(dis_before, dis_after, reward))
        # print('next_obs: {}\tact: {}\treward: {}'.format(next_obs, absorbPartsNum, action*self.vr, reward))

        # total time update
        self.t += self.dt

        return next_obs, reward, done, info

    def reset(self, eat=0, useRl=True):
        self.useRl = useRl
        self.particles = []
        self.t = 0.0

        # 重置 eat eat的修改放在外面
        self.eatRate = eat

        # source of particles (offset from lower right corner)
        if self.test:
            self.source = self.source_test_init
        else:
            self.source = [ np.random.rand() * self.win_x,
                            np.random.rand() * self.win_y]
        self.u = self.u_init

        # print(self.source)
        self.wind = self.u - self.source # 吹向 agent
        self.wind = self.wind * 5 / np.linalg.norm(self.wind) # normal

        return np.array([0, 0], dtype=np.float32)

    def render(self, mode='human'):
        if self.pygame:
            self.screen.fill(white)

            ## Draw the agent
            (u_x, u_y) = self.u.tolist()
            pygame.draw.rect(self.screen, green, (int(u_x), int(u_y), int(self.agent_x), int(self.agent_y)))
            # with blue racing stripes
            pygame.draw.rect(self.screen, blue, (int(u_x + self.agent_x / 2) - 1, int(u_y), 2, int(self.agent_y)))
            pygame.draw.rect(self.screen, blue, (int(u_x), int(u_y + self.agent_y / 2) - 1, int(self.agent_x), 2))

            ## Draw the odor source
            pygame.draw.circle(self.screen, blue, self.intPair(self.source), 5)


            ## Draw the particles
            for p in self.particles:
                pygame.draw.circle(self.screen, red, self.intPair(p), 2)
            self.screen.blit(self.font.render(' t = ' + str(round(self.t, 3)), False, (0, 0, 0)), (0, 0))

            my_ui = MyInterface(self.screen)
            my_ui.show()
            pygame.display.update()

    def close(self):
        pygame.quit()

    def seed(self, seed=None):
        pass

    def _agent_center(self):
        # 计算agent中心坐标
        return self.u + [self.agent_x/2, self.agent_y/2]

    def _distance(self):
        # 报酬函数
        distance = np.sqrt(
                        np.sum(
                            np.square(self._agent_center() - self.source)
                        )
                    )

        return distance

    def _updateParticles(self):

        # 更新离子  1, 新离子从source生成;  2, 离子消亡 由eatRate决定

        ## Maybe particles are born:
        self.particles += [self.source] * self.rng.poisson(self.releaseRate * self.dt)
        ## Initialize particle loop variables
        self.quads = [0, 0, 0, 0]

        newParticles = []

        for idx, p in enumerate(self.particles):
            # Update particle position
            if self.pattern == 'spire':
                p_x, p_y = p = np.array(
                    getSpirePoint(idx, theta_per_time, self.releaseRate, self.t,
                                  (self.source[0], self.source[1]), a_arch_spire, b_arch_spire)
                )
            else:
                p_x, p_y = p = updatePosition(p, self.wind, self.diffusionVariance, self.dt)

            u_x, u_y = self.u

            i = whatQuad(
                p_x - (u_x + self.agent_x / 2),
                p_y - (u_y + self.agent_y / 2),
                self.agent_x / 2, self.agent_y / 2
            )

            if i >= 0:
                self.quads[i] += 1

            newParticles.append(p.tolist())

        self.particles = newParticles

    def _terminal(self):
        # episode 终止判断. 1, agent到达source，任务结束; 2, agent出界 任务结束

        cent = self._agent_center()

        out_of_range  = self.u + [self.agent_x, self.agent_y] > [self.win_x, self.win_y]

        # print('to goal {}'.format(np.linalg.norm(cent - self.source)))
        if np.any( self.u < 0 ) or  np.any( out_of_range ) or np.linalg.norm( cent-self.source ) < 20 :
            # todo: 到达source的阈值需要调整
            # agent 出界 or agent 到达source
            return True
        elif self.t > self.dt * self.timeStep :
            # 环境最长时间步 3000
            return True
        else:
            return False

    def _get_obs(self):
        # 获得状态观测
        # return: 接触agent离子,相对agent中心的平均位置 [x, y], 吸收离子数量

        cent = self._agent_center()

        def _help(coor):
            # 判断 离子 是否碰到agent
            return np.all( np.abs( coor - cent ) < [self.agent_x / 2, self.agent_y / 2] )


        observed_particels = list(filter(_help, self.particles))



        # 离子碰到agent 会被吸收
        beforeAbsorbed = len(self.particles)

        absorbFilter = lambda coor: ~_help(coor)
        filted_particles = list(filter(absorbFilter, self.particles))
        # 按照 self.eatRate 的概率 吸收离子
        filted_particles += [p for p in observed_particels if np.random.rand() < self.eatRate]
        self.particles = filted_particles

        afterAbsorbed = len(self.particles)

        absorbNum = beforeAbsorbed - afterAbsorbed


        if len(observed_particels) == 0:
            # 没有接触到agent的离子
            return np.array([0, 0], dtype=np.float32), absorbNum
        else:
            observed_particels = np.array(observed_particels, dtype=np.float32)
            observed_particels = np.mean( observed_particels, axis=0 ) - self.u # 平均位置

            return observed_particels, absorbNum

    @classmethod
    def intPair(cls, p):
        return int(p[0]), int(p[1])
# My User Interface
class MyInterface:

    ## Init function
    def __init__(self, screen):
        self.screen = screen
        self.pattern = False
        self.color = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'black': ( 0, 0, 0),
            'gray':  ( 204, 204, 204)
        }

        self.font_small = pygame.font.SysFont('Cambria', 18)

        self.button_width = 130
        self.button_height = 40

        self.panel_width = 200
        self.panel_height = 500
        self.panel_x = 1000 - (self.panel_width + 20)
        self.panel_y = 20



    def button(self, msg, order, action=None ,width=None, height=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        width = width or self.button_width
        height = height or self.button_height
        button_position = self.get_button_slot(order)

        if button_position[0] + width > mouse[0] > button_position[0] and button_position[1] + height > mouse[1] > button_position[1]:
            pygame.draw.rect(self.screen, self.color['red'], (button_position[0], button_position[1], self.button_width, self.button_height))
            if click[0] == 1 and action != None:
                action()
        else:
            pygame.draw.rect(self.screen, self.color['gray'], (button_position[0], button_position[1], self.button_width, self.button_height))

        text = self.font_small.render(msg, True, self.color['black'])
        textPosition = text.get_rect()
        textPosition.center = (button_position[0]+self.button_width/2, button_position[1]+self.button_height/2)
        self.screen.blit(text, textPosition)


    # Button of spire
    def button_spire(self):
        # print('Spire particle')
        self.pattern = 'spire'

    # Button of particle decay
    def button_decay(self):
        # print('Decay particle')
        self.pattern = 'decay'

    def get_button_slot(self, order, button_width=None, button_height=None):
        button_width = button_width or self.button_width
        button_height = button_height or self.button_height
        button_x = self.panel_x + self.panel_width - 20 - button_width
        button_y = order*(button_height + 20) + 20
        return (button_x, button_y)

    def show(self):
        self.button('Spire particle', 0, self.button_spire)
        self.button('Decay particle', 1, self.button_decay)
        return self.pattern

# My User Interface END
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    from config import args

    args.pygame = True

    env = Env(args)

    obs, done = env.reset(), False

    while not done:
        a = env.action_space.sample()
        obs, reward, done, _ = env.step( a )
        env.render()
