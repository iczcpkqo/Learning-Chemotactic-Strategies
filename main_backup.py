#!/usr/bin/python3

import random
import argparse
# no pygame
import pygame
import numpy as np
# no pygame
from pygame.locals import *
import datetime
# no simple agent
# no import agent.simple_agent as sp_a
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import particle.simple_particle as sp_p
# import maze.office as mz_of
import math
# no databox
# no import calculate.databox as cal_db
from decimal import Decimal
# import pycuda.autoinit
# from pycuda.compiler import SourceModule

# Class

class SimpleAgent:
    # Base data
    color_body = (0, 255, 0)
    color_grain = (0, 0, 255)

    # Screen data
    SCREEN = None

    # Args data
    is_init_args = False
    __eatRate = None
    agent_x = None
    agent_y = None
    u = None
    u_x = None
    u_y = None
    vr = None
    source = None
    win_x = None
    win_y = None
    releaseRate = None
    diffusionVariance = None
    dt = None
    frame = None
    u_start = None
    u_recoder = None

    # General data
    quad = None
    time_box = None
    steps = None
    sch = None
    is_touch_me = None
    ## How much time pass
    runtime = None

    # Calculative data
    ## Lateral offset from target
    bias_target = None
    ## Lateral offset from last step
    bias_step = None
    ## The distance of each step from the source
    distance = None
    ## How far is each step forward
    forward_step = None

    def __init__(self, switch_cartoon, screen, args):
        # args.ax0 += random.randint(-40, 40)
        # args.ay0 += random.randint(-40, 40)
        if switch_cartoon:
            self.init_yeacartoon(screen, args)
        elif not switch_cartoon:
            self.init_nocartoon(args)

    # No cartoon
    def init_nocartoon(self, args):
        self.init_args(args)
        self.init_general()
        self.init_cal()

    # yea cartoon
    def init_yeacartoon(self, screen, args):
        self.init_screen(screen)
        self.init_args(args)
        self.init_general()
        self.init_cal()

    # Init screen
    def init_screen(self, screen):
        self.SCREEN = screen

    # Init args
    def init_args(self, args):
        self.__eatRate = args.eat
        self.agent_x = args.ax
        self.agent_y = args.ay
        self.u = [args.ax0, args.ay0]
        self.u_x = args.ax0
        self.u_y = args.ay0
        self.vr = args.avel
        self.source = (args.sx, args.sy)
        self.win_x = args.winx
        self.win_y = args.winy
        self.releaseRate = args.emit
        self.diffusionVariance = args.brown
        self.dt = args.dt
        self.frame = args.frame
        self.u_start = (self.u_x, self.u_y)
        self.u_recoder = [self.u_start]
        self.is_init_args = True
        self.source = (args.winx - self.source[0], args.winy - self.source[1])

    # Init source
    def init_source(self, source):
        return None

    # Init general data
    def init_general(self):
        self.quad = [0, 0, 0, 0]
        self.time_box = [0]
        self.steps = 0
        self.sch = True
        self.is_touch_me = False
        # How much time pass
        self.runtime = 0

    # Init calculative data
    def init_cal(self):
        # Lateral offset from target
        self.bias_target = [0]
        # Lateral offset from last step
        self.bias_step = [0]
        # How far is each step forward
        self.forward_step = [0]
        if self.is_init_args:
            # The distance of each step from the source
            self.distance = [math.hypot(self.source[0] - self.u[0], self.source[1] - self.u[1])]

    # Setter
    ## Set eat rate
    def set_eat(self, i_eat):
        self.__eatRate = i_eat

    # Getter
    def get_eat(self):
        return self.__eatRate

    # Updater
    ## Update draw
    # no pygame
    def update_agent(self):
        pygame.draw.rect(self.SCREEN, self.color_grain,
                         (int(self.u_x), int(self.u_y), int(self.agent_x), int(self.agent_y)))
        # With blue racing stripes
        pygame.draw.rect(self.SCREEN, self.color_body,
                         (int(self.u_x + self.agent_x / 2) - 1, int(self.u_y), 2, int(self.agent_y)))
        pygame.draw.rect(self.SCREEN, self.color_body,
                         (int(self.u_x), int(self.u_y + self.agent_y / 2) - 1, int(self.agent_x), 2))

    ## Update bounds
    def bounds(self):
        self.u_x = np.clip(0.0, self.u_x, self.win_x - self.agent_x)
        self.u_y = np.clip(0.0, self.u_y, self.win_y - self.agent_y)

        self.u = [self.u_x, self.u_y]

    # Calculative data

    ## Update position
    def set_position(self, u_x, u_y):
        # Trigger every time the position changes
        if self.u_x != u_x or self.u_y != u_y:
            self.is_touch_me = True
        # After the first trigger, every frame is triggered
        if self.is_touch_me:
            self.u_recoder.append((u_y, u_y))
            self.u_x = u_x
            self.u_y = u_y
            self.u = [self.u_x, self.u_y]

            ### Temp delete ###
            # self.distance.append(math.hypot(self.source[0] - self.u[0], self.source[1] - self.u[1]))
            # self.bias_step.append(self.point_distance_line(self.u_recoder[len(self.u_recoder) - 1],
            #                                                self.u_recoder[len(self.u_recoder) - 2],
            #                                                self.source))
            # self.bias_target.append(self.point_distance_line(self.u_recoder[len(self.u_recoder) - 1],
            #                                                  self.u_start,
            #                                                  self.source))
            # self.forward_step.append(self.distance[len(self.distance) - 1] - self.distance[len(self.distance) - 2])
            # self.time_box.append(self.time_box[len(self.time_box) - 1] + 1)
            ###  ###
            self.add_steps(1)

        # if (len(self.time_box)==501 and ):
        # print(len(self.time_box))
        # print(self.sch)

        # if (len(self.time_box)==201 and self.sch):
        #     print("dfsfsdf")
        #     del self.u_recoder[0:200]
        #     del self.distance[0:200]
        #     del self.bias_step[0:200]
        #     del self.bias_target[0:200]
        #     del self.forward_step[0:200]
        #     del self.time_box[0:200]
        #     self.time_box[0] = 0
        #     # for i in range(len(self.time_box)):
        #     #     self.time_box[i] = i
        #     self.sch = False
        #     plt.clf() #清空画布上的所有内容
        #     print(self.time_box)
        #     print(self.bias_target)

    def point_distance_line(self, point, line_point1, line_point2):
        # 计算向量
        vec1 = (line_point1[0] - point[0], line_point1[1] - point[1])
        vec2 = (line_point2[0] - point[0], line_point2[1] - point[1])
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(
            (line_point1[0] - line_point2[0], line_point1[1] - line_point2[1]))
        return distance

    def is_arrive(self):
        if self.is_init_args:
            # The distance of each step from the source
            now_distance = math.hypot(self.source[0] - self.u[0], self.source[1] - self.u[1])
            if now_distance < 60:
                return True
        return False

    def set_runtime(self, runtime):
        self.runtime = runtime

    def add_steps(self, i_step):
        self.steps += i_step

    def get_times(self):
        return len(self.time_box)

    def get_runtime(self):
        return self.runtime

    def get_steps(self):
        return self.steps

# # lateral offset from target
# self.bias_target = [0]
#
# # lateral offset from last step
# self.bias_step = [0]
#
# # The distance of each step from the source
# self.distance = [math.hypot(self.source[0] - self.u[0], self.source[1] - self.u[1])]

class SingleData:
    # Data, just for one agent
    ## Eat of this agent
    __eat = None

    ## How many times the agent has moved
    __step = None

    ## Run time
    __time = None

    ## Did the agent successfully reach the goal?
    __arrive_successfully = None

    # Set data, setter
    ## Set eat
    def set_eat(self, eat):
        self.__eat = eat

    ## Set step
    def set_step(self, step):
        self.__step = step

    ## Set time
    def set_time(self, time):
        self.__time = time

    ## Set arrive or not
    def set_arrive(self, is_arrive):
        self.__arrive_successfully = is_arrive

    # Get the data
    ## Get eat
    def get_eat(self):
        return self.__eat

    ## Get step
    def get_step(self):
        return self.__step

    ## Get run time
    def get_time(self):
        return self.__time

    ## Get arrive or not
    def get_arrive(self):
        return self.__arrive_successfully

    # Some function
    ## Take all the data by one time when this agent finish her work.
    def now_finish(self, is_arrive, i_eat, i_time, i_step):
        self.set_arrive(is_arrive)
        self.set_eat(i_eat)
        self.set_time(i_time)
        self.set_step(i_step)


class DataBox:
    # Data box for putting all singledata in.

    # Init
    ## Init function
    def __init__(self):
        self.init_general()
        self.init_steps()
        self.init_caldata()

    # Init data
    ## Init general data
    def init_general(self):
        # General
        ## Eat.old
        self.eats = []
        ## Record every data .old
        self.data_box = []
        # New data
        ## Eat box,
        self.__eats = []
        ## save all the singledata
        self.__databox = []

    ## Init step data
    def init_steps(self):
        # Step data
        ## How many steps should evey singleData cost
        self.total_steps_group = []
        ## Save avg steps of every singleData
        self.avg_steps = []
        ## Save variance of every group of singleData which have same eat.
        self.variance_steps = []
        # new data
        # step data
        self.__avg_steps = []
        self.__variance_steps = []

    ## Init time
    def init_time(self):
        # time data
        self.total_seconds_group = []
        self.avg_seconds = []
        self.variance_time = []

        # new data
        self.__avg_time = []
        self.__variance_time = []

    ## Init some data about calculate
    def init_caldata(self):
        # Calculate data
        ## Make a name as a ID for Databox
        self.id_box = self.create_data_id()
        self.arrive_times = []
        self.success_rate = []

        # new data
        # Calculate data
        ## Make a name as a ID for Databox
        self.__id_box = self.create_data_id()
        self.__success_times = []
        self.__success_rate = []
        self.__total_times = []

    # Add
    ## Add a eat .old
    def add_eat(self, eat):
        self.eats.append(eat)

    ## Add a result
    def add_data(self, single_data):
        if single_data.get_eat() not in self.__eats:
            self.__eats.append(single_data.get_eat())
            self.__databox.append([])
        self.__databox[self.__eats.index(single_data.get_eat())].append(single_data)

    # Add steps
    ## Add step.old
    def add_step(self, step):
        while len(self.total_steps_group) < len(self.eats):
            self.total_steps_group.append([])
        self.total_steps_group[len(self.eats) - 1].append(step)

    ## .old
    def add_success(self):
        while len(self.arrive_times) < len(self.eats):
            self.arrive_times.append([0, 0])
        self.arrive_times[len(self.eats) - 1][0] += 1
        self.arrive_times[len(self.eats) - 1][1] += 1

    ## .old
    def add_fail(self):
        while len(self.arrive_times) < len(self.eats):
            self.arrive_times.append([0, 0])
        self.arrive_times[len(self.eats) - 1][1] += 1

    ## Get the average running time corresponding to each absorption rate
    def get_avg_steps(self):
        # return self.avg_steps
        return self.__avg_steps

    ## Get the variance corresponding to each absorption rate
    def get_variance(self):
        # return self.variance_steps
        return self.__variance_steps

    ## Get the success rate corresponding to each absorption rate
    def get_success_rate(self):
        # return self.success_rate
        return self.__success_rate

    # calculate data
    ## Calculate the average running time of each absorption rate under the current data
    def cal_avg_time(self):
        while len(self.__avg_time) < len(self.__databox):
            self.__avg_time.append(0)
        for i in range(len(self.__databox)):
            i_sum_time = 0
            for j in range(len(self.__databox[i])):
                i_sum_time += j.get_time()
            self.__avg_time[i] = i_sum_time / len(self.__databox[i])

    ## Calculate the average number of runs corresponding to each absorption rate under the current data
    def cal_avg_steps(self):
        ### old
        # while len(self.avg_steps) < len(self.eats):
        #     self.avg_steps.append(0)
        # for i in range(len(self.total_steps_group)):
        #     i_total_steps = 0
        #     for j in self.total_steps_group[i]:
        #         i_total_steps += j
        #     self.avg_steps[i] = i_total_steps / len(self.total_steps_group[i])
        # return self.avg_steps
        ### new
        while len(self.__avg_steps) < len(self.__databox):
            self.__avg_steps.append(0)
        for i in range(len(self.__databox)):
            i_sum_steps = 0
            for j in range(len(self.__databox[i])):
                i_sum_steps += j.get_step()
            self.__avg_steps[i] = i_sum_steps / len(self.__databox[i])

    ## Calculate the variance of the arrival time in each group.old
    def cal_variance(self):
        self.cal_avg_steps()
        while len(self.variance_steps) < len(self.eats):
            self.variance_steps.append(0)
        for i in range(len(self.total_steps_group)):
            i_variance = 0
            for j in self.total_steps_group[i]:
                i_variance += (j - self.avg_steps[i]) ** 2
            self.variance_steps[i] = i_variance / len(self.total_steps_group[i])
        return self.variance_steps

    ## Calculate the variance of the arrival time in each group
    def cal_variance_time(self):
        self.cal_avg_time()
        while len(self.__variance_time) < len(self.__databox):
            self.__variance_time.append(0)
        for i in range(len(self.__databox)):
            i_variance = 0
            for j in self.__databox[i]:
                i_variance += (j.get_time() - self.__avg_time[i]) ** 2
            self.__variance_time[i] = i_variance / len(self.__databox[i])
        return self.__variance_time

    ## Calculate the variance of the number of arrivals in each group
    def cal_variance_steps(self):
        self.cal_avg_steps()
        while len(self.__variance_steps) < len(self.__databox):
            self.__variance_steps.append(0)
        for i in range(len(self.__databox)):
            i_variance = 0
            for j in self.__databox[i]:
                i_variance += (j.get_step() - self.__avg_steps[i]) ** 2
            self.__variance_steps[i] = i_variance / len(self.__databox[i])
        return self.__variance_steps

    ## Calculate the success rate corresponding to each absorption rate (the goal can be reached within 1000 steps)
    def cal_success_rate(self):
        # while len(self.success_rate) < len(self.eats):
        #     self.success_rate.append(0)
        # for i in range(len(self.arrive_times)):
        #     self.success_rate[i] = self.arrive_times[i][0] / self.arrive_times[i][1]
        # return self.success_rate
        while len(self.__success_rate) < len(self.__databox):
            self.__success_rate.append(0)
        for i in range(len(self.arrive_times)):
            i_success = 0
            i_total = 0
            for j in self.__databox[i]:
                i_total += 1
                if j.get_arrive() == True:
                    i_success += 1
            self.__success_rate[i] = i_success / i_total
        return self.__success_rate

    ## Calculate all statistics
    def cal_all(self):
        # old
        self.cal_variance()
        self.cal_success_rate()
        self.cal_avg_steps()

        # new
        self.cal_avg_steps()
        self.cal_avg_time()
        self.cal_variance_steps()
        self.cal_variance_time()
        self.cal_success_rate()

    ## Determine whether the incoming agent has reached the source [Which agent, did it succeed?]. old
    def arrive(self, a_agent, is_success):
        step = a_agent.get_runtime()
        if is_success:
            self.add_step(step)
            self.add_success()
        else:
            self.add_step(0)
            self.add_fail()

    ## Make a .txt name also ID
    def create_data_id(self):
        self.now = datetime.datetime.now()
        return str(self.now.year) + '-' + str(self.now.month) + '-' + str(self.now.day) + '_' + str(self.now.hour) + '.' + str(self.now.minute) + '.' + str(self.now.second)
        # Get a random name
        ## name_a = ['two', 'three', 'four', 'many', 'some']
        ## name_big = ['tiny', 'puny', 'wee', 'weeny', 'small', 'light', 'mild', 'gentle', 'ethereal', 'heavy', 'fat',
        ##             'black', 'white', 'green', 'blue', 'expensive']
        ## name_thing = ['cats', 'dogs', 'cars', 'apples', 'oranges', 'pears', 'classmates']
        ## return str(name_a[random.randint(0, len(name_a) - 1)]) + " " + str(
        ##     name_big[random.randint(0, len(name_big) - 1)]) + " " + str(
        ##     name_thing[random.randint(0, len(name_thing) - 1)])

    ## Print some data I want to see
    def print_databox(self):
        print('========================================')

        print('eats:', len(self.eats))
        print('--------------------')
        print(self.eats)
        print('----------------------------------------')

        print('')

        print('total_steps_group', len(self.total_steps_group))
        print('--------------------')
        print(self.total_steps_group)
        print('----------------------------------------')

        print('')

        print('========================================')

    # Data stored in file
    ## Save all data.old
    def save_as_txt(self):
        file = open(self.id_box + ".txt", "w")
        # file.write("eat  |  time\n")
        for i in range(len(self.eats)):
            for j in range(len(self.total_steps_group)):
                for k in range(len(self.total_steps_group[j])):
                    file.write(str(self.eats[i]) + "\t" + str(self.total_steps_group[j][k]) + "\n")
        file.close()

    ## Save the latest piece of data.old
    def savenew_as_txt(self):
        file = open(self.id_box + ".txt", "a")
        file.write(str(self.eats[-1]) + "\t" + str(self.total_steps_group[-1][-1]) + "\n")
        file.close()

    ## Save the time spent in each run
    def save_databox_time(self):
        file = open(self.id_box + ".txt", "w")
        # file.write("eat  |  time\n")
        for i in self.__databox:
            for j in i:
                file.write(str(j.get_eat()) + "\t" + str(j.get_time()) + "\n")
        file.close()

    ## Save the time spent in a run
    def save_singledata_time(self, single_data):
        file = open(self.id_box + ".txt", "a")
        # file.write("eat  |  time\n")
        file.write(str(Decimal(single_data.get_eat()).quantize(Decimal("0.00"))) + "\t" + str(Decimal(single_data.get_time()).quantize(Decimal("0.00"))) + "\n")
        file.close()


def main():
    ## Process command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-debug", help="debug level (default: %(default)s)", type=int, default=0)
    parser.add_argument("-eat", help="agent's particle absorption rate (default: %(default)s)", type=float,
                        default=0.25)
    parser.add_argument("-ax", help="agent width (default: %(default)s)", type=float, default=50.0)
    parser.add_argument("-ay", help="agent height (default: %(default)s)", type=float, default=50.0)
    parser.add_argument("-ax0", help="agent initial position down (default: %(default)s)", type=float, default=250.0)
    parser.add_argument("-ay0", help="agent initial position right (default: %(default)s)", type=float, default=250.0)
    parser.add_argument("-avel", help="agent chase velocity (default: %(default)s)", type=float, default=5.0)
    # parser.add_argument("-sx", help="source offset left from LR (default: %(default)s)", type=float, default=100.0)
    # parser.add_argument("-sy", help="source offset up from LR (default: %(default)s)", type=float, default=100.0)
    parser.add_argument("-sx", help="source offset left from LR (default: %(default)s)", type=float, default=500.0)
    parser.add_argument("-sy", help="source offset up from LR (default: %(default)s)", type=float, default=300.0)
    parser.add_argument("-winx", help="arena width (default: %(default)s)", type=int, default=1000)
    parser.add_argument("-winy", help="arena height (default: %(default)s)", type=int, default=700)
    parser.add_argument("-emit", help="particle emission rate (default: %(default)s)", type=float, default=5.0)
    parser.add_argument("-brown", help="particle diffusion variance (default: %(default)s)", type=float, default=5.0)
    parser.add_argument("-dt", help="simulation timestep (default: %(default)s)", type=float, default=0.3)
    parser.add_argument("-frame", help="display every i-th frame (default: %(default)s)", type=int, default=1)

    # args = parser.parse_args()
    args = parser.parse_args(args=[])  # 原来是opt=parser.parse_args()

    # debug level
    debug = args.debug

    if debug > 0:
        print(args)

    # absorption rate: chance of a particle being eaten when on agent for one unit of time
    eatRate = args.eat

    # size of arena
    win_x = args.winx
    win_y = args.winy

    # size of agent
    agent_x = args.ax
    agent_y = args.ay

    # expected number of particles released per unit time
    releaseRate = args.emit

    # time step of simulation
    dt = args.dt

    # particle diffusion variance per unit time
    diffusionVariance = args.brown

    # initial position of agent (offset from upper left corner)
    u = [args.ax0, args.ay0]

    # chase velocity of agent
    vr = args.avel

    # source of particles (offset from lower right corner)
    source = (args.sx, args.sy)

    frame = args.frame

    ## Update derived quantities
    source = (win_x - source[0], win_y - source[1])

    print("source:", source)
    if debug > 0:
        print("source:", source)

    # ============ [ On-Off Menu ] ============ #
    switch_cartoon = True        # Draw cartoon #
    switch_graph   = False       # Draw graph   #
    ## The name of this pattern, which pattern will be execute #
    pattern_name = 'spire'
    # =========== [ Parameter Menu ] ========== #
    i_eat_start = 0.21   # first eat of loop     #
    i_eat_end   = 0.25   # last eat of loop      #
    i_eat_step  = 0.02  # increase of each time #
    i_run_times = 1   # run times of each eat #
    # ===== [ Archimedean Spire Menu ] ======== #
    ## Is chose Archimedean Spire or not
    switch_arch_spire = False


    ## Parameter of 'a' in the parametric equation of Archimedean spiral
    ## Radius = a , when theta = 0
    a_arch_spire = -30

    ## Parameter of 'b' in the parametric equation of Archimedean spiral
    ## Growth of radius that from center to (x, y) of theta per unit
    b_arch_spire = 5

    ## Parameter of 'theta' in the parametric equation of Archimedean spiral
    ## Degree of angle of change per unit of time, theta = 360 = 2 * math.pi
    ## It's different from theta, just per time theta
    theta_per_time = 5
    theta_per_time = theta_per_time * (math.pi/180)

    ## The center of spire
    ## Spire center can be different from source center
    x_center_spire = source[0]
    y_center_spire = source[1]
    # ========================================= #
    # ===== [ Archimedean Spire Menu ] ======== #
    # Direct numerical simulation
    # Reynolds-averaged Navier–Stokes equations RANS
    # ========================================= #

    ## Initialize random number generator
    rng = np.random.default_rng()

    screen = None
    white = None
    black = None
    red = None
    green = None
    blue = None

    number_eats = 0
    number_single_data = 0

    # no pygame
    if switch_cartoon:
        ## Initialize display (game) engine
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((win_x, win_y))
        pygame.display.set_caption("Smelly the Chemotactic Agent")
        font = pygame.font.SysFont('Cambria', 40)

        ## Create My User Interface
        my_ui = MyInterface(screen)
        ui_clicked = ''

        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)


    ## Initialize a DataBox
    # no databox
    # db_box = cal_db.DataBox()
    db_box = DataBox()

    # This loop just for receiving the order button click for change pattern
    while True:
        my_ui.pattern = my_ui.pattern or pattern_name
        ui_clicked = my_ui.pattern
        pattern_name = my_ui.pattern

        if (pattern_name == 'spire'):
            args.sx = 500
            args.sy = 300
            source = (args.sx, args.sy)
            source = (win_x - source[0], win_y - source[1])
            pass
        elif (pattern_name == 'decay'):
            args.sx = 100
            args.sy = 100
            source = (args.sx, args.sy)
            source = (win_x - source[0], win_y - source[1])
            pass
        else:
            args.sx = 200
            args.sy = 500
            source = (args.sx, args.y)
            source = (win_x - source[0], win_y - source[1])
            pass

        ## range of eat - x
        for i_eat in np.arange(i_eat_start, i_eat_end, i_eat_step):
            number_eats += 1
            # db_box.add_eat(i_eat)
            ## run times of the agent
            for run_times in range(i_run_times):
                number_single_data += 1

                # Initialize temportal loop variables
                particles = []
                t = 0.0
                framei = 0
                steps = 0

                # Initialize a Agent
                # no agent
                # no a_agent = sp_a.SimpleAgent(switch_cartoon, screen, args)
                a_agent = SimpleAgent(switch_cartoon, screen, args)
                a_agent.set_eat(i_eat)
                ## Initialize a single data for agent
                # no databox
                # each_data = cal_db.SingleData()
                each_data = SingleData()

                # Initialize Many Agent
                agent_number = 0
                agent_box = []
                for i in range(0, agent_number):
                    # no agent
                    # no tp_agent = sp_a.SimpleAgent(switch_cartoon, screen, args)
                    tp_agent = SimpleAgent(switch_cartoon, screen, args)
                    tp_agent.set_eat(i_eat)
                    agent_box.append(tp_agent)

                # ii = 0
                # time_box = [0]

                while True:
                    if framei == 0:
                        True;
                        ##  Time to die?

                        # no pygame
                        if switch_cartoon is True:
                            for event in pygame.event.get():
                                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                                    pygame.quit()

                            ## Run, run, as fast as you can. You can't catch me, I'm the Stinky Cheese Man!
                            ## So comment this out:
                            # pygame.time.delay(100)

                            screen.fill(white)

                            ## Draw the agent
                            # (u_x,u_y) = u
                            # pygame.draw.rect(screen, green, (int(u_x), int(u_y), int(agent_x), int(agent_y)))
                            # # with blue racing stripes
                            # pygame.draw.rect(screen, blue, (int(u_x + agent_x/2)-1, int(u_y), 2, int(agent_y)))
                            # pygame.draw.rect(screen, blue, (int(u_x), int(u_y + agent_y/2)-1, int(agent_x), 2))

                            a_agent.update_agent()

                            for i_agent in agent_box:
                                i_agent.update_agent()

                            ## Draw the odor source
                            pygame.draw.circle(screen, blue, intPair(source), 5)


                    ## Maybe particles are born: | 泊松分布，具体每一步的运算结果是什么？
                    particles += [source] * rng.poisson(releaseRate * dt)

                    # if ii < 10:
                    #     print(particles)
                    #     ii+=1

                    if debug > 1:
                        print("particles =", particles)

                    if debug > 0:
                        print("#particles:", len(particles))

                    ## Initialize particle loop variables
                    for i_agent in agent_box:
                        i_agent.quad = [0, 0, 0, 0]

                    quads = [0, 0, 0, 0]
                    newParticles = []

                    for idx, p in enumerate(particles):

                        if pattern_name == 'spire':
                            # get spire point
                            (p_x, p_y) = p = p_spire = getSpirePoint(idx, theta_per_time, releaseRate, t, (x_center_spire, y_center_spire), a_arch_spire, b_arch_spire)
                        elif pattern_name == 'decay':
                            # Update particle position
                            (p_x, p_y) = p = updatePosition(p, wind(p, t), diffusionVariance, dt, rng)
                        else:
                            pygame.quit()

                        # (p_x, p_y) = p = ((p[0]+p_spire[0])/2, (p[1]+p_spire[1])/2)

                        # no pygame
                        if switch_cartoon is True:
                            if framei == 0:
                                pygame.draw.circle(screen, red, intPair(p), 2)

                        i = whatQuad(p_x - (a_agent.u_x + a_agent.agent_x / 2), p_y - (a_agent.u_y + a_agent.agent_y / 2),
                                     a_agent.agent_x / 2, a_agent.agent_y / 2)
                        p_survive = -1
                        for i_agent in agent_box:
                            p_touch = whatQuad(p_x - (i_agent.u_x + i_agent.agent_x / 2),
                                               p_y - (i_agent.u_y + i_agent.agent_y / 2),
                                               i_agent.agent_x / 2, i_agent.agent_y / 2)
                            if p_touch >= 0:
                                p_survive = 0
                                i_agent.quad[p_touch] += 1

                        if debug > 2:
                            print("agent", a_agent.u, "particle", p, "quad", i)
                        if i >= 0:
                            quads[i] += 1

                        if p_y < 0 or p_y > win_y or p_x < 0 or p_x > win_x:
                            if debug > 2:
                                print("Not adding: particle at", p, "off screen")
                        elif a_agent.get_eat() == 0 or (i < 0 and p_survive < 0) or rng.exponential(
                                1 / a_agent.get_eat()) >= dt:
                            newParticles.append(p)
                        else:
                            if debug > 2:
                                print("Not adding: particle at", p, "quad", i, "eaten")

                    particles = newParticles

                    if debug > 0:
                        print("quads:", quads)

                    # Calculate the new coordinates of the agent
                    # (a_agent.u_x, a_agent.u_y) = updatePosition(a_agent.u, calcv(quads, a_agent.vr), 0.0, dt, rng)
                    u_tmp = updatePosition(a_agent.u, calcv(quads, a_agent.vr), 0.0, dt, rng)
                    a_agent.set_position(u_tmp[0], u_tmp[1])

                    for i_agent in agent_box:
                        (i_agent.u_x, i_agent.u_y) = updatePosition(i_agent.u, calcv(i_agent.quad, i_agent.vr), 0.0, dt,
                                                                    rng)

                    if debug > 0:
                        if a_agent.u_x < 0 or a_agent.u_y < 0 or a_agent.u_x + a_agent.agent_x > win_x or a_agent.u_y + a_agent.agent_y > win_y:
                            print("Agent out of bounds")

                    a_agent.bounds()
                    for i_agent in agent_box:
                        i_agent.bounds()

                    # no pygame
                    if switch_cartoon is True:
                        if framei == 0:
                            # Update screen
                            screen.blit(
                                font.render(' t = ' + str(round(t, 3)) + ', steps = ' + str(a_agent.get_times()), False,
                                            (0, 0, 0)), (0, 0))

                            ## Draw My User Interface
                            ui_clicked = my_ui.show()


                            pygame.display.update()

                    if a_agent.is_touch_me:
                        t += dt

                    framei = np.mod(framei + 1, frame)

                    if a_agent.is_arrive() is True or t > 1000:
                        a_agent.set_runtime(t)
                        each_data.now_finish(a_agent.is_arrive(),
                                             a_agent.get_eat(),
                                             a_agent.get_runtime(),
                                             a_agent.get_steps())
                        del a_agent
                        break

                    steps += 1
                    if steps % 100 == 0:
                        print("i_eat: "+str(i_eat)+" | run_times: "+str(run_times)+" | steps: " + str(steps))

                    # Is the partern changed by clicking button
                    if switch_cartoon is True and ui_clicked !=  pattern_name and ui_clicked is not False:
                        break

                # Is the partern changed by clicking button
                if switch_cartoon is True and ui_clicked !=  pattern_name and ui_clicked is not False:
                    break

                print("eats: " + str(number_eats) + ", single data: " + str(number_single_data))
                # db_box.add_data(each_data)
                db_box.save_singledata_time(each_data)
                del each_data

                if switch_graph is True:
                    # draw some table
                    db_box.cal_all()
                    # db_box.save_as_txt()
                    db_box.savenew_as_txt()
                    plt.ion()  # 开启interactive mode 成功的关键函数
                    plt.figure("some graph as you see")

                    # average time of each eat
                    plt.subplot(2, 2, 1)
                    plt.title("avg arrive time")
                    plt.xlabel("eat rate")
                    plt.ylabel("avg time")
                    plt.grid(True)
                    plt.plot(db_box.eats, db_box.avg_steps, '-g', lw=1)

                    # probability of arrive no more than 1000time-step of each eat
                    plt.subplot(2, 2, 2)
                    plt.title("probability of arrive in 1000")
                    plt.xlabel("eat rate")
                    plt.ylabel("probability")
                    plt.grid(True)
                    plt.plot(db_box.eats, db_box.success_rate, '-g', lw=1)

                    # variance of time of each eat
                    plt.subplot(2, 2, 3)
                    plt.title("arrive time variance")
                    plt.xlabel("eat rate")
                    plt.ylabel("variance")
                    plt.grid(True)
                    plt.plot(db_box.eats, db_box.variance_steps, '-g', lw=1)

                    plt.pause(0.001)
                    # clear memory
                    plt.clf()  # clear

            # Is the partern changed by clicking button
            if switch_cartoon is True and ui_clicked != pattern_name and ui_clicked is not False:
                break

        # Is the partern changed by clicking button
        if switch_cartoon is True and ui_clicked != pattern_name and ui_clicked is not False:
            continue

        # no pygame
        if switch_cartoon is True:
           pygame.quit()


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

    return (x_next, y_next)



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
    main()
