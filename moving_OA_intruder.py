import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from shapely.geometry import Polygon
from shapely.geometry import Point
from gym.envs.classic_control import rendering
import pandas as pd


#######################origins for test##############################

def randomorigin(shape):
    # ori = []
    # for i in range(0, 40):
    #     ori.append([0, i*10])
    #     ori.append([400, i*10])
    #     ori.append([i * 10, 0])
    #     ori.append([i * 10, 400])
    #
    # ori = np.asarray(ori)
    # np.savetxt('origintest1.txt', ori)
    # return ori
    
    aa = [(0, np.random.randint(shape)), (shape, np.random.randint(shape)),
            (np.random.randint(shape), 0), (np.random.randint(shape), shape)]
    a = range(0, 4)
    o = random.choice(a)
    return aa[o]


#########################################################################

testorigin = np.loadtxt('origintest1.txt')

class SingleIntruderEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.window_width = 200
        self.window_height = 200
        origin = np.random.randint(self.window_width, size=2)
        goal = (150, 150)
        speed = 2
        heading = np.arctan2((goal[1] - origin[1]), (goal[0] - origin[0]))
        self.origin = origin
        self.goal = goal
        self.speed = speed
        self.heading = heading

        self.safedis = 15
        self.dt = 1
        self.max_speed = 25
        self.min_speed = 15

        self.shape = (self.window_width, self.window_height)
        self.goal_radius = 20

        self.viewer = None

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


        self.observation_space = spaces.Box(low=np.zeros(26), #38
            high=np.ones(26), dtype=np.float32)

        self.state_image = np.zeros((self.window_width, self.window_height, 3), dtype='f')

        self.test = False

        self.testcount = 0
        self.testfailcount = 0
        self.testdonecount = 0
        self.fail = 0
        self.doneC = 0
        self.failorigin = []
        self.seed()
        self.ccc = 0
        self.dis = []
        self.failpp = []

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0])
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1])
        coord[1] = max(coord[1], 0)
        return coord

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        # x, y, v, heading
        self.vmatrix = np.array(
            [self.speed * self.dt * math.cos(self.heading), self.speed * self.dt * math.sin(self.heading)])
        self.vframe = np.dot(self.Rmatrix, self.vmatrix)
        self.goalframe = np.dot(self.Rmatrix, self.goal) - np.dot(self.Rmatrix, self.state)
        #s = []

        rank = [0, 1, 2]

        self.intruder1p = self.IntruPP[rank[0]]
        self.intruder2p = self.IntruPP[rank[1]]
        self.intruder3p = self.IntruPP[rank[2]]


        self.intruder1v = self.Intruvv[rank[0]]
        self.intruder2v = self.Intruvv[rank[1]]
        self.intruder3v = self.Intruvv[rank[2]]

#
        self.intruder1pframe = np.dot(self.Rmatrix, self.intruder1p) - np.dot(self.Rmatrix, self.state)
        self.intruder2pframe = np.dot(self.Rmatrix, self.intruder2p) - np.dot(self.Rmatrix, self.state)
        self.intruder3pframe = np.dot(self.Rmatrix, self.intruder3p) - np.dot(self.Rmatrix, self.state)


        self.intruder1vframe = np.dot(self.Rmatrix, self.intruder1v) - np.dot(self.Rmatrix, self.state)
        self.intruder2vframe = np.dot(self.Rmatrix, self.intruder2v) - np.dot(self.Rmatrix, self.state)
        self.intruder3vframe = np.dot(self.Rmatrix, self.intruder3v) - np.dot(self.Rmatrix, self.state)

        s = np.zeros(26)
        nn = 8
        #
        ############## binary #############
        #ownship
        s[0] = dist(self.state, self.goal) / self.window_width / np.sqrt(2)
        s[1] = self.vframe[0] / 2
        s[2] = self.vframe[1] / 2
        s[3] = self.goalframe[0] / self.window_width / np.sqrt(2)
        s[4] = self.goalframe[1] / self.window_width / np.sqrt(2)

        du = self.window_width * np.sqrt(2) / nn


        ############## no binary #############
        s[5] = self.intruder1pframe[0] / self.window_width #/ np.sqrt(2)
        s[6] = self.intruder1pframe[1] / self.window_width #/ np.sqrt(2)
        s[7] = self.intruder1vframe[0] / 4
        s[8] = self.intruder1vframe[0] / 4
        s[9] = dist(self.state, self.intruder1p) / self.safedis

        s[10] = self.intruder2pframe[0] / self.window_width #/ np.sqrt(2)
        s[11] = self.intruder2pframe[1] / self.window_width #/ np.sqrt(2)
        s[12] = self.intruder2vframe[0] / 4
        s[13] = self.intruder2vframe[0] / 4
        s[14] = dist(self.state, self.intruder2p) / self.safedis

        s[15] = self.intruder3pframe[0] / self.window_width #/ np.sqrt(2)
        s[16] = self.intruder3pframe[1] / self.window_width #/ np.sqrt(2)
        s[17] = self.intruder3vframe[0] / 4
        s[18] = self.intruder3vframe[0] / 4
        s[19] = dist(self.state, self.intruder3p) / self.safedis

        s[20] = np.subtract(self.vmatrix, self.intruder1v)[0] / 4
        s[21] = np.subtract(self.vmatrix, self.intruder1v)[1] / 4

        s[22] = np.subtract(self.vmatrix, self.intruder2v)[0] / 4
        s[23] = np.subtract(self.vmatrix, self.intruder2v)[1] / 4

        s[24] = np.subtract(self.vmatrix, self.intruder3v)[0] / 4
        s[25] = np.subtract(self.vmatrix, self.intruder3v)[1] / 4

        return s

    def obs_scale(self, s):
        s2 = []

        s2.append(s[0] / self.window_width)
        s2.append(s[1] / self.window_height)
        s2.append(s[2] / (2 * np.pi))


        return np.array(s2)

    def reset(self):  # reset(self, origin1):
        self.timestep = 0
        self.dis0 = 400
        self.fail = 0
        self.doneC = 0
        self.goal =  (100, 200)
        if self.test == True:
            self.origin = (75+np.random.randint(60), np.random.randint(25))
            self.testcount += 1
        else:
            self.origin = (75+np.random.randint(60), np.random.randint(25))

        self.state = self.origin
        self.heading = np.arctan2((self.goal[1] - self.origin[1]), (self.goal[0] - self.origin[0]))
        theta = np.arctan2(self.goal[1] - self.origin[1], self.goal[0] - self.origin[0])
        self.Rmatrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        self.Intruori, self.Intruvv = GenIntruder(self.window_width, 3, self.state)

        self.IntruPP = 1.0*self.Intruori

        self.ccc += 1
        print(self.ccc)

        if self.test == True and self.testdonecount == 0:
            self.iii = 0
            self.intruder1recordp = np.zeros([200, 2])
            self.intruder2recordp = np.zeros([200, 2])
            self.intruder3recordp = np.zeros([200, 2])
            self.ownrecordp = []
            self.intruder1recordp[self.iii, :] = self.IntruPP[0]
            self.intruder2recordp[self.iii, :] = self.IntruPP[1]
            self.intruder3recordp[self.iii, :] = self.IntruPP[2]
            self.ownrecordp.append(self.origin)

        return self._get_obs()

    def step(self, action):
        self.timestep += 1
        self.heading = self.heading + action * 0.52

        vx = self.speed * self.dt * math.cos(self.heading)
        vy = self.speed * self.dt * math.sin(self.heading)
        self.velocity = np.array([vx, vy])

        theta = np.arctan2(self.goal[1] - self.state[1], self.goal[0] - self.state[0])
        self.Rmatrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        self.state = np.add(self.state, self.velocity)

        for i in range(len(self.IntruPP)):
            self.IntruPP[i] = np.add(1.0*self.IntruPP[i], 1.0*self.Intruvv[i])

        self.intruder1p = self.IntruPP[0]
        self.intruder2p = self.IntruPP[1]
        self.intruder3p = self.IntruPP[2]

        if  self.test == True and self.testdonecount == 0:
            self.iii += 1
            self.intruder1recordp[self.iii, :] = self.IntruPP[0]
            self.intruder2recordp[self.iii, :] = self.IntruPP[1]
            self.intruder3recordp[self.iii, :] = self.IntruPP[2]
            self.ownrecordp.append(self.state)

        reward = -0.007 * dist(self.state, self.goal) -0.15 + 17 * (np.arctan(0.1 * (dist(self.state, self.intruder1p) - 12)) - np.pi / 2) + 17 * (np.arctan(0.1 * (dist(self.state, self.intruder2p) - 12)) - np.pi / 2) + 17 * (np.arctan(0.1 * (dist(self.state, self.intruder3p) - 12)) - np.pi / 2)
        d1 = dist(self.state, self.intruder1p)
        d2 = dist(self.state, self.intruder2p)
        d3 = dist(self.state, self.intruder3p)
        dist_intruder = min(d1, d2, d3)

        done = False
        point = Point(self.state)


        self.dis0 = min(self.dis0,  dist(self.state, self.intruder1p), dist(self.state, self.intruder2p), dist(self.state, self.intruder3p))  #


        ######################## Intruder #########################################
        if dist(self.state, self.goal) >= self.goal_radius:
            if  dist(self.state, self.intruder2p) <= self.safedis or dist(self.state, self.intruder1p) <= self.safedis \
                    or dist(self.state, self.intruder3p) <= self.safedis : #or dist(self.state, self.intruder4p) <= 25:
                reward = reward - 180 #500 #-180

                if self.test == True:
                    self.fail = 1
                    self.failpp.append(self.state)
                    self.failpp.append(self.intruder1p)
                    self.failpp.append(self.intruder2p)
                    self.failpp.append(self.intruder3p)
                    self.failpp.append(dist(self.state, self.intruder1p))
                #done = True
                    #print(self.failpp)
                    #print(self.state, self.intruder1p, self.intruder2p, self.intruder3p, dist(self.state, self.intruder1p))


        if dist(self.state, self.goal) < self.goal_radius or self.timestep == 450:
            reward = 1000
            done = True
            if self.test == True:
                self.doneC = 1
            print('done')

        if done or self.timestep == 450:
            if  self.test == True and self.testdonecount == 0:
                self.intruder1recordp = np.asarray(self.intruder1recordp)
                self.intruder2recordp = np.asarray(self.intruder2recordp)
                self.intruder3recordp = np.asarray(self.intruder3recordp)
                self.ownrecordp = np.asarray(self.ownrecordp)

            if self.test == True:
                self.testfailcount += self.fail
                print(self.testfailcount)
                self.testdonecount += self.doneC
                print(self.testdonecount)
                self.dis.append(self.dis0)
                if self.fail == 1:
                    self.failorigin.append(self.origin)

        if int((self.testcount-1)/2) == 500:
            self.dis = np.asarray(self.dis)
            np.savetxt('325mindisr1192.txt', self.dis,  delimiter=', ', fmt='%1.3f')
            np.savetxt('325mindisr119failpp2.txt', self.failpp,  delimiter=', ')

        return self._get_obs(), reward/100, done, {}

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(3*self.window_width, 3*self.window_height)
            self.viewer.set_bounds(-50, self.window_width+50, -50, self.window_height+50)

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        ownship_img = rendering.Image(os.path.join(__location__, 'images/aircraft.png'), 15, 15)
        jtransform = rendering.Transform(rotation=self.heading - math.pi / 2, translation=self.state)
        ownship_img.add_attr(jtransform)
        ownship_img.set_color(255, 241, 4)  # set it to yellow
        self.viewer.onetime_geoms.append(ownship_img)

        # draw goal
        goal_img = rendering.Image(os.path.join(__location__, 'images/goal.png'), 15, 15)
        jtransform = rendering.Transform(rotation=0, translation=self.goal)
        goal_img.add_attr(jtransform)
        goal_img.set_color(15, 210, 81)  # green
        self.viewer.onetime_geoms.append(goal_img)

        # draw intruder
        for i in range(len(self.IntruPP)-2, -1, -1):
            intruder_img = rendering.Image(os.path.join(__location__, 'images/intruder0.png'), 10, 10)
            jtransform = rendering.Transform(rotation=np.arctan2(self.Intruvv[i][1], self.Intruvv[i][0]) - math.pi / 2,
                                              translation=self.IntruPP[i])
            intruder_img.add_attr(jtransform)
            intruder_img.set_color(255, 241, 4)  # set it to yellow
            self.viewer.onetime_geoms.append(intruder_img)


        intruder1_img = rendering.Image(os.path.join(__location__, 'images/intruder1.png'), 10, 10)
        jtransform1 = rendering.Transform(rotation=np.arctan2(self.intruder1v[1], self.intruder1v[0]) - math.pi / 2 , translation=self.intruder1p)
        intruder1_img.add_attr(jtransform1)
        intruder1_img.set_color(255, 241, 4)  # set it to yellow
        self.viewer.onetime_geoms.append(intruder1_img)

        intruder2_img = rendering.Image(os.path.join(__location__, 'images/intruder2.png'), 10, 10)
        jtransform2 = rendering.Transform(rotation=np.arctan2(self.intruder2v[1], self.intruder2v[0]) - math.pi / 2,
                                          translation=self.intruder2p)
        intruder2_img.add_attr(jtransform2)
        intruder2_img.set_color(255, 241, 4)  # set it to yellow
        self.viewer.onetime_geoms.append(intruder2_img)
        #
        intruder3_img = rendering.Image(os.path.join(__location__, 'images/intruder3.png'), 10, 10)
        jtransform3 = rendering.Transform(rotation=np.arctan2(self.intruder3v[1], self.intruder3v[0]) - math.pi / 2,
                                          translation=self.intruder3p)
        intruder3_img.add_attr(jtransform3)
        intruder3_img.set_color(255, 241, 4)  # set it to yellow
        self.viewer.onetime_geoms.append(intruder3_img)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')  # (return_rgb_array=False)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def boundVelocity(ownp, intruderp, safedis, speed):
    ownp = np.asarray(ownp)
    intruderp = np.asarray(intruderp)
    x1 = np.asarray(0.5 * (ownp + intruderp))
    x2 = intruderp
    r1 = np.linalg.norm(intruderp - x1)
    r2 = safedis

    R = np.linalg.norm(x1 - x2)

    a = 0.5 * (x1 + x2) + (r1 ** 2 - r2 ** 2) / (2 * R ** 2) * (x2 - x1) + 0.5 * np.sqrt(
        2 * (r1 ** 2 + r2 ** 2) / R ** 2 - (r1 ** 2 - r2 ** 2) ** 2 / R ** 4 - 1) * (
            np.asarray([x2[1] - x1[1], x1[0] - x2[0]]))
    b = 0.5 * (x1 + x2) + (r1 ** 2 - r2 ** 2) / (2 * R ** 2) * (x2 - x1) - 0.5 * np.sqrt(
        2 * (r1 ** 2 + r2 ** 2) / R ** 2 - (r1 ** 2 - r2 ** 2) ** 2 / R ** 4 - 1) * (
            np.asarray([x2[1] - x1[1], x1[0] - x2[0]]))

    a = np.asarray(a)
    b = np.asarray(b)

    v1 = np.asarray(a - ownp)
    v2 = np.asarray(b - ownp)

    v1 = speed / np.linalg.norm(v1) * v1
    v2 = speed / np.linalg.norm(v2) * v2


    return v1, v2

def dist(a, b):
    return np.linalg.norm(np.subtract(a, b))


def make_circle2(center, radius, filled=True):
    points = []
    for i in range(30):
        ang = 2 * math.pi * i / 30
        points.append((math.cos(ang) * radius + center[0], math.sin(ang) * radius + center[1]))
    if filled:
        return rendering.FilledPolygon(points)
    else:
        return rendering.PolyLine(points, True)


def GenIntruder(window_width, n, state):
    ori = []
    vv = []
    v = 2
    
    #####################################################################
    for i in range(3):
        intruder1ori = (100, 200)
        while dist(intruder1ori, state) < 70 or dist(intruder1ori, (100, 200)) < 70:
            if i == 0:
                intruder1ori = (5+np.random.randint(30), 200) #(20+1.0 * np.random.randint(60), 120+1.0 * np.random.randint(60))
                aa = -np.pi/2 + np.pi/2 * np.random.random() #-1*np.pi / 4.5
            if i == 1:
                intruder1ori = (20 + 1.0 * np.random.randint(60), 20 + 1.0 * np.random.randint(60))
                aa = np.pi/2 * np.random.random() #np.pi / 8
            if i == 2:
                intruder1ori = (120 + 1.0 * np.random.randint(60), 120 + 1.0 * np.random.randint(60))
                aa = -np.pi + np.pi/2 * np.random.random() #-3.5*np.pi/4
        ori.append(intruder1ori)
        vv.append((v * np.cos(aa), v * np.sin(aa)))


    ori=np.asarray(ori)
    vv=np.asarray(vv)

    return ori, vv

def distIntru(pp, state):
    dist00 = []
    for i in range(len(pp)):
        dist0 = dist(pp[i], state)
        dist00.append(dist0)
    dist00 = np.asarray(dist00)
    K = 4

    lst = pd.Series(dist00)
    i = lst.nsmallest(K)
    rank = [0, 1, 2, 3]
    return rank


