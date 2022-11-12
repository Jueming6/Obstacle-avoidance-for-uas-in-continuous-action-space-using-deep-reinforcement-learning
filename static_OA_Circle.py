import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from shapely.geometry import Polygon
from shapely.geometry import Point
from gym.envs.classic_control import rendering

polysize1 = np.dot(2, [9, 20, 18, 21])
polysize2 = np.dot(2, [32, 36, 15, 21])
polysize3 = np.dot(2, [14, 18, 34, 38])
polysize4 = np.dot(2, [35, 37, 36, 39])


def polyReal(polysize):
    return [(polysize[0] * 4, polysize[2] * 4), (4 * polysize[0], 4 * (polysize[3])),
            (4 * (polysize[1]), 4 * (polysize[3])), (4 * (polysize[1]), 4 * polysize[2])]


poly1 = polyReal(polysize1)
poly2 = polyReal(polysize2)
poly3 = polyReal(polysize3)
poly4 = polyReal(polysize4)


def polycenter(poly):
    return [int((poly[0][0] + poly[3][0]) / 2), int((poly[0][1] + poly[1][1]) / 2)]


polycenter1 = polycenter(poly1)
polycenter2 = polycenter(poly2)
polycenter3 = polycenter(poly3)
polycenter4 = polycenter(poly4)
print(polycenter1, polycenter2, polycenter3, polycenter4)

def polylh(poly):
    return int(abs(poly[0][0] - poly[3][0])), int(abs(poly[0][1] - poly[1][1]))


poly1lh = polylh(poly1)
poly2lh = polylh(poly2)
poly3lh = polylh(poly3)
poly4lh = polylh(poly4)

polygon1 = Polygon(poly1)
polygon2 = Polygon(poly2)
polygon3 = Polygon(poly3)
polygon4 = Polygon(poly4)

r1 = 25
r2 = 30
r3 = 30
r4 = 20

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

class Single4cirEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        origin = np.random.randint(400, size=2)
        goal = (250, 250)
        speed = 2
        heading = np.arctan2((goal[1] - origin[1]), (goal[0] - origin[0]))
        self.origin = origin
        self.goal = goal
        self.speed = speed
        self.heading = heading

        self.dt = 1
        self.max_speed = 25
        self.min_speed = 15
        self.window_width = 400
        self.window_height = 400
        self.shape = (self.window_width, self.window_height)
        self.goal_radius = 40

        self.viewer = None

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array(
            [0, -2, -2, 0,0, 0, -400, 0, 0,0, 0, -400, 0, 0, 0, 0, -400, 0, 0,0, 0, -400, 0]),
            high=np.array([400 * np.sqrt(2), 2, 2,
                           0,0,0, 400, 400 * np.sqrt(2),
                           0, 0, 0, 400, 400 * np.sqrt(2),
                           0, 0, 0, 400, 400 * np.sqrt(2),
                           0, 0, 0, 400, 400 * np.sqrt(2)]), dtype=np.float32)

        self.test = False

        self.testcount = 0
        self.testfailcount = 0
        self.fail = 0
        self.failorigin = []
        self.seed()

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
        s = []
        self.poly1frame = np.dot(self.Rmatrix, polycenter1) - np.dot(self.Rmatrix, self.state)
        self.poly2frame = np.dot(self.Rmatrix, polycenter2) - np.dot(self.Rmatrix, self.state)
        self.poly3frame = np.dot(self.Rmatrix, polycenter3) - np.dot(self.Rmatrix, self.state)
        self.poly4frame = np.dot(self.Rmatrix, polycenter4) - np.dot(self.Rmatrix, self.state)
        self.windframe = np.dot(self.Rmatrix, [0.5, 0])


        # ownship
        s.append(dist(self.state, self.goal) / 400 / np.sqrt(2))
        s.append(self.vframe[0] / 2)
        s.append(self.vframe[1] / 2)

        # obstacle1
        if dist(self.state, polycenter1) < r1 + 400:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(self.poly1frame[1]/ 400 )
            s.append(dist(self.state, polycenter1) / 400 / np.sqrt(2))
        else:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)

        # obstacle2
        if dist(self.state, polycenter2) < r2 + 400:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(self.poly2frame[1] / 400)
            s.append(dist(self.state, polycenter2) / 400 / np.sqrt(2))
        else:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)

        # obstacle3
        if dist(self.state, polycenter3) < r3 + 400:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(self.poly3frame[1] / 400)
            s.append(dist(self.state, polycenter3) / 400 / np.sqrt(2))
        else:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)

        # obstacle4
        if dist(self.state, polycenter4) < r4 + 400:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(self.poly4frame[1] / 400)
            s.append(dist(self.state, polycenter4)/ 400 / np.sqrt(2))
        else:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)

        s = np.array(s)

        return s

    def obs_scale(self, s):
        s2 = []

        s2.append(s[0] / self.window_width)
        s2.append(s[1] / self.window_height)
        s2.append(s[2] / (2 * np.pi))

        return np.array(s2)

    def reset(self):
        self.fail = 0
        self.goal = (250, 250)
        if self.test == True:
            self.origin = testorigin[int((self.testcount-1)/2)]
            self.testcount += 1
        else:
            self.origin = randomorigin(self.window_width)

        self.state = self.origin
        self.heading = np.arctan2((self.goal[1] - self.origin[1]), (self.goal[0] - self.origin[0]))
        theta = np.arctan2(self.goal[1] - self.origin[1], self.goal[0] - self.origin[0])
        self.Rmatrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        return self._get_obs()

    def step(self, action):

        self.heading = self.heading + action * np.pi/6
        vx = self.speed * self.dt * math.cos(self.heading)
        vy = self.speed * self.dt * math.sin(self.heading)
        self.velocity = np.array([vx, vy])
        self.state = np.add(self.state, self.velocity)

        self.state = self._limit_coordinates([1.0 * self.state[0], 1.0 * self.state[1]])
        theta = np.arctan2(self.goal[1] - self.state[1], self.goal[0] - self.state[0])
        self.Rmatrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        reward = -0.001 * dist(self.state, self.goal) - 0.05
        done = False
        point = Point(self.state)

        ######################### CIRCLE #########################################
        if dist(self.state, polycenter1) <= r1 or dist(self.state, polycenter2) <= r2 \
                or dist(self.state, polycenter3) <= r3 or dist(self.state, polycenter4) <= r4:
            reward = reward - 16
            if self.test == True:
               self.fail = 1
            done = True

        if dist(self.state, self.goal) < self.goal_radius:
            reward = 10
            done = True

        if done:
            if self.test == True:
                print(int((self.testcount-1)/2))
                self.testfailcount += self.fail
                print(self.testfailcount)
                if self.fail == 1:
                    self.failorigin.append(self.origin)

            if int((self.testcount-1)/2) == 159:
                np.savetxt('failorigin4cir_lr0.txt', self.failorigin,  delimiter=', ', fmt='%.0f')

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        ownship_img = rendering.Image(os.path.join(__location__, 'images/aircraft.png'), 32, 32)
        jtransform = rendering.Transform(rotation=self.heading - math.pi / 2, translation=self.state)
        ownship_img.add_attr(jtransform)
        ownship_img.set_color(255, 241, 4)  # set it to yellow
        self.viewer.onetime_geoms.append(ownship_img)

        # draw goal
        goal_img = rendering.Image(os.path.join(__location__, 'images/goal.png'), 32, 32)
        jtransform = rendering.Transform(rotation=0, translation=self.goal)
        goal_img.add_attr(jtransform)
        goal_img.set_color(15, 210, 81)  # green
        self.viewer.onetime_geoms.append(goal_img)

        # draw obstacle
        obs=make_circle2(polycenter1, r1, filled=True)
        self.viewer.onetime_geoms.append(obs)  # add_geom(obs)

        obs2 = make_circle2(polycenter2, r2, filled=True)
        self.viewer.onetime_geoms.append(obs2)

        obs3=make_circle2(polycenter3, r3, filled=True)
        self.viewer.onetime_geoms.append(obs3)

        obs4 = make_circle2(polycenter4, r4, filled=True)
        self.viewer.onetime_geoms.append(obs4)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


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
