import math
import numpy as np

class TerrainEnv:
    def __init__(self, goal=(15.0, 12.0), goal_radius=0.5, clip=True):
        self.goal = np.array(goal, dtype=float)
        self.goal_radius = float(goal_radius)
        self.clip = bool(clip)

    def height(self, x, y):
        return (8.0 * math.exp(-((x - 8)**2 + (y - 13)**2) / 18.0)
              + 7.5 * math.exp(-((x - 13)**2 + (y - 7)**2) / 22.0)
              + 6.5 * math.exp(-((x - 10)**2 + (y - 10)**2) / 30.0)
              - 8.5 * math.exp(-((x - 11)**2 + (y - 15)**2) / 30.0)
              - 9.0 * math.exp(-((x - 10)**2 + (y - 1)**2) / 20.0))

    def step(self, state, theta):
        x, y = float(state[0]), float(state[1])
        nx = x + math.cos(theta)
        ny = y + math.sin(theta)
        if self.clip:
            nx = min(max(0.0, nx), 20.0)
            ny = min(max(0.0, ny), 20.0)
        h1 = self.height(x, y)
        h2 = self.height(nx, ny)
        cost = 1.0 + max(0.0, h2 - h1)
        reward = -cost
        next_state = np.array([nx, ny], dtype=float)
        done = np.linalg.norm(next_state - self.goal) <= self.goal_radius
        return next_state, reward, done

    def reset(self, start=None):
        if start is None:
            while True:
                s = np.random.uniform(0.0, 20.0, size=2)
                if np.linalg.norm(s - self.goal) > 2.0:
                    return s
        return np.array(start, dtype=float)
