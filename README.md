# Reinforcement-Learning-on-Unknown-Terrain
## Problem Statement

An agent starts at a given position and must reach a circular goal region while minimizing total movement cost. The surface consists of hills and valleys of varying steepness, but the exact shape of the terrain is unknown to you. The environment should be treated as a **black box** — you can only interact with it through its public methods. You are provided with a Python file named `terrain_env.py`, which defines a continuous 2D environment called `TerrainEnv`. 

## Environment Description

### State
A NumPy array `[x, y]` representing the agent’s current position. Both `x` and `y` are continuous values between `0` and `20`.

### Action
A real-valued angle `theta` in radians. Each step moves the agent one unit in that direction: `x := x + cos(theta)` and `y := y + sin(theta)`.

### Reward
Each step gives a negative reward equal to `-(1 + height gain)`, where height gain is the increase in terrain height from the current position to the next. If the step is flat or downhill, the height gain is treated as zero. Hence, every flat or downhill step gives a reward of `-1`, while uphill steps give rewards **less than -1**, proportional to how steep the climb is. The agent’s goal is to **maximize cumulative reward** (i.e., minimize total movement cost).

### Goal Region
A circular terminal area centered at `(15, 12)` with a radius of `0.5`. An episode ends when the agent enters this circle.

### Reset
`env.reset()` returns a starting position `[x, y]`. You may also call `env.reset(start=(x, y))` to specify a start point (for example, `(0, 0)`).

### Step
`next_state, reward, done = env.step(state, theta)` performs one step of length `1` in direction `theta` and returns the next position, the reward obtained, and a Boolean flag `done` that indicates if the goal has been reached.

### Boundary
Positions are clipped to remain within `[0, 20] × [0, 20]`.

## Assignment Tasks

### Implement Reinforcement Learning Algorithms
Implement **three reinforcement learning algorithms** of your choice. You may select any combination from the following broad categories: **Tabular methods** (e.g., Q-learning, SARSA), **Function approximation methods** (e.g., linear or tile-coded Q-learning), **Deep RL methods** (e.g., DQN, REINFORCE, Actor–Critic). You may optionally include **planning-based methods** (e.g., Dyna-Q).

The environment allows infinitely many possible actions (one per angle). You may discretize the action space (for example, use 8, 16, or 32 angles). Train each agent on the given environment. Each agent should learn a policy that maximizes expected cumulative reward. You may decide the number of episodes, learning rate, exploration strategy, and other hyperparameters.

### Compare Learning Progress
Record total reward per episode during training. Plot a single graph showing how the average or smoothed reward evolves over episodes for all three algorithms on the same axes. The plot should allow a visual comparison of which method learns faster or performs better.

### Visualize One Successful Rollout
Choose the best-performing agent after training. Simulate one complete episode starting from `(0, 0)`. Plot the trajectory of the agent’s movement over the terrain. You may use the terrain as a background heatmap for clarity. Indicate the goal location clearly on the plot.

## Output
A Python file named `terrain_agents.py` containing the implementations of your three chosen algorithms, all necessary code to train, evaluate, and plot results, and a `main` function that performs the following when executed: trains three agents on the terrain with the best parameters/hyperparameters you managed to tune, plots a comparison of reward vs. episode for all three, and performs one rollout with the best agent from starting points `(0, 0)` and three other random points, plotting the paths.

Aniket Das (AniketDas13)
