# Reinforcement Learning with RealAnt

Video:
<p align="center">
  <a href="https://www.youtube.com/watch?v=pG-XhH-9s7o">
    <img src="https://github.com/AaltoVision/realant-rl/blob/main/video_thumbnail.jpg?raw=true" alt="RealAnt Summary Video"/>
  </a>
</p>

This repository contains source code for experiments in the paper titled "[RealAnt: An Open-Source Low-Cost Quadruped for Research in Real-World Reinforcement Learning](https://arxiv.org/abs/2011.03085)" by Rinu Boney*, Jussi Sainio*, Mikko Kaivola, Arno Solin, and Juho Kannala. It consists of:
- Supporting software for reinforcement learning with the RealAnt robot
- PyTorch implementations of TD3 and SAC algorithms
- MuJoCo and PyBullet environments of the RealAnt robot

### RealAnt

RealAnt is a minimal and low-cost (~350€ in materials) physical version of the popular 'Ant' benchmark used in reinforcement learning. It can be built using easily available electronic components and a 3D printed body. Code for the RealAnt platform including the 3D models, microcontroller board firmware, Python interface and pose estimation is available here: https://github.com/OteRobotics/realant

<p align="center">
  <img src="https://github.com/OteRobotics/realant/blob/master/stl/RealAnt-v1.jpeg?raw=true" width="50%" alt="Photo of RealAnt"/>
</p>

**Observation space** (29-dim):
1. x, y, and z velocities of the torso (3),
2. z position of the torso (1),
3. sin and cos values of Euler angles of the torso (6),
4. velocities of Euler angles of the torso (3),
5. angular positions of the joints (8), and
6. angular velocities of the joints (8).

We rely on augmented reality (AR) tag tracking using ArUco tags for pose estimation.

**Action space** (8-dim): set-points for the angular positions of the robot joints.

### Experiments with RealAnt Robot

We consider three benchmark tasks:
1. **Stand** upright.
2. **Turn** 180 degrees.
3. **Walk** forward as fast as possible.

TD3 algorithm is able to successfully learn all three tasks. Learning to stand takes around 12 minutes of experience, learning to turn takes 35 minues of experience, and learning to walk takes 40 minutes of experience.

<p align="center">
  <img src="https://github.com/AaltoVision/realant-rl/blob/main/training_results.jpg?raw=true" alt="Training results"/>
</p>

The training code is decoupled into a training client and a rollout server, communicating using ZeroMQ. The training client (`train_client.py`) controls the whole learning process. It sends the latest policy weights to the rollout server (`rollout_server.py`) at the beginning of each episode. The rollout server loads the policy weights, collects the latest observations from the robot, and sends the action computed using the policy network back to the robot. After completing an episode, the rollout server sends back the collected data to the train client. The newly collected data is added to a replay buffer and the agent is updated a few times by sampling from this replay buffer.

The train client and rollout server can be run in different machines. For example, the data collection (with rollout server) can be performed on a low-end computer and training (with train client) can be performed on a high-end computer.

Setup the robot and run 
```
python rollout_server.py
```
and 
```
python train_client.py --n_episodes 250
```
for reinforcement learning with the robot.

The train client logs all data into a newly created experiment folder. After each episode, the robot position and orientation should be reset manually (if necessary). If training gets stuck due to broken serial link or camera observations, restart the the respective script(s), the rollout server, and run `python train_client.py --resume <exp_folder>` to resume training.

We also provide plotting code:
- `visualize_episode.py` to visualize the observations, actions, and rewards during an episode.
- `visualize_returns.py` to plot the cumulative rewards of a training run.

### Experiments with RealAnt Simulation

The simulator environments of the RealAnt robot can be used as:
```
import gym
import realant_sim

env = gym.make('RealAntMujoco-v0')
env = gym.make('RealAntBullet-v0')
```

The results reported in the paper can be reproduced by running:
```
python train.py
```
Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --agent 		| td3	 | 'td3' or 'sac'
| --env   		| mujoco | 'mujoco' or 'pybullet'
| --task  		| walk 	 | 'sleep' or 'turn' or 'walk
| --seed  		| 1	 | random seed
| --latency 		| 2	 | number of steps by which observations are delayed, where 1 step = 0.05 s
| --xyz_noise_std 	| 0.01 	| std of Gaussian noise added to body_xyz measurements
| --rpy_noise_std 	| 0.01 	| std of Gaussian noise added to body_rpy measurements
| --min_obs_stack 	| 4 	| number of past observations to be stacked

The PyBullet environment only supports the 'walk' task and does not support the latency or delay argments.

### License

This project is licensed under the terms of the MIT license. See LICENSE file for details.
