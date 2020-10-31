import numpy as np
import gym
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.robot_locomotors import Ant as OriginalAnt
import pybullet_data
import os

#
# This is a modified PyBullet Ant robot environment that
#
#  * Has setpoint control instead of motor torques. This mimics
#    the usage of RC servo motors in the leg joints, they are also
#    controlled using setpoints.
#
#  * The observation space is different to the vanilla Ant. It's roughly
#    similar to what we can get from the physical ant (no feet sensors).
#    Angles or joint ordering is probably not matched still.
#
#  * Possibility to tweak power, joint damping, dimensions and masses.


# Points of reference
#
#  * https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur.py#  * https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_locomotion_envs.py   # profit function in step
#  * https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/robot_locomotors.py  # calc_state i.e. obs space


class Ant(WalkerBase):
    """ Ant robot """

    foot_list = [
        "front_left_foot",
        "front_right_foot",
        "left_back_foot",
        "right_back_foot",
    ]

    def __init__(self):
        # XXX: Hack to load local MJCF xml as PyBullet robot bases has a fixed prefix for the path
        default_path = os.path.join(pybullet_data.getDataPath(), "mjcf")
        rel_path = os.path.relpath(os.path.abspath("realant_sim/pybullet.xml"), default_path)

        WalkerBase.__init__(
            self, rel_path, "torso", action_dim=8, obs_dim=29, power=0.1
        )  # modify power here

    def alive_bonus(self, z, pitch):
        return (
            +1 if z > 0.065 else -1
        )  # 0.25 is central sphere rad, die if it scrapes the ground

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        bullet_client.setGravity(0, 0, -9.81)  # modify gravity here
        bullet_client.setRealTimeSimulation(False)
        bullet_client.changeDynamics(self.objects[0], -1, mass=1.162 + 0.275 - (8*0.05)) 
        for j in self.ordered_joints:
            bullet_client.changeDynamics(
                self.objects[0],j.jointIndex, mass = 0.050
            )
        # add damping to joints
        for j in self.ordered_joints:
            bullet_client.changeDynamics(
                self.objects[0], j.jointIndex, jointDamping=0.2
            )  # modify joint damping here
            bullet_client.changeDynamics(self.objects[0], j.jointIndex, lateralFriction=1)


class AntBulletEnv(WalkerBaseBulletEnv):
    """ Ant environment """

    def __init__(self, task='walk', render=False):
        self.robot = Ant()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

        self.task = task

        # PID params
        self.Kp = 0.8
        self.Ki = 0
        self.Kd = 1

        # obs and act space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(29,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,))

        # camera position
        self._cam_dist = 1

        self.alpha = 0.7

        self.timestep = 1 / 60 / 4
        self.frame_skip = 4
        self.dt = self.timestep * self.frame_skip

        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': int(np.round(1.0/self.dt))}

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client, gravity=9.8, timestep=self.timestep, frame_skip=self.frame_skip)
        return self.stadium_scene

    def reset(self):
        WalkerBaseBulletEnv.reset(self)

        self.int_error = 0
        self.past_err = 0

        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        self.old_joint_positions = j[0::2]
        self.old_body_xyz = np.array(self.robot_body.pose().xyz(), dtype=np.float32)
        self.old_body_rpy = np.array(self.robot_body.pose().rpy(), dtype=np.float32)

        # self.setpoints = np.zeros(8)
        self.torques = np.zeros(8)
        self.old_joint_velocities = np.zeros(8)
        self.old_joint_accelerations = np.zeros(8)
        self.old_joint_jerks = np.zeros(8)

        return self._get_obs()

    def step(self, setpoints):
        # compute torques
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        ).flatten()
        joint_positions = j[0::2]
        joint_velocities = j[1::2]

        #
        # motor control using a PID controller
        #

        # limit motor maximum speed (this matches the real servo motors)
        timestep = self.scene.timestep
        vel_limit = 0.1  # rotational units/s

        # joint positions are scaled somehow roughly between -1.8...1.8
        # to meet these limits, multiply setpoints by two.
        err = 2 * setpoints - joint_positions
        self.int_error += err
        d_err = err - self.past_err
        self.past_err = err

        torques = np.minimum(
            1,
            np.maximum(-1, self.Kp * err + self.Ki * self.int_error + self.Kd * d_err),
        )

        # clip available torque if the joint is moving too fast
        lowered_torque = 0.0
        torques = np.clip(torques,
            np.minimum(-lowered_torque, (-vel_limit-np.minimum(0, joint_velocities)) / vel_limit),
            np.maximum(lowered_torque, (vel_limit-np.maximum(0, joint_velocities)) / vel_limit))

        # low-pass filter the torques to combat exploiting vibrating movements
        torques = self.torques = self.alpha * torques + (1-self.alpha) * self.torques

        # step
        obs, reward, done, info = super().step(torques)

        info['torques'] = torques
        info['joint_velocities'] = obs[-8:]
        info['joint_accelerations'] = info['joint_velocities'] - self.old_joint_velocities
        info['joint_jerks'] = info['joint_accelerations'] - self.old_joint_accelerations

        self.old_joint_velocities = info['joint_velocities']
        self.old_joint_accelerations = info['joint_accelerations']
        self.old_joint_jerks = info['joint_jerks']

        # replace obs and reward
        obs = self._get_obs()

        if self.task == 'walk':
            reward = obs[0]  # x (forward) speed
        elif self.task == 'sleep':
            reward = -np.square(obs[3])
        else:
            raise Exception('Unknown task')

        return obs, reward, done, info

    def _get_obs(self):
        # Get the wanted observations
        body_xyz = np.array(self.robot_body.pose().xyz(), dtype=np.float32)
        body_rpy = np.array(self.robot_body.pose().rpy(), dtype=np.float32)

        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        ).flatten()
        joint_positions = j[0::2]

        feet_contact = self.robot.feet_contact

        body_xyz_vel = body_xyz - self.old_body_xyz
        body_rpy_vel = body_rpy - self.old_body_rpy

        # Generate a new observation space
        body_rpy_cos = np.cos(body_rpy)
        body_rpy_sin = np.sin(body_rpy)

        joint_positions_vel = joint_positions - self.old_joint_positions

        self.old_joint_positions = joint_positions
        self.old_body_xyz = body_xyz
        self.old_body_rpy = body_rpy

        return np.array(
            [
                *body_xyz_vel,          # 3 values
                body_xyz[2],            # 1
                *body_rpy_vel,          # 3
                *body_rpy_sin,          # 3
                *body_rpy_cos,          # 3
                *joint_positions,       # 8
                *joint_positions_vel,   # 8
            ]
        )
