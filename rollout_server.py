import time
import torch
import zmq
import json
import threading
import numpy as np
import datetime
from collections import deque
from td3 import TD3

import multiprocessing

FRAME_STACKING = 4
ACT_SIZE = 8
OBS_SIZE = 29 * FRAME_STACKING 

ctx = zmq.Context()

act_pub = ctx.socket(zmq.PUB)
act_pub.connect('tcp://localhost:3002')

last_ant_meas = None
last_camera_meas = None

last_frame_ant_meas = None
last_frame_camera_meas = None
last_frame_jpos = None

past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)


def collect_and_distribute_measurements(child_conn):
    ctx1 = zmq.Context()

    obs_sub = ctx1.socket(zmq.SUB)
    obs_sub.connect('tcp://localhost:3001')
    obs_sub.setsockopt(zmq.SUBSCRIBE, b'')

    print("observations collection started")
    last_ant_time = 0
    last_ant_meas = None
    last_camera_meas = None

    while True:
        d = obs_sub.recv_multipart()
        if d[0][0] == ord(b"{"):
            j = json.loads(d[0])
            if j["id"] == "serial":
                last_ant_meas = j
                last_ant_time = j["ant_time"]
            if j["id"] == "external_tag_tracking_camera":
                last_camera_meas = j
         
        if child_conn.poll():
            child_conn.recv()
            child_conn.send([last_ant_meas, last_camera_meas])


class EnvironmentHandler():
    def __init__(self):
        self.running = True
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.p = multiprocessing.Process(target=collect_and_distribute_measurements, args=(self.child_conn,))
        self.p.start()
        self.zero_j_cnt = 0
        self.zero_c_cnt = 0

    def get_obs(self):
        global last_ant_meas, last_camera_meas, last_frame_ant_meas, last_frame_camera_meas, last_frame_jpos, past_obses
        self.parent_conn.send([])
        last_ant_meas, last_camera_meas = self.parent_conn.recv() 

        while last_ant_meas == None and last_camera_meas == None:
            time.sleep(0.01)

        # pybullet world frame
        # x forward   (roll)
        # y left      (pitch)
        # z up        (yaw)

        # realsense world frame
        # x right       (pitch)
        # y up          (yaw)
        # z backward    (roll)

        default_dt = 0.05 # s
        camera_dt = (last_camera_meas['server_epoch_ms'] - last_frame_camera_meas['server_epoch_ms']) / 1000 if last_frame_camera_meas != None else None
        #print("last_camera based dt", camera_dt)
        if last_frame_ant_meas != None:
            print("ant time vs last ant", last_ant_meas['ant_time'], last_frame_ant_meas['ant_time'])
        joint_dt = (float(last_ant_meas['ant_time']) - float(last_frame_ant_meas['ant_time'])) / 1000 if last_frame_ant_meas != None else None
        #print("last_joint based dt", joint_dt)

        # sanity checks
        if camera_dt == 0:
            self.zero_c_cnt += 1
            if self.zero_c_cnt > 3:
                print("observations stuck, quitting (camera)")
                quit()
            camera_dt = default_dt 
        else:
            self.zero_c_cnt = 0


        if joint_dt == 0:
            self.zero_j_cnt += 1
            if self.zero_j_cnt > 3:
                print("observations stuck, quitting (serial)")
                quit()
            joint_dt = default_dt 
        else:
            self.zero_j_cnt = 0
        # calculate speed (this must be based on camera_dt) 
        # don't reorder (external webcam)
        x_vel = last_camera_meas["xvel"]
        y_vel = last_camera_meas["yvel"]
        z_vel = last_camera_meas["zvel"]

        # Ote Robotics RealAnt action space
        # 0 - hip right front  
        # 1 - ankle right front
        # 2 - hip right back
        # 3 - ankle right back
        # 4 - hip left back
        # 5 - ankle left back
        # 6 - hip left front
        # 7 - ankle left front

        # servo angles to joint positions
        angles = ["s%d_angle" %d for d in range(1,9)]
       
        angles = np.array([float(last_ant_meas[a]) for a in angles])


        # re-order and position accordingly
        jpos = np.zeros(8)
        servo_middle = 512  # ax12a value for servo middle position
        servo_half_range = 512-224  # ax12a range from middle to zero degrees
        jpos[0] = -np.clip(-(angles[6] - servo_middle) / servo_half_range, -1, 1)
        jpos[1] = (np.clip((angles[7] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)
        jpos[2] = -np.clip(-(angles[4] - servo_middle) / servo_half_range, -1, 1)
        jpos[3] = -(np.clip((angles[5] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)
        jpos[4] = -np.clip(-(angles[2] - servo_middle) / servo_half_range, -1, 1)
        jpos[5] = -(np.clip((angles[3] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)
        jpos[6] = -np.clip(-(angles[0] - servo_middle) / servo_half_range, -1, 1)
        jpos[7] = (np.clip((angles[1] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)

        jpos_vel = (last_frame_jpos - jpos)/joint_dt if last_frame_jpos is not None else np.zeros((8,))

        torso_pos_and_angle = np.array([x_vel, y_vel, z_vel, last_camera_meas["z"], 
    
            (last_frame_camera_meas["roll"] - last_camera_meas["roll"])/camera_dt if last_frame_camera_meas != None else 0,
            (last_frame_camera_meas["pitch"] - last_camera_meas["pitch"])/camera_dt if last_frame_camera_meas != None else 0,
            (last_frame_camera_meas["yaw"] - last_camera_meas["yaw"])/camera_dt if last_frame_camera_meas != None else 0,

            np.sin(last_camera_meas["roll"] / 180. * np.pi), 
            np.sin(last_camera_meas["pitch"] / 180. * np.pi), 
            np.sin(last_camera_meas["yaw"] / 180. * np.pi), 
            np.cos(last_camera_meas["roll"] / 180. * np.pi), 
            np.cos(last_camera_meas["pitch"] / 180. * np.pi), 
            np.cos(last_camera_meas["yaw"] / 180. * np.pi), 

        ])

        obs = np.concatenate([torso_pos_and_angle, jpos, jpos_vel])

        past_obses.append(obs)
        obs = np.concatenate(past_obses)

        last_frame_ant_meas = last_ant_meas
        last_frame_camera_meas = last_camera_meas
        last_frame_jpos = jpos

        info = np.array([last_camera_meas["x"], last_camera_meas["y"], last_camera_meas["z"]])

        return obs, info

    def apply_controls(self, a):
        a = np.array(a)

        a = (np.clip(np.array(a),-1,1) + 1) / 2.0  # scale to 0...1

        hip_range = 256 
        hip_offset = 368 # this limits hip from middle to +-45deg
        ankle_range = 224
        ankle_offset = 288

        # adjust ordering, range and offsets for the physical ant
        b = np.zeros(8)
        b[0] = a[6] * hip_range + hip_offset      # right front
        b[1] = a[7] * ankle_range + ankle_offset
        b[2] = a[4] * hip_range + hip_offset      # right back
        b[3] = a[5] * ankle_range + ankle_offset
        b[4] = a[2] * hip_range + hip_offset     # left back
        b[5] = a[3] * ankle_range + ankle_offset 
        b[6] = a[0] * hip_range + hip_offset     # left front
        b[7] = a[1] * ankle_range + ankle_offset
        a = b

        act_pub.send_multipart([b"cmd", b"s1 %d s2 %d s3 %d s4 %d s5 %d s6 %d s7 %d s8 %d\n" % (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])])

    def reset_tracking(self):
        """ reset tracking state and tracking camera pose """
        global last_frame_ant_meas, last_frame_camera_meas, last_frame_jpos, past_obses

        last_frame_ant_meas = None
        last_frame_camera_meas = None
        last_frame_jpos = None

        past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)

        act_pub.send_multipart([b"tracking_cmd", b"reset_tracking"])

    def reset_servos(self):
        """ reset orientation and servos to initial state """
        act_pub.send_multipart([b"cmd", b"reset\n"])

    def detach_servos(self):
        """ cut torque to servos to save power """
        act_pub.send_multipart([b"cmd", b"detach_servos\n"])
    
    def attach_servos(self):
        """ enable torque to servos to start actuation """
        act_pub.send_multipart([b"cmd", b"attach_servos\n"])


def reset():
    """ reset robot joints and everything before rollout """
    env.reset_tracking()
    env.reset_servos()

def detach_servos():
    """ cut torque to servos to save power """
    env.detach_servos()

def attach_servos():
    """ enable torque to servos to start actuation """
    env.attach_servos()

def get_state():
    """ get current state of joints and realsense data """
    return env.get_obs()

def apply_controls(pid_setpoints):
    """apply controls to the robot"""
    env.apply_controls(pid_setpoints)

def compute_reward_walk(state, action, next_state):
    """ compute reward based on state changes and action applied """
    
    # walk
    reward = forward_vel = next_state[0]
    return reward

def compute_reward_stand(state, action, next_state):
    # stand
    goal_z = 0.12
    body_z = next_state[3]
    reward = -(body_z - goal_z)**2
    return reward

def compute_reward_turn(state, action, next_state):
    # turn
    goal = np.array([0, 0, -np.pi/2])
    body_rpy = np.arctan2(next_state[7:10], next_state[10:13])
    reward = -np.square(goal[2]-body_rpy[2])
    return reward

def rollout(agent, length=200, train=False, random=False, task='walk'):
    """ rollout policy for fixed length and collect data to buffer """
    global last_camera_meas
    attach_servos()
    reset()
    time.sleep(0.2) # tracking reset takes some time

    state, _ = get_state()
    time.sleep(0.05)
    episode_return = 0
    last_time = datetime.datetime.utcnow()
    for t in range(length):
        now = datetime.datetime.utcnow()
        interval = (now - last_time).total_seconds()
        last_time = now

        print("rollout t", t, "time", now, "dt", interval)
        if random:
            action = np.random.uniform(-1, 1, ACT_SIZE)
        else:
            action = agent.act(state, train=train)

        apply_controls(action)
        next_state, info = get_state()
        if task == 'walk':
            reward = compute_reward_walk(state, action, next_state)
        elif task == 'turn':
            reward = compute_reward_turn(state, action, next_state)
        elif task == 'stand':
            reward = compute_reward_stand(state, action, next_state)
        else:
            print("unknown task %s" % task)
            quit()

        not_done = True
        agent.replay_buffer.append([state, action, [reward], next_state, [not_done]])
        agent.info_buffer.append(info)

        episode_return += reward
        state = next_state

        # aim for a 0.05s cycle time, i.e. 20Hz, so sleep however much is still remaining
        time.sleep(max(0.05 - (datetime.datetime.utcnow() - last_time).total_seconds(), 0))

    time.sleep(0.2)
    reset()
    time.sleep(1.5) # allow servos to turn for 1.5 second
    detach_servos()

    return episode_return


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    td3 = TD3(device, OBS_SIZE, ACT_SIZE)
    td3.info_buffer = []

    socket = ctx.socket(zmq.REP)
    socket.bind('tcp://*:5555')

    env = EnvironmentHandler()
    time.sleep(0.5)
    env.reset_servos()
    time.sleep(0.5)
    env.detach_servos()
    time.sleep(1)

    print("Running")

    while True:
        # Wait for actor weights
        (task, actor_weights) = socket.recv_pyobj()
        print("Received actor weights", actor_weights, "task", task)

        # collect new data
        if actor_weights is None:
            rollout(td3, random=True, task=task)
        else:
            td3.actor.load_state_dict(actor_weights)
            rollout(td3, task=task)

        # send new transitions
        socket.send_pyobj((td3.replay_buffer, td3.info_buffer))

        # reset replay buffer
        td3.replay_buffer.clear()
        td3.info_buffer.clear()
