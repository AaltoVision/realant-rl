import os
from datetime import datetime
import glob
import argparse
import pickle
import torch
import zmq
from td3 import TD3
from rollout_server import OBS_SIZE, ACT_SIZE

parser = argparse.ArgumentParser(description='RealAnt training client')
parser.add_argument('--n_episodes', default=100, type=int)
parser.add_argument('--resume', default='', type=str) # folder path of past run
parser.add_argument('--task', default='walk', type=str)
args = parser.parse_args()

if args.resume == '':
    # Create new folder
    now = datetime.now()
    project_dir = now.strftime('%Y_%m_%d_%H_%M_%S') + "_" + args.task
    os.mkdir(project_dir)
    start_episode = 0
else:
    # Find latest episode and continue from there
    project_dir = args.resume
    list_of_files = glob.glob(f'{args.resume}/td3_*')
    latest_file = max(list_of_files, key=os.path.getctime)
    start_episode = int(latest_file.split('.')[-2].split('_')[-1]) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.resume != '':
    # load exisiting agent
    with open(latest_file, 'rb') as f:
        td3 = pickle.load(f)
else:
    # create new agent if it doesn't exist
    td3 = TD3(device, OBS_SIZE, ACT_SIZE)

# connect to robot
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')

for episode in range(start_episode, start_episode+args.n_episodes):
    now = datetime.now().isoformat()
    print(f'\nEpisode {episode} {now}')
    print(f'Collecting data...')
    if (not args.resume) and (episode < 10):
        print("Random episode")
        socket.send_pyobj((args.task, None))
    else:
        # send actor weights from CPU
        td3.actor.to('cpu')
        socket.send_pyobj((args.task, td3.actor.state_dict()))
        td3.actor.to(device)

    # recieve new data
    new_data = socket.recv_pyobj()

    # save episodic data
    with open(f'{project_dir}/episode_{episode}.pickle', 'wb') as f:
        pickle.dump(new_data, f)

    new_transitions, new_info = new_data

    _, _, rewards, _, _ = zip(*new_transitions)
    print(f'Return: {sum([r[0] for r in rewards])}')

    # update replay buffer
    td3.replay_buffer.extend(new_transitions)

    print('Training...')
    start_time = datetime.utcnow()
    for _ in range(len(new_transitions)):
        td3.update_parameters()

    print("Training took: %3.2fs" % (datetime.utcnow() - start_time).total_seconds())

    # save agent
    with open(f'{project_dir}/td3_{episode}.pickle', 'wb') as f:
        pickle.dump(td3, f)
