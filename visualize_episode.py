import argparse
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot(obses, actions, rewards, info, episode):
    obses = np.array(obses).T
    actions = np.array(actions).T

    if info is not None:
        body_xyz = np.array(info).T[:3]

    body_xyz_vel = obses[:3]
    body_height = obses[3]
    body_rpy_vel = obses[4:7]
    body_rpy_sin = obses[7:10]
    body_rpy_cos = obses[10:13]
    joint_positions = obses[13:21]
    joint_positions_vel = obses[21:]
    rewards = [r[0] for r in rewards]

    fig = plt.figure(figsize=(15, 18), constrained_layout=True)
    gs = fig.add_gridspec(14, 3)

    if info is None:
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(body_height)
        ax.set_xlim(0, 200)
        ax.set_ylabel('Z')

        ax = fig.add_subplot(gs[0, 0])
        ax.axis('off')
        ax.set_title('Body XYZ ')

    for j in range(3):
        if info is not None:
            ax = fig.add_subplot(gs[j, 0])
            ax.plot(body_xyz[j])
            ax.plot(np.cumsum(body_xyz_vel[j], axis=0))
            if j == 0:
                ax.plot(np.cumsum(rewards, axis=0))
            ax.set_xlim(0, 200)
            ax.set_ylabel(['X', 'Y', 'Z'][j])
            if j == 0:
                ax.set_title('Body XYZ Positions [m]')

            #ax = fig.add_subplot(gs[j, 2])
            #ax.plot(body_maxacc_xyz[j])
            #ax.plot(body_acc_xyz[j])
            #ax.plot(-body_maxacc_xyz[j])
            #ax.set_xlim(0, 200)
            #ax.set_ylabel(['X', 'Y', 'Z'][j])
            #if j == 0:
            #    ax.set_title('Body XYZ Accelerations [m/s^2]')


        ax = fig.add_subplot(gs[j, 1])
        ax.plot(body_xyz_vel[j])
        if j == 0:
            ax.plot(rewards)
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.02, 0.02)
        ax.set_ylabel(['X', 'Y', 'Z'][j])
        if j == 0:
            ax.set_title('Body XYZ Velocities [m/s]')

        ax = fig.add_subplot(gs[3+j, 0])
        ax.plot(body_rpy_sin[j])
        ax.plot(body_rpy_cos[j])
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 200)
        ax.set_ylabel(['R', 'P', 'Y'][j])
        if j == 0:
            ax.set_title('Body RPY Sine and Cosine')

        ax = fig.add_subplot(gs[3+j, 1])
        ax.plot(body_rpy_vel[j])
        ax.set_xlim(0, 200)
        ax.set_ylabel(['R', 'P', 'Y'][j])
        if j == 0:
            ax.set_title('Body RPY Velocities')

    for j in range(8):
        ax = fig.add_subplot(gs[6+j, 0])
        ax.plot(joint_positions[j])
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 200)
        ax.set_ylabel(j+1)
        if j == 0:
            ax.set_title('Joint Positions')

        ax = fig.add_subplot(gs[6+j, 1])
        ax.plot(joint_positions_vel[j])
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 200)
        ax.set_ylabel(j+1)
        if j == 0:
            ax.set_title('Joint Velocities')

        ax = fig.add_subplot(gs[6+j, 2])
        ax.plot(actions[j])
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(0, 200)
        ax.set_ylabel(j+1)
        if j == 0:
            ax.set_title('PID Setpoints')

    fig.suptitle(f'Episode {episode}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize observations from RealAnt')
    parser.add_argument('--exp_folder', default='', type=str)
    parser.add_argument('--episode', default=-1, type=int)
    args = parser.parse_args()

    if args.episode == -1:
        # plot all episodes
        list_of_files = glob.glob(f'{args.exp_folder}/episode_*.pickle')
    else:
        # plot one episode
        list_of_files = [f'{args.exp_folder}/episode_{args.episode}.pickle']

    for filename in list_of_files:
        print("processing", filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if len(data) == 2:
            transitions, info = data
        else:
            transitions, info = data, None

        obses, actions, rewards, _, _ = zip(*transitions)
        print(f'Return: {sum([r[0] for r in rewards])}')

        episode = filename.split('/')[-1].split('.')[0].split('_')[-1]
        plot(obses, actions, rewards, info, episode)

        plt.savefig(filename.split('.')[0] + '.png')
