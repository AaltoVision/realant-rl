from gym.envs.registration import register

register(
    id='RealAntBullet-v0',
    entry_point='realant_sim.pybullet:AntBulletEnv',
    max_episode_steps=600,
)

register(
    id='RealAntMujoco-v0',
    entry_point='realant_sim.mujoco:AntEnv',
    max_episode_steps=200,
)
