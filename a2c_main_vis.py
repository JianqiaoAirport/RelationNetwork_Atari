import numpy as np

from stable_baselines.common.cmd_util import make_atari_env
# from new_cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

# from stable_baselines import A2C   # need to change
# from a2c_aux_class import A2C        # need to change
from a2c_vis import A2C

from ARNSRF_VIS_PV_NET import RNCNNPolicy as Policy

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DATE = '20200227'
ATARI_ENV = 'PongNoFrameskip-v4'
log_dir = "A2CVIS_ARNSRF_Pong_log_%s/" % DATE
os.makedirs(log_dir, exist_ok=True)
model_training_dir = log_dir + "/model_training"
os.makedirs(model_training_dir, exist_ok=True)

n_cpu = 16  # multiprocess environment

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, env_test
    # Print stats every 4000 calls
    # Note that each "n_steps" contains n_env(16 here) * n_batch(5 by default) time_steps, should be divided by 80
    if (n_steps + 1) % int(200000/80) == 0:
        # Evaluate policy training performance
        _locals['self'].save(model_training_dir + '/model_%d' % (n_steps + 1))
        pass
    n_steps += 1
    return True


if __name__ == "__main__":

    env = make_atari_env(ATARI_ENV, num_env=n_cpu, seed=0)  # Create n_cpu environments
    env = VecFrameStack(env, n_stack=4)

    total_steps = 10000000

    model = A2C(Policy, env, lr_schedule='constant', verbose=1, tensorboard_log=log_dir+"a2c_noframeskip_v4_tensorboard_%s/" % DATE)
    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(log_dir + "a2c_noframeskip_v4_%s" % DATE)

    del model  # remove to demonstrate saving and loading
