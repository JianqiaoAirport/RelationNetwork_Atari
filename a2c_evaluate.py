import numpy as np

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
# from stable_baselines import A2C
from a2c_main import A2C

import os
import logging
import time


import a2c_main as a2c


def evaluate_model(model_name, env, log_dir=a2c.log_dir, model_dir=a2c.model_training_dir, test_num=3):
    """
    evaluate and delete that model
    :param model_name: string. e.g. model_101.pkl
    :param log_dir: string
    :param model_dir: string
    :param test_num: int
    :return:
    """
    total_score = 0
    global best_mean_reward

    try:
        model = A2C.load(model_dir + "/" + model_name)
    except Exception as e:
        time.sleep(0.5)
        return

    try:
        os.remove(model_dir + "/" + model_name)
    except Exception as e:
        time.sleep(0.3)
        return

    n_steps = int(model_name[6: -4])-1  # model_ ->6   .zip->-4

    for _ in range(test_num):

        reward_sum = 0
        obs = env.reset()
        dones = [False]
        while not dones[0]:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, dones, info = env.step(action)
            reward_sum += rewards[0]

        total_score += reward_sum

    ave_score = total_score / test_num
    # Note that each "n_steps" contains n_env(16 here) * n_batch(5 by default) time_steps, here should time 80
    logging.info("Time_step: %d, Score: %f" % ((n_steps + 1) * 80, ave_score))

    # New best model, you could save the agent here
    if ave_score > best_mean_reward:
        best_mean_reward = ave_score
        # Example for saving best model
        # print("Saving new best model")
        model.save(log_dir + 'best_model.zip')


if __name__ == "__main__":

    import random

    log_dir = a2c.log_dir
    model_dir = a2c.model_training_dir

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    logging.basicConfig(filename=log_dir + 'training_record.log', filemode="a", level=logging.DEBUG)

    env_test = make_atari_env(a2c.ATARI_ENV, num_env=1, seed=a2c.n_cpu+random.randint(1, 10),
                              wrapper_kwargs={"episode_life": False, "clip_rewards": False})
    env_test = VecFrameStack(env_test, n_stack=4)

    best_mean_reward = -np.inf

    count = 0

    while True:
        # if count >= 50000000/800:
        #     break

        model_files = os.listdir(model_dir)
        model_files = sorted(model_files, key=lambda x: int(x[6: -4]))

        if len(model_files) >= 1:
            if len(model_files) == 1:
                time.sleep(2)  # avoid a bug: when the file name is created by the os, the zip has not be completed...
            model_name = model_files[0]
            evaluate_model(model_name, env_test, log_dir=log_dir, model_dir=model_dir)
            count += 1

        else:
            time.sleep(1)
