import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.a2c.utils import conv, conv_to_fc
import numpy as np
# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor


class CNNPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CNNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            # activ = tf.nn.relu
            # extracted_features = nature_cnn(self.processed_obs, **kwargs)
            # extracted_features = tf.layers.flatten(extracted_features)

            activ = tf.nn.relu
            self.n1 = activ(conv(self.processed_obs, 'c1', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            self.n2 = activ(conv(self.n1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
            self.n3 = activ(conv(self.n2, 'c3', n_filters=64, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
            self.flattened_n3 = conv_to_fc(self.n3)

            pi_h = self.flattened_n3
            for i, layer_size in enumerate([512]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = pi_latent

            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


if __name__ == "__main__":

    import gym

    from stable_baselines.common.vec_env import SubprocVecEnv
    from stable_baselines import A2C

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    # multiprocess environment
    n_cpu = 1
    env = SubprocVecEnv([lambda: gym.make('Breakout-v0') for i in range(n_cpu)])

    model = A2C(CNNPolicy, env, verbose=1, tensorboard_log="./a2c_rn_breakoutv0_tensorboard/")
    model.learn(total_timesteps=100)
    model.save("a2c_rn_Breakout_v0")
