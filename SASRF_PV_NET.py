import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.a2c.utils import conv, conv_to_fc
import numpy as np
# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor


class CNNSAPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CNNSAPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):

            activ = tf.nn.relu
            self.n1 = activ(conv(self.processed_obs, 'c1', n_filters=32, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            self.n2 = activ(conv(self.n1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
            self.n3 = activ(conv(self.n2, 'c3', n_filters=64, filter_size=2, stride=2, init_scale=np.sqrt(2), **kwargs))
            self.flattened_n3 = conv_to_fc(self.n3)

            self.entities = self.concat_coordinates(self.n3)

            shape_list = self.entities.get_shape().as_list()

            self.entities_flatten = tf.reshape(self.entities, [-1, shape_list[1] * shape_list[2], shape_list[3]])

            self.entity_shape = self.entities_flatten.get_shape().as_list()

            self.entities_updated = self.sa_relational_module(self.entities_flatten)  # Note! output shape: [b, n, n, d]

            self.reduced = tf.reduce_max(self.entities_updated, axis=[1])

            pi_h = self.reduced
            for i, layer_size in enumerate([64]):
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

    def sa_relational_module(self, entities_flattened, name='sa', num_heads=3):

        dim = entities_flattened.get_shape().as_list()[2]

        def self_attention(entities_sa, name='self_attention'):

            Q = tf.layers.dense(entities_sa, dim, activation=None,
                                use_bias=False, name=name+'_Q')
            K = tf.layers.dense(entities_sa, dim, activation=None,
                                use_bias=False, name=name+'_K')
            V = tf.layers.dense(entities_sa, dim, activation=None,
                                use_bias=False, name=name+'_V')

            att = tf.matmul(Q, K, transpose_b=True)

            A = tf.matmul(tf.nn.softmax(att/tf.sqrt(tf.cast(dim, tf.float32))), V)

            return A

        A_list = []
        for i in range(num_heads):
            A = self_attention(entities_flattened, name='self_attention_%d'%i)
            A_list.append(A)

        A_concated = tf.concat(A_list, axis=-1)

        A_squeezed = tf.layers.dense(A_concated, dim, activation=tf.nn.relu, name=name+'_squeeze_0')

        A_squeezed = tf.layers.dense(A_squeezed, dim, activation=tf.nn.relu, name=name + '_squeeze_1')

        entity_add_norm = tf.contrib.layers.layer_norm(A_squeezed+entities_flattened)

        return entity_add_norm

    def concat_coordinates(self, input):
        shape = tf.shape(input)

        positions = tf.range(shape[0] * shape[1] * shape[2])
        position_embedding = tf.reshape(positions,
                                        shape=[shape[0], shape[1], shape[2], 1])
        position_embedding = position_embedding % (shape[1] * shape[2])
        position_embedding = tf.cast(position_embedding, tf.float32)
        #
        x_embedding = tf.cast(tf.floor_div(position_embedding, tf.cast(shape[2], tf.float32)), tf.float32)
        x_mean = tf.reduce_mean(x_embedding, axis=[1, 2], keep_dims=True)
        x_max = tf.reduce_max(x_embedding, axis=[1, 2], keep_dims=True)
        x_min = tf.reduce_min(x_embedding, axis=[1, 2], keep_dims=True)
        x_embedding = (x_embedding - x_mean) / (x_max - x_min) * 2

        y_embedding = tf.cast(tf.cast(position_embedding, tf.int32) % shape[2], tf.float32)
        y_mean = tf.reduce_mean(y_embedding, axis=[1, 2], keep_dims=True)
        y_max = tf.reduce_max(y_embedding, axis=[1, 2], keep_dims=True)
        y_min = tf.reduce_min(y_embedding, axis=[1, 2], keep_dims=True)
        y_embedding = (y_embedding - y_mean) / (y_max - y_min) * 2

        entities = tf.concat([input, x_embedding], axis=-1)
        entities = tf.concat([entities, y_embedding], axis=-1)

        return entities


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

    model = A2C(CNNSAPolicy, env, verbose=1, tensorboard_log="./a2c_rn_breakoutv0_tensorboard/")
    model.learn(total_timesteps=100)
    model.save("a2c_rn_Breakout_v0")
