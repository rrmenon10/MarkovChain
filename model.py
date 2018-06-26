__author__ = 'Rakesh R Menon'

import numpy as np
import tensorflow as tf
import tf_utils
import random
import pdb

class dqn:

    def __init__(self, input_size, output_size, hidden_layers, activation_fn, norm, learning_rate, scope_name):

        self.epsilon = 1
        self._gamma = 0.999
        self._lr = learning_rate
        self.state = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="state")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        self.next_state = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name="next_state")
        self.done = tf.placeholder(shape=[None], dtype=tf.float32, name="done")
        self.q_values = tf_utils.mlp(self.state, output_size, hidden_layers, activation_fn=activation_fn, norm=norm, scope_name=scope_name)
        self.target_q_values = tf_utils.mlp(self.next_state, output_size, hidden_layers, activation_fn=activation_fn, norm=norm, scope_name="target_"+scope_name)
        self._vars = tf_utils.trainable_vars(scope_name=scope_name)
        self._sess = tf.get_default_session()
        self.optim = self.optim_op(self.loss)
        self.mem = None

    @property
    def loss(self):

        #Target Q-evaluations
        target_q = self.target_q_values
        action_space = target_q.shape[-1]
        target_q_best_actions = tf.reduce_max(target_q, axis=-1)
        target_q_best_masked = (1. - self.done)*target_q_best_actions

        #Current net Q-evaluations
        q_values = self.q_values
        masked_actions = tf.one_hot(self.action, action_space)
        q_values_masked = tf.reduce_sum(q_values*masked_actions, axis=-1)

        #Compute TD-error
        td_error = self.reward + self._gamma*target_q_best_masked - q_values_masked
        self._loss = self._huber_loss(td_error)

        return self._loss

    def optim_op(self, loss):
        self._optimize = tf.train.AdamOptimizer(learning_rate=self._lr)
        return self._optimize.minimize(loss, var_list=self._vars)

    def get_q(self, state):
        return self.sess.run(self.q_values, feed_dict={self.state : state})
    
    def act(self, state, mode):

        q_values = self.get_q(state)
        best_action = np.argmax(q_values, axis=-1)
        action_space = q_values.shape[-1]

        if mode=="train":
            if random.random()< self.epsilon:
                action = random.randint(0, action_space-1)
            else:
                action = best_action
        elif mode=="test":
            if self.epsilon> 0.00 and random.random()< 0.05:
                action = random.randint(0, action_space-1)
            else:
                action = best_action
        return action

    @property
    def sess(self):
        return self._sess

    def _huber_loss(self, x, delta=1.0):
        # Implementation taken from openai/baselines
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
            )

    def learn(self, minibatch_size):

        if self.mem is None:
            raise ValueError('Replay Memory not initialized!')

        #Sample minibatch
        s_t, a_t, r_t, s_tp, done = self.mem.get_sample(minibatch_size)
        a_t = np.array(a_t).reshape([-1])
        r_t = np.array(r_t).reshape([-1])
        done = np.array(done).astype(int)

        #Optimize
        self.sess.run(self.optim, feed_dict={self.state : s_t, self.action : a_t, self.reward : r_t, self.next_state : s_tp, self.done : done})