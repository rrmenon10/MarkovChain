__author__ = 'Rakesh R Menon'

import numpy as np
import random
from collections import deque

class ReplayBuffer():

    def __init__(self, max_samples):
        self.buffer_size = max_samples
        self.buffer = deque(maxlen=max_samples)

    def store_sample(self, s_t, a_t, r_t, s_t_1, done):

        transition = (s_t, a_t, r_t, s_t_1, done)
        self.buffer.append(transition)

    def get_sample(self, minibatch_size):

        minibatch_size = min(minibatch_size, self.size)
        samples = random.sample(self.buffer, minibatch_size)
        return zip(*samples)

    def _clear_buffer(self):
        self.buffer.clear()

    @property
    def size(self):
        return len(self.buffer)