from keras.models import Model
from keras.layers import Input, Lambda, Concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import numpy as np
from collections import deque
import random


class QModel:

    def __init__(self, state_shape, action_size, batch_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.__memory = deque(maxlen=2000)

        value_input = Input(shape=state_shape, name='value_input')
        x = Dense(128, activation='relu')(value_input)
        x = Dense(action_size, activation='softmax')(x)

        self.__model = Model(inputs=value_input, outputs=x, name='T-Rex_net')
        self.__model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        self.__model.summary()
        self.KEY_NONE = 0
        self.KEY_UP = 1
        self.KEY_DOWN = 2
        self.keys = [self.KEY_NONE, self.KEY_UP, self.KEY_DOWN]
        self.key_to_str = ["NONE", "UP", "DOWN"]

    def get_action(self, state):  # return action, q-value
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), self.epsilon
        act_values = self.__model.predict(state, batch_size=1, verbose=0)[0]
        return np.argmax(act_values), np.max(act_values)

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        adding = True
        for mem in self.__memory:
            s, a, _, _, _ = mem
            if (not (state-s).any()) and a == action:
                adding = False
                break
        if adding:
            self.__memory.append(item)
        return len(self.__memory)

    def train(self):
        if len(self.__memory) < self.batch_size:
            minibatch = random.sample(self.__memory, len(self.__memory))
        else:
            minibatch = random.sample(self.__memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.__model.predict(next_state)[0]))
            target_f = self.__model.predict(state)
            target_f[0][action] = target
            self.__model.fit(state, target_f, batch_size=1, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


