import random
import time

import numpy as np
import skimage
from tqdm import trange
# from vizdoom import *
import vizdoom as vzd

from dqn_agent import DQN_Agent

# Game settings
SCREEN_FORMAT = vzd.ScreenFormat.GRAY8
SCREEN_RESOLUTION = vzd.ScreenResolution.RES_320X240

# Q-learning hyperparameters
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.99
EPOCHS = 5
STEPS_PER_EPOCH = 1000
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
TEST_EPISODES_PER_EPOCH = 100

RESOLUTION = (30, 45)  # why these values?
CONFIG = "scenarios/basic.cfg"

# Utility methods


def preprocess(img):
    '''Downsample image and change pixels from bytes to 32-byte floats in [0,1].'''
    img = skimage.transform.resize(img, RESOLUTION)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_game():
    # Create game
    game = vzd.DoomGame()

    # Load basic configuration
    game.load_config(CONFIG)

    # Set these in case they weren't set in CONFIG
    # IMP: Need this to be False for training
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(SCREEN_FORMAT)
    game.set_screen_resolution(SCREEN_RESOLUTION)

    game.init()
    return game


def run():
    pass


if __name__ == '__main__':

    # Define actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK. See basic.cfg.
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [shoot, left, right]  # Order is irrelevant

    episodes = 10

    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            # img = state.screen_buffer
            # misc = state.game_variables
            reward = game.make_action(random.choice(actions))
            # print("\treward:", reward)

            print("State #" + str(state.number))
            print("Game variables: ", state.game_variables)
            print("Reward: ", reward)
            print("=====================")

            if (sleep_time > 0):
                time.sleep(sleep_time)

        print("Episode finished.")
        print("Total reward: ", game.get_total_reward())
        print("************************")
        time.sleep(2)
