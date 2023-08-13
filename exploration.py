'''
This file is used to explore the game environment.
'''

from vizdoom import *
import random
import time

if __name__ == '__main__':
    # Setup game
    game = DoomGame()

    # Load basic configuration
    # game.load_config("scenarios/basic.cfg")
    game.load_config("scenarios/simpler_basic.cfg")

    # Initialize
    game.init()

    # Define actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK. See basic.cfg.
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions = [shoot, left, right]  # Order is irrelevant

    episodes = 10

    sleep_time = 1.0 / DEFAULT_TICRATE  # = 0.028

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
