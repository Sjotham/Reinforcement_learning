import numpy as np
import random
import argparse
from FourRooms import FourRooms


# Function to execute the Q-learning logic
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Q-Learning Agent for FourRooms Environment")
    parser.add_argument("-stochastic", action="store_true", help="Enable stochastic action space")
    args = parser.parse_args()

    # Initialize FourRooms environment
    scenario = 'simple'
    stochastic = args.stochastic
    fourRoomsObj = FourRooms(scenario, stochastic)

    if stochastic:
        print("Training started using Stochastic Actions")
    else:
        print("Training started using Deterministic Actions")

    # Initialize the parameters
    alpha = 0.8  # learning rate
    gamma = 0.8  # discount factor

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    # Initialize Q-table
    q_table = np.zeros((13, 13, 4))
    rewards = np.full((13, 13), -1)

    def stochastic_policy(current_row_index, current_column_index, epsilon):
        """
        Stochastic optimal policy with exploration.
        """
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold < epsilon:
            # Exploration: choose a random action
            return np.random.randint(0, 4)
        else:
            # Exploitation: choose the best action but allow stochastic deviation
            best_action = np.argmax(q_table[current_row_index, current_column_index])
            if random.uniform(0, 1) < 0.2:  # 20% chance of choosing a different action
                actions = [0, 1, 2, 3]
                actions.remove(best_action)
                return random.choice(actions)
            return best_action

    def reward(grid_type):
        """
        Reward function based on the grid cell type.
        """
        if grid_type == -1:
            return -100
        elif grid_type == 0:
            return -1
        else:
            return 100

    # Q-learning loop
    for epoch in range(1000):
        fourRoomsObj.newEpoch()
        row_index, column_index = fourRoomsObj.getPosition()  # Initial state
        is_terminal = fourRoomsObj.isTerminal()

        print(f"Epoch {epoch} of {1000}")
        while not is_terminal:
            # Choose an action using stochastic policy
            action_index = stochastic_policy(row_index, column_index, exploration_rate)

            old_row_index, old_column_index = row_index, column_index
            grid_type, new_pos, packages_remaining, is_terminal = fourRoomsObj.takeAction(action_index)
            row_index, column_index = new_pos

            # Determine reward
            current_reward = reward(grid_type)

            # Temporal Difference Update
            old_q_value = q_table[old_row_index, old_column_index, action_index]
            temporal_difference = current_reward + (gamma * np.max(q_table[row_index, column_index])) - old_q_value
            new_q_value = old_q_value + (alpha * temporal_difference)
            q_table[old_row_index, old_column_index, action_index] = new_q_value

            if is_terminal or packages_remaining == 0:
                break

        # Decay exploration rate
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * epoch)

    print('Done Training!')

    # Visualize the path the agent took
    fourRoomsObj.showPath(-1)


# Entry point
if __name__ == "__main__":
    main()
