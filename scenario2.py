import numpy as np
import random
from FourRooms import FourRooms


def main():
    var = input("Welcome \n Would you like to run this in a stochastic simulation?\n Y or N\n ")
    if var.lower() == "y":
        fourRoomsObj = FourRooms('multi', True)
        print("Training started using Stochastic Actions")
    else:
        fourRoomsObj = FourRooms('multi')
        print("Training started using Deterministic Actions")

    # parameters
    alpha = 0.1  # learning rate
    gamma = 0.99  # discount factor

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    # Q-tables for different package counts
    q_table1 = np.zeros((13, 13, 4))  # Q-table when 3 packages are remaining
    q_table2 = np.zeros((13, 13, 4))  # Q-table when 2 packages are remaining
    q_table3 = np.zeros((13, 13, 4))  # Q-table when 1 package is remaining

    # Define function to choose the next action using epsilon-soft policy
    def get_next_action(current_row_index, current_column_index, epsilon, q_table):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold < epsilon:
            # Choose a random action with probability epsilon
            return np.random.randint(0, 4)
        else:
            # Choose the best action with probability 1-epsilon
            best_action = np.argmax(q_table[current_row_index, current_column_index])
            return best_action

    # Function to define the reward based on grid type
    def reward(gridType):
        if gridType == -1:
            return -100  # Penalty for hitting a wall
        elif gridType == 0:
            return -1  # Penalty for a neutral step
        else:
            return 100  # Reward for finding a package

    # Main Q-learning algorithm
    for epoch in range(2000):
        fourRoomsObj.newEpoch()
        row_index, column_index = fourRoomsObj.getPosition()
        isTerminal = fourRoomsObj.isTerminal()
        print(f"Epoch {epoch} of {2000}")

        while not isTerminal:
            # Determine which Q-table to use
            packagesRemaining = fourRoomsObj.getPackagesRemaining()
            q_table = [q_table1, q_table2, q_table3][3 - packagesRemaining]

            action_index = get_next_action(row_index, column_index, exploration_rate, q_table)
            old_row_index, old_column_index = row_index, column_index

            # Execute action
            gridType, newPos, _, isTerminal = fourRoomsObj.takeAction(action_index)
            row_index, column_index = newPos
            rewards = reward(gridType)

            # Update Q-table
            old_q_value = q_table[old_row_index, old_column_index, action_index]
            temporal_difference = rewards + gamma * np.max(q_table[row_index, column_index]) - old_q_value
            q_table[old_row_index, old_column_index, action_index] = old_q_value + alpha * temporal_difference

            if isTerminal:
                break

        # Decay exploration rate
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * epoch)

    print('Done Training!')
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
