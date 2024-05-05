import numpy as np
import random
from FourRooms import FourRooms


def main():
    # Ask user whether to use stochastic or not
    var = input("Welcome \n Would you like to run this in a stochastic simulation?\n Y or N\n ")
    if var == "yes":
        # Create FourRooms Object that uses stochastic actions
        fourRoomsObj = FourRooms('rgb', True)
        print("Stochastic Actions, Training")
    else:
        # Create FourRooms Object that uses deterministic actions
        fourRoomsObj = FourRooms('rgb')
        print("Deterministic Actions, Training")

    # Create the Q-table of each (state, action) pair and fill with zeros
    q_table1 = np.zeros((13, 13, 4))
    q_table2 = np.zeros((13, 13, 4))
    q_table3 = np.zeros((13, 13, 4))

    alpha = 0.8  # learning rate
    gamma = 0.8  # discount factor

    exploration_rate = 1
    max_exploration_rate = 0.1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    # Epsilon greedy algorithm
    def next_action(current_row_index, current_column_index, epsilon):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold < epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(q_table[current_row_index, current_column_index])

    # Reward system
    def reward(gridType):
        if gridType == -1:
            return -100
        elif gridType == 0:
            return -1
        else:
            return 100

    # Q-learning
    for epoch in range(3000):
        # Initialize the first package
        fourRoomsObj.newEpoch()
        row_index, column_index = fourRoomsObj.getPosition()  # Current state
        isTerminal = fourRoomsObj.isTerminal()
        packagesRemaining = fourRoomsObj.getPackagesRemaining()

        package_order = 1

        # Current epoch while running
        print(f"Epoch {epoch} of {3000}")
        # find package
        while packagesRemaining > 0:
            action_index = next_action(row_index, column_index, exploration_rate)

            # store the old row and column indexes
            old_row_index, old_column_index = row_index, column_index
            # Take new action: perform the chosen action, and transition to the next state
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action_index)
            row_index, column_index = newPos

            # Keep track of correct order of packages
            if gridType == package_order:
                package_order += 1

            # Find package
            if isTerminal and gridType != package_order and packagesRemaining > 0:
                fourRoomsObj.newEpoch()
                q_table1 = np.zeros((13, 13, 4))
                q_table2 = np.zeros((13, 13, 4))
                q_table3 = np.zeros((13, 13, 4))
                exploration_rate = 1
                package_order = 1

            # new state rewards
            rewards = reward(gridType)

            # Choose which q-table
            if packagesRemaining == 3:
                q_table = q_table1
            elif packagesRemaining == 2:
                q_table = q_table2
            else:
                q_table = q_table3

            # TD and update the Q-table for the previous state and action pair
            old_q_value = q_table[old_row_index, old_column_index, action_index]
            temporal_difference = rewards + (gamma * np.max(q_table[row_index, column_index])) - old_q_value
            new_q_value = old_q_value + (alpha * temporal_difference)

            # q-table updates
            if packagesRemaining == 3:
                q_table1[old_row_index, old_column_index, action_index] = new_q_value
            elif packagesRemaining == 2:
                q_table2[old_row_index, old_column_index, action_index] = new_q_value
            else:
                q_table3[old_row_index, old_column_index, action_index] = new_q_value

            if isTerminal and packagesRemaining == 0:
                break
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * epoch)

    print('Done Training')

    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()