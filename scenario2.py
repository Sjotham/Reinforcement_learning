import numpy as np
import random
from FourRooms import FourRooms


def main():
    var = input("Welcome \n Would you like to run this in a stochastic simulation?\n Y or N\n ")
    if var == "yes":

        fourRoomsObj = FourRooms('multi', True)
        print("Training started using Stochastic Actions")
    else:
        # Create the FourRooms Object
        fourRoomsObj = FourRooms('multi')
        print("Training started using Deterministic Actions")

    # parameters
    alpha = 0.1  # learning rate
    gamma = 0.99  # discount factor

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    # Q-table
    q_table1 = np.zeros((13, 13, 4))
    q_table2 = np.zeros((13, 13, 4))
    q_table3 = np.zeros((13, 13, 4))

    def get_next_action(current_row_index, current_column_index, epsilon):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold < epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(q_table[current_row_index, current_column_index])

    # Check if the action leads to a desired state
    def reward(gridType):
        if gridType == -1:
            return -100
        elif gridType == 0:
            return -1
        else:
            return 100

    # Q-learning algorithm
    for epoch in range(2000):
        # Start new epoch and get current location
        fourRoomsObj.newEpoch()
        row_index, column_index = fourRoomsObj.getPosition()  # Current state
        isTerminal = fourRoomsObj.isTerminal()

        # Display current epoch for user while awaiting training
        print(f"Epoch {epoch} of {2000}")
        # Continue to take actions until a package is found
        while not isTerminal:

            action_index = get_next_action(row_index, column_index, exploration_rate)

            old_row_index, old_column_index = row_index, column_index

            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action_index)
            row_index, column_index = newPos

            # receive the reward for moving to the new state
            rewards = reward(gridType)

            # q-table to use based on remaining packages
            if packagesRemaining == 3:
                q_table = q_table1
            elif packagesRemaining == 2:
                q_table = q_table2
            else:
                q_table = q_table3

            old_q_value = q_table[old_row_index, old_column_index, action_index]
            temporal_difference = rewards + (gamma * np.max(q_table[row_index, column_index])) - old_q_value
            new_q_value = old_q_value + (alpha * temporal_difference)

            if packagesRemaining == 3:
                q_table1[old_row_index, old_column_index, action_index] = new_q_value
            elif packagesRemaining == 2:
                q_table2[old_row_index, old_column_index, action_index] = new_q_value
            else:
                q_table3[old_row_index, old_column_index, action_index] = new_q_value

            if isTerminal or packagesRemaining == 0:
                break
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * epoch)

    print('Done Training!')

    # Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()