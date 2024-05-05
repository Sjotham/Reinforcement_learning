import numpy as np
import random
from FourRooms import FourRooms


def main():
    var = input("Welcome \n Would you like to run this in a stochastic simulation?\n Y or N\n ")
    if var == "yes":

        # FourRooms Object
        fourRoomsObj = FourRooms('simple', True)
        print("Training started using Stochastic Actions")
    else:

        fourRoomsObj = FourRooms('simple')
        print("Training started using Deterministic Actions")

    # Initialize the parameters
    alpha = 0.8  # learning rate
    gamma = 0.8  # discount factor

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01

    # initialize the tables
    q_table = np.zeros((13, 13, 4))
    rewards = np.full((13, 13), (-1))

    def next_action(current_row_index, current_column_index, epsilon):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold < epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(q_table[current_row_index, current_column_index])

    def reward(gridType):
        if gridType == -1:
            return -100
        elif gridType == 0:
            return -1
        else:
            return 100

    # q-learning
    for epoch in range(1000):

        fourRoomsObj.newEpoch()
        row_index, column_index = fourRoomsObj.getPosition()  # Current state
        isTerminal = fourRoomsObj.isTerminal()

        print(f"Epoch {epoch} of {1000}")
        while not isTerminal:

            action_index = next_action(row_index, column_index, exploration_rate)

            old_row_index, old_column_index = row_index, column_index
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action_index)
            row_index, column_index = newPos

            # reward for next state
            rewards = reward(gridType)

            # TD and update
            old_q_value = q_table[old_row_index, old_column_index, action_index]
            temporal_difference = rewards + (gamma * np.max(q_table[row_index, column_index])) - old_q_value
            new_q_value = old_q_value + (alpha * temporal_difference)
            q_table[old_row_index, old_column_index, action_index] = new_q_value

            if isTerminal or packagesRemaining == 0:
                break

        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * epoch)

    print('Done Training!')

    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()