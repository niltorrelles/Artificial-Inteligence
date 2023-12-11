import numpy as np

# Since all the practicals are in english, I will use the same language for my code

# Global variables
environment_columns = 4
environment_rows = 3

# epsilon greedy algorithm 90% will be taken, other 10% will be random
epsilon = 0.9 
# gamma
discount_factor = 0.9
# alpha
learning_rate = 0.9

# actions performed to find the solution
final_actions = []


# Constants for actions
# Definition of actions, let's say our agent can only go up, right and left (not down)
# numeric action codes: up = 0, right = 1, left = 2
UP = 0
RIGHT = 1
LEFT = 2

# IMP! Reward table != Q_Table!
def rewards_table_create(n_rows, n_cols, goal_position, obstacle_position):
    rewards_table = np.zeros((n_rows, n_cols))

    # Calculate the Manhattan distance from each cell to the goal
    for i in range(n_rows):
        for j in range(n_cols):
            distance_to_goal = abs(i - goal_position[0]) + abs(j - goal_position[1])

            # Avoid obstacle position
            if (i, j) == obstacle_position:
                rewards_table[i, j] = -np.inf  # Set a large negative value for obstacle position
            else:
                rewards_table[i, j] = -distance_to_goal

    return rewards_table


#------------------------- Q-Learning functions ---------------------------------#
'''
Create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a) 
The third dimension is the "action", which consists in 3 layers that will allow us to keep track
of the Q_values for each possible action. Remember our agent can do 3 actions, UP, Right and LEFT
'''
def q_table_create(n_rows, n_cols, num_actions):
    # The table, will be fullfilled by 0, q-table != rewards table!
    q_values = np.zeros((n_rows, n_cols, num_actions))
    return q_values

# upgrade q_table with the previous choosen action
def upgrade_q_table(q_table, currentState, nextState):
    pass


# Implementation of the q_learning_algorithm!
def q_learning_algorithm(start_state, q_values, rewards_table, convergence_threshold=0.001):
    print_interval = 10  # The Q table will be printed every 10 episodes

    prev_q_values = q_values.copy()  # Safe the latest q-table (will be used for the convergence)

    episode = 0
    while True:
        # Get the starting location for this episode
        current_state = start_state

        # Continue taking actions (i.e., moving) until we reach a terminal state
        while not is_terminal_state(current_state, rewards_table):
            # Choose which action to take (i.e., where to move next)
            action_index = get_next_action(current_state, q_values)

            # Perform the chosen action, and transition to the next state
            old_row_index, old_column_index = current_state
            next_state = get_next_location(current_state, action_index)

            # Receive the reward for moving to the new state, and calculate the temporal difference
            reward = rewards_table[next_state[0], next_state[1]]
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[next_state[0], next_state[1], :])) - old_q_value

            # Update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value

            # Update the current state for the next iteration
            current_state = next_state

        # Verify if the convergence was reached
        if episode > 0 and np.max(np.abs(q_values - prev_q_values)) < convergence_threshold:
            print(f'Convergence achieved after {episode} episodes!')
            print('Training completed!')
            break

        # Print the intermediate Q-table
        if (episode + 1) % print_interval == 0 or episode == 0:
            print(f'\nQ-Table after {episode + 1} episodes:\n')
            print(q_values)

        prev_q_values = q_values.copy()
        episode += 1

    print('\n Showing Final Q-Table:\n')
    print(q_values)
    return

        

# ----------------- Helper functions -------------------------- #

# This function determines if the state is the goal or the whole
def is_terminal_state(current_state, rewards_table):
    row_index = int(current_state[0])
    column_index = int(current_state[1])
    # check if the reward is diferent of 100 and -100, then is not a terminal state
    if (rewards_table[row_index][column_index] != -100 and rewards_table[row_index][column_index] != 100):
        return False
    else:
        # whole or goal -> terminal state raised!
        return True

# check if an action is valid
def is_valid_move(current_state, action_index):
    current_row_index, current_column_index = current_state
    if action_index == UP and current_row_index > 0:
        return True
    elif action_index == RIGHT and current_column_index < environment_columns - 1:
        return True
    elif action_index == LEFT and current_column_index > 0:
        return True
    else:
        return False

'''
Choose next action method, we will use an epsilon greedy algorithm to choose the next action
our epsilon value will be 0.9, which means that 90% of the actions will be taken, the other 10%
will be random, which encores the agent to explore diferents paths in order to discver better ways
to resolve the problem.
Return an action index
'''
def get_next_action(current_state, q_values):
    # a random value between 0 to 1, will be generated if this value is < than epsilon
    # the best q-value will be taken
    random_num = np.random.rand() # random number between 0 to 1
    row_index = int(current_state[0])
    column_index = int(current_state[1])

    if random_num < epsilon:
        return np.argmax(q_values[row_index, column_index, :])
    else:
        # choose a random action
        random_action = np.random.randint(3)
        while not(is_valid_move(current_state, random_action)):
            random_action = np.random.randint(3)
        return random_action
              

# Define a function that will get the next location based on the chosen action
def get_next_location(current_state, action_index):
    current_row_index, current_column_index = current_state
    new_row_index = current_row_index
    new_column_index = current_column_index
    
    if action_index == UP and current_row_index > 0:
        new_row_index -= 1
    elif action_index == RIGHT and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif action_index == LEFT and current_column_index > 0:
        new_column_index -= 1
    
    next_state = (new_row_index, new_column_index)
    return next_state

# get the shortest path given an start state
def shortest_path(start_state, rewards_table, q_values):
    start_row = int(start_state[0])
    start_column = int(start_state[1])

    # check is we are in a terminal state
    if is_terminal_state(start_state, rewards_table):
        return []
    
    else: # good start state
        current_state = start_state
        shortest_path = []
        shortest_path.append([current_state])

        # let's continue until we reach a terminal state
        while not is_terminal_state(current_state, rewards_table):
            # let's choose the best action
            action_index = get_next_action(current_state, q_values)
            if action_index == 0: 
                final_actions.append('UP')
            elif action_index == 1:
                final_actions.append('RIGHT')
            else:
                final_actions.append('LEFT')
            # now we move to the next location in the path
            next_state = get_next_location(current_state, action_index)
            shortest_path.append([next_state[0], next_state[1]])

            # update the current state for the next iteration
            current_state = next_state

    return shortest_path

def main():
    # Our initial matrix is a 3,4 so let's inicialize it, and 3 possible actions
    q_values = q_table_create(environment_rows, environment_columns ,3)
    print("\nInitial Q-Table:\n")
    print(q_values)

    # creation of rewards table
    # Define the goal and the obstacle positions
    goal_position = (0, 3)
    obstacle_position = (1, 1)
    rewards_table = rewards_table_create(environment_rows, environment_columns, goal_position, obstacle_position)

    # Assigning the values, to our terminal states!
    rewards_table[0,3] = 100 # end, goal max reward
    rewards_table[1,1] = -100 # whole, MAX PUNISHMENT
    print("\nShowing rewards matrix:\n")
    print(rewards_table)

    # Defininition of our initial state
    #start_state = rewards_table[2,0] #2,0
    start_state = (2,0)

    # Run our algorithm
    q_learning_algorithm(start_state, q_values, rewards_table)

    # print our shortest path
    print("\nShowing Shortest path:\n")
    print(shortest_path(start_state, rewards_table, q_values)) #starting at row 3, column 9

    print("\nShowing sequence of actions:\n")
    print(final_actions)


# This is to run our program, cause looks for a function called "main"
if __name__ == "__main__":
    main()


