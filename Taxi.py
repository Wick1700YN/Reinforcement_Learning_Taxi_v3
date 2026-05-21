import random
import gym
import pickle
import os


# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save path in same directory
save_path = os.path.join(script_dir, "q_table.pkl")
env = gym.make('Taxi-v3', render_mode='ansi')

if os.path.exists(save_path):
    print("Trained model found. Loading Q-table...")

    with open(save_path, "rb") as f:
        q = pickle.load(f)

else:
    print("No trained model found. Training started...")

    # Create the environment with a specific rendering mode

    # Initialize the environment and render the initial state
    prev_state = env.reset()[0]  # Extract the state from the returned tuple
    print(env.render())

    # Initialize Q-table as a dictionary
    # Q-table will store Q-values for all state-action pairs
    q = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            q[(s, a)] = 0.0

    # Function to update the Q-table using the Q-learning update rule
    def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
        # Find the maximum Q-value for the next state
        qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
        # Update the Q-value for the (state, action) pair
        q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

    # Function to select an action using the epsilon-greedy policy
    def epsilon_greedy_policy(state, epsilon):
        if random.uniform(0, 1) < epsilon:
            # Explore: Select a random action
            return env.action_space.sample()
        else:
            # Exploit: Select the action with the highest Q-value for the current state
            return max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])

    # Set the parameters for the Q-learning algorithm
    alpha = 0.4  # Learning rate
    gamma = 0.999  # Discount factor
    epsilon = 0.017  # Exploration rate

    # Run the Q-learning algorithm for a certain number of episodes
    for i in range(8000):
        r = 0  # Initialize total reward for the current episode

        # Reset the environment to the initial state
        prev_state = env.reset()[0]  # Extract the state from the returned tuple

        while True:
            # Render the environment at each step
            print(env.render())

            # Select the action based on the epsilon-greedy policy
            action = epsilon_greedy_policy(prev_state, epsilon)

            # Perform the action and move to the next state, and receive the reward
            nextstate, reward, done, truncated, _ = env.step(action)  # Extract truncated as well

            # Update the Q-value for the (prev_state, action) pair
            update_q_table(prev_state, action, reward, nextstate, alpha, gamma)

            # Update the previous state to be the current state
            prev_state = nextstate

            # Accumulate the reward
            r += reward

            # Break the loop if we reach the terminal state
            if done or truncated:
                break

        # Print the total reward obtained in the current episode
        print("Total reward: ", r)

    # Close the environment
    env.close()

    # Save the trained Q-table to a file
    with open(save_path, "wb") as f:
        pickle.dump(q, f)

    print(f"Q-table saved at: {save_path}")

# =========================
# LOAD TRAINED AGENT
# =========================
with open(save_path, "rb") as f:
    q = pickle.load(f)

print("Q-table loaded successfully!")

# =========================
# TEST TRAINED AGENT
# =========================
print("\nTesting trained agent...\n")

state = env.reset()[0]
done = False

while not done:
    print(env.render())

    action = max(range(env.action_space.n), key=lambda a: q[(state, a)])

    state, reward, done, truncated, _ = env.step(action)

    if done or truncated:
        break

env.close()