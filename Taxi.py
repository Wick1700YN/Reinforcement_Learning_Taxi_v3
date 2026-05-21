import random
import gym
import pickle
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save path in same directory
save_path = os.path.join(script_dir, "q_table.pkl")

env = gym.make('Taxi-v3', render_mode='ansi')

# =========================
# LOAD OR TRAIN
# =========================
if os.path.exists(save_path):

    print("Trained model found. Loading Q-table...")

    with open(save_path, "rb") as f:
        q = pickle.load(f)

else:

    print("No trained model found. Training started...")

    # Initialize Q-table
    q = {}
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            q[(s, a)] = 0.0

    # Q-learning update
    def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
        qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
        q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

    # Epsilon-greedy policy
    def epsilon_greedy_policy(state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            return max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])

    # Hyperparameters
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.1

    # =========================
    # TRAINING LOOP
    # =========================
    for i in range(8000):

        r = 0
        prev_state = env.reset()[0]

        while True:

            # (optional visualization)
            print(env.render())

            action = epsilon_greedy_policy(prev_state, epsilon)

            nextstate, reward, done, truncated, _ = env.step(action)

            update_q_table(prev_state, action, reward, nextstate, alpha, gamma)

            prev_state = nextstate
            r += reward

            if done or truncated:
                break

        print("Total reward:", r)

    env.close()

    # Save model
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
