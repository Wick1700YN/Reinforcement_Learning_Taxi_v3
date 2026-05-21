# Taxi-v3 Reinforcement Learning (Gymnasium)

## Overview
This project is based on the **Gymnasium Taxi-v3 environment**, a classic reinforcement learning problem where an agent (taxi) learns to pick up a passenger and drop them off at the correct destination in a 5×5 grid world.

The objective is to train or analyze an agent that learns an optimal policy for navigation using reward-based feedback.

---

## Environment Description

The Taxi environment consists of:

- A 5×5 grid world
- 4 fixed locations (Red, Green, Yellow, Blue)
- A taxi agent
- A passenger with a pickup and destination location

The agent must:
1. Pick up the passenger
2. Navigate to the destination
3. Drop off the passenger successfully

---

## Action Space

The environment has 6 discrete actions:

- 0: Move South  
- 1: Move North  
- 2: Move East  
- 3: Move West  
- 4: Pickup passenger  
- 5: Drop off passenger  

---

## State Space

- 500 possible states:
  - Taxi position (25)
  - Passenger location (4 + in-taxi state)
  - Destination location (4)

---

## Reward System

- +20 → Successful drop-off
- -1 → Each step
- -10 → Illegal pickup/drop-off

Goal: maximize reward while minimizing steps.

---

## Tech Stack

- Python
- Gymnasium (Taxi-v3)
- NumPy
- Reinforcement Learning (Q-learning / policy-based methods if used)

---

## How to Run

```bash
pip install gymnasium
python Taxi.py