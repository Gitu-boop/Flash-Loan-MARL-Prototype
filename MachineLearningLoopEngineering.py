# %% MACHINE LEARNING LOOP ENGINEERING.

# IMPORTING ESSENTIAL LIBRARIES.
import os
import csv
import numpy as np
import random
import importlib
from EnvironmentEngineering import DefiEnvironment

# RANDOM SEED CONFIGURATION
random.seed(5)
np.random.seed(5)

# BUILDING Q-LEARNING AGENT FOR MULTI-AGENT REINFORCEMENT LEARNING SETUP
class QLearningAgent:
    def __init__(self, options_available_to_agents, discrete_slots_for_memory_grid = (10,10,10)):
        self.options_available_to_agents = options_available_to_agents
        self.discrete_slots_for_memory_grid = discrete_slots_for_memory_grid
        q_learning_table_shape = self.discrete_slots_for_memory_grid + (self.options_available_to_agents,)
        self.q_table = np.zeros(shape=q_learning_table_shape, dtype = np.float32)

# INITIALIZING SIMULATION PLAYGROUND AND BUILDING INDEPENDENT BRAINS.
class SimulationManager:
    def __init__(self):
     self.environment = DefiEnvironment()
     self.attacker_agent = QLearningAgent(options_available_to_agents = 3)
     self.defender_agent = QLearningAgent(options_available_to_agents = 4)
     self.integer_buckets = 10

     # Selection Engine Parameters using Epsilon Greedy Decay Schedule.
     self.epsilon = 1
     self.minimum_epsilon_floor = 0.01
     self.epsilon_decay_multiplier = 0.9995
     self.alpha = 0.1
     self.gamma = 0.95

    # Translating Raw Numbers.
    def translating_environment (self, raw_numbers):
        liquidity_pool_size, governance_token_velocity, oracle_price_discrepancy = raw_numbers

        liquidity_pool_size_cleaned_values = int(np.clip((liquidity_pool_size - 1000) / (100000 - 1000) * self.integer_buckets,0, self.integer_buckets - 1))
        governance_token_velocity_cleaned_values = int(np.clip((governance_token_velocity - 0) / (5000 - 0) * self.integer_buckets, 0, self.integer_buckets - 1))
        oracle_price_discrepancy_cleaned_values = int(np.clip((oracle_price_discrepancy - (-0.05)) / (0.05 - (-0.05)) * self.integer_buckets, 0, self.integer_buckets - 1))
        return (liquidity_pool_size_cleaned_values, governance_token_velocity_cleaned_values, oracle_price_discrepancy_cleaned_values)

    # Epsilon Greedy Action Selection Strategy for Attacker.
    def attacker_action_selection(self, discrete_state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.attacker_agent.options_available_to_agents))
        else:
            return int(np.argmax(self.attacker_agent.q_table[discrete_state]))

    # Epsilon Greedy Action Selection Strategy for Defender.
    def defender_action_selection(self, discrete_state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.defender_agent.options_available_to_agents))
        else:
            return int(np.argmax(self.defender_agent.q_table[discrete_state]))

    # Dual-Agent Competitive Optimization Pipeline.
    def executing_training_pipeline(self, total_episodes = 10000):
        print("The reinforcement learning run has " + str(total_episodes) + " episodes")
        for episode in range(total_episodes):
            raw_state, _ = self.environment.reset()
            discrete_state = self.translating_environment(raw_state)
            game_over = False
            step_count = 0

            while not game_over and step_count < 100:
                attacker_action = self.attacker_action_selection(discrete_state)
                defender_action = self.defender_action_selection(discrete_state)
                joint_action = [attacker_action, defender_action]
                raw_next_state, attacker_reward, defender_reward, game_terminated, game_truncated, _ = self.environment.step(joint_action)
                discrete_next_state = self.translating_environment(raw_next_state)

                # Recalculating Attacker Strategy.
                old_attacker_strategy = self.attacker_agent.q_table[discrete_state + (attacker_action,)]

                maximum_future_attacker_strategy = np.max(self.attacker_agent.q_table[discrete_next_state])
                self.attacker_agent.q_table[discrete_state + (attacker_action,)] = old_attacker_strategy + self.alpha * (attacker_reward + self.gamma * maximum_future_attacker_strategy - old_attacker_strategy)

                # Recalculating Defender Strategy.
                old_defender_strategy = self.defender_agent.q_table[discrete_state + (defender_action,)]

                maximum_future_defender_strategy = np.max(self.defender_agent.q_table[discrete_next_state])
                self.defender_agent.q_table[discrete_state + (defender_action,)] = old_defender_strategy + self.alpha * (defender_reward + self.gamma * maximum_future_defender_strategy - old_defender_strategy)

                # Advance Tracking Environment Parameters.
                discrete_state = discrete_next_state
                game_over = game_terminated or game_truncated
                step_count += 1

            # Trailing Decay Schedule.
            if self.epsilon > self.minimum_epsilon_floor:
                self.epsilon *= self.epsilon_decay_multiplier
            else:
                self.epsilon = self.minimum_epsilon_floor

            # Creating Status Reporter to Check Workability.
            if(episode + 1) % 1000 == 0:
                print("The Epsilon or core exploration boundary of episode " + str(episode + 1) + " is " + str(round(self.epsilon * 100, 2)) + "%")


if __name__ == "__main__":
    master_training_pipeline = SimulationManager()
    master_training_pipeline.executing_training_pipeline(total_episodes = 10000)
