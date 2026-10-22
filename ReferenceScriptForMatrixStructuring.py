# %% REFERENCE SCRIPT FOR MATRIX STRUCTURING.

# IMPORTING ESSENTIAL LIBRARIES.
import gymnasium as gym
from gymnasium import spaces
import numpy as np

"""
The matrix tracks 3 attributes:
1. Liquidity Pool Valuation Size (Total Capital Locked)  --> Index is [0].
2. Governance Token Velocity (Speed of Asset Movement)  --> Index is [1].
3. Oracle Price Discrepancy (Price Discrepancy Ratio)  --> Index is [2].

--------------------------------------------------------------------------------------------------

The 3 attacker choices:
1. Pass  --> Index is [0].
2. Small  --> Index is [1].
3. Large  --> Index is [2].

There are 3 flash loan size choices (in thousands) corresponding to the 3 attacker action spaces:
1. $ 0  --> Index is [0].
2. $ 1000  --> Index is [1].
3. $ 25000  --> Index is [2].

The attacker action space has 3 choices:
1. Pass (Flash Loan Size is $0)  --> Index is [0].
2. Small (Flash Loan Size is $1000)  --> Index is [1].
3. Large (Flash Loan Size is $25000)  --> Index is [2].

--------------------------------------------------------------------------------------------------

The dynamic slippage tax scale has 4 values:
1. 0%  --> Standard Friction Free Baseline.
2. 1%  --> Upper Boundary of a Standard Retail Transaction Fee.
3. 5%  --> Used During High Volatile Events.
4. 20%  --> To Halt Hyper-aggressive Automated Drain Strategies.

There are 3 block time-locks:
1. 0 Block  --> Instant Settlement Within Exact Same Transaction Block.
2. 1 Block  --> Forces Transaction to Split Across Block Boundary.
3. 5 Block  --> Extended Lock-up Duration.

The defender action space has 4 operational levels:
1. Passive State (0% Tax, 0 Block Lock)  --> Index is [0].
2. Low Defense State (1% Tax, 0 Block Lock)  --> Index is [1].
3. Medium Defense State (5% Tax, 1 Block Lock)  --> Index is [2].
4. High Defense State (20% Tax, 5 Block Lock)  --> Index is [3].
"""

# CLASS AND INITIALIZATION.
class simulation_object:
    def __init__(self):

        # Setting Lower Bounds for the Attributes.
        self.lower_bounds_of_attribute = np.array([
            1000.0,
            0.0,
            -0.05
        ], dtype=np.float32)

        # Setting Upper Bounds for the Attributes.
        self.upper_bounds_of_attribute = np.array([
            100000.0,
            5000.0,
            0.05
        ], dtype=np.float32)

        #  Developing the Matrix.
        self.observation_space = spaces.Box(
            low = self.lower_bounds_of_attribute,
            high = self.upper_bounds_of_attribute,
            dtype = np.float32
        )

        # Developing Attacker Mapping Space.
        self.attacker_mapping_space = spaces.Discrete(3)
        self.attacker_loan_sizes = np.array([
            0.0,
            1000.0,
            25000.0
        ], dtype=np.float32)

        # Developing Defender Mapping Spaces.
        self.defender_mapping_space = spaces.Discrete(4)
        self.defender_slippage_taxes = np.array([
            0.0,
            0.01,
            0.05,
            0.20
        ], dtype=np.float32)
        self.defender_block_timelock = np.array([
            0,
            0,
            1,
            5
        ], dtype=np.int32)

# ALGORITHMIC OBJECTIVE REWARD LOOPS.
    # Calculating Attacker Reward.
    def attacker_reward_parameters(self, treasury_exploited, gas_penalties, loan_premium):
        attacker_reward = treasury_exploited - gas_penalties - loan_premium
        return float(attacker_reward)

    # Calculating Defender Reward.
    def defender_reward_parameters(self, capital_preserved, honest_user_friction_fee):
        defender_reward = capital_preserved - honest_user_friction_fee
        return float(defender_reward)

# TESTING AND VERIFICATION ENVIRONMENT.
    # Deriving a random sample from the matrix to test if it works correctly.
    def matrix_testing(self):
       matrix_testing_sample = self.observation_space.sample()

       print("RANDOM SAMPLE VERIFICATION TEST")
       print("The vector shape is " + str(self.observation_space.shape[0]) + ", which is a 1*3 matrix.")
       print("Liquidity pool valuation size of the testing sample is " + str(round(matrix_testing_sample[0],2)))
       print("Governance token velocity of the testing sample is " + str(round(matrix_testing_sample[1],2)))
       print("Oracle price discrepancy of the testing sample is " + str(round(matrix_testing_sample[2],2)))
       print("-------------------------------------------------------------------------")

    # Deriving a random sample from the attacker mapping space to test if it works correctly.
    def attacker_testing(self):
        selected_attacker_index = self.attacker_mapping_space.sample()
        loan_size_of_selected_attacker_index = self.attacker_loan_sizes[selected_attacker_index]

        print("ATTACKER SAMPLE VERIFICATION TEST")
        print("The selected attacker action index is " + str(selected_attacker_index))
        print("Loan size of selected attacker index is " + str(loan_size_of_selected_attacker_index))
        print("-------------------------------------------------------------------------")

    # Deriving a random sample from the defender mapping space to test if it works correctly.
    def defender_testing(self):
        selected_defense_index = self.defender_mapping_space.sample()
        slippage_tax_of_selected_defender_index = self.defender_slippage_taxes[selected_defense_index]
        block_timelock_of_selected_defender_index = self.defender_block_timelock[selected_defense_index]

        print("DEFENDER SAMPLE VERIFICATION TEST")
        print("The selected defender action index is " + str(selected_defense_index))
        print("The slippage tax of the selected defender index is " + str(slippage_tax_of_selected_defender_index * 100) + "%")
        print("The block timelock parameter of the selected defender index is " + str(block_timelock_of_selected_defender_index) + " blocks")
        print("-------------------------------------------------------------------------")

    # Testing a Random Sample to Check if Reward System is Understood.
    def reward_testing(self):
        attacker_reward_score = self.attacker_reward_parameters(
            treasury_exploited = 5000,
            gas_penalties = 50,
            loan_premium = 15
        )

        defender_reward_score = self.defender_reward_parameters(
            capital_preserved = 25000,
            honest_user_friction_fee = 250
        )

        print("SCALAR REWARD TEST")
        print("The attacker reward is " + str(attacker_reward_score))
        print("The defender reward is " + str(defender_reward_score))


if __name__ == "__main__":
    attribute_matrix = simulation_object()
    attribute_matrix.matrix_testing()
    attribute_matrix.attacker_testing()
    attribute_matrix.defender_testing()
    attribute_matrix.reward_testing()