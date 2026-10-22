# %% ENVIRONMENT ENGINEERING

# IMPORTING ESSENTIAL LIBRARIES.
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# CLASS AND INITIALIZATION.
class DefiEnvironment(gym.Env):
    metadata = {'render_modes': ["ansi"]}

# RUNNING PARENT INITIALIZATION AND BOUNDARY DECLARATION.
    def __init__(self):
        super(DefiEnvironment, self).__init__()

        # Setting Lower and Upper Bounds and Developing Matrix.
        self.lower_bounds_of_attribute = np.array([
            1000,
            0,
            -0.05
        ], dtype = np.float32)

        self.upper_bounds_of_attribute = np.array([
            100000,
            5000,
            0.05
        ], dtype = np.float32)

        self.observation_space = spaces.Box(
            low = self.lower_bounds_of_attribute,
            high = self.upper_bounds_of_attribute,
            dtype = np.float32
        )

        # Setting Attacker Matrix.
        self.attacker_mapping_space = spaces.Discrete(3)

        self.attacker_loan_size = np.array([
            0,
            1000,
            25000
        ], dtype = np.float32)

        # Setting Defender Matrix.
        self.defender_mapping_space = spaces.Discrete(4)

        self.defender_slippage_taxes = np.array([
            0,
            0.01,
            0.05,
            0.2,
        ], dtype = np.float32)

        self.defender_block_timelock = np.array([
            0,
            0,
            1,
            5
        ], dtype = np.int32)

# ATTACKER AND DEFENDER PARAMETERS.
    @staticmethod
    def attacker_reward_parameters(treasury_exploited, gas_penalties, loan_premium):
            attacker_reward = treasury_exploited - gas_penalties - loan_premium
            return float(attacker_reward)

    @staticmethod
    def defender_reward_parameters(capital_preserved, friction_fee_for_honest_user):
            defender_reward = capital_preserved - friction_fee_for_honest_user
            return float(defender_reward)

# RESET ENGINE SCRIPTING.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Setting Initial Values for the 3 Attributes.
        initial_liquidity_pool_size = 50000
        initial_governance_token_velocity = 0
        initial_oracle_price_discrepancy = 0

        # Packing Initial Values into a 1*3 Matrix.
        self.state = np.array([
            initial_liquidity_pool_size,
            initial_governance_token_velocity,
            initial_oracle_price_discrepancy
        ], dtype = np.float32)

        # Creating Informational Dictionary.
        information_dictionary = {}
        return self.state, information_dictionary

# CORE STATE TRANSITION LOGIC.
    def step(self, action):
        attacker_action, defender_action = tuple(action)

        # Current State Environments.
        current_liquidity_pool_size = self.state[0]
        current_governance_token_velocity = self.state[1]
        current_oracle_price_discrepancy = self.state[2]

        # Agent Choices from Menu.
        loan_size = self.attacker_loan_size[attacker_action]
        slippage_tax = self.defender_slippage_taxes[defender_action]
        block_timelock = self.defender_block_timelock[defender_action]

# CONSTANT-PRODUCT MARKET-MAKER MATH ( x * y = k )
        # Establishing Symmetrical Virtual Liquidity Pool.
        virtual_value_of_x = current_liquidity_pool_size
        virtual_value_of_y = current_liquidity_pool_size
        virtual_value_of_k = virtual_value_of_x * virtual_value_of_y

        # Attacker and Defender Condition.
        if loan_size > 0:
           effective_inflow = loan_size * (1 - slippage_tax)

           # Calculating New Virtual Values.
           new_virtual_value_of_x = virtual_value_of_x + effective_inflow
           new_virtual_value_of_y = virtual_value_of_k / new_virtual_value_of_x
           tokens_given_to_attacker = virtual_value_of_y - new_virtual_value_of_y

           # Measuring Price Slippage and Deviation.
           instantaneous_spot_price = effective_inflow / tokens_given_to_attacker if tokens_given_to_attacker > 0 else 1
           slippage_deviation = float((instantaneous_spot_price - 1) / 1)

           # Bounding Oracle State:
           oracle_price_discrepancy = np.clip(slippage_deviation, -0.05, 0.05)

           # Calculating Governance Token Spikes.
           spike_in_governance_token = np.clip(current_governance_token_velocity + (tokens_given_to_attacker * 0.1), 0, 5000)
        else:
            drift_factor = -0.05 * current_oracle_price_discrepancy
            random_normal_shock = np.random.normal(0,1)
            volatility_shock = 0.01 * random_normal_shock
            updated_discrepancy = current_oracle_price_discrepancy + drift_factor + volatility_shock
            oracle_price_discrepancy = np.clip(updated_discrepancy, -0.05, 0.05)
            random_retail_volume = int(np.random.randint(5, 25))
            spike_in_governance_token = np.clip(current_governance_token_velocity - 10 + random_retail_volume, 0, 5000)
            tokens_given_to_attacker = 0

# GOVERNANCE VALIDATION LOGIC AND CAPITAL EXPLOIT ARBITRATION.
        # Initial Value for Stolen Capital.
        treasury_exploited = 0

        # Conditions for Exploit.
        if loan_size >= 25000 and block_timelock == 0:
            exploit_efficiency = abs(oracle_price_discrepancy) * 2
            treasury_exploited = np.clip(current_liquidity_pool_size * exploit_efficiency, 0, current_liquidity_pool_size)

# REWARD SCORING
        gas_penalties = 50 if loan_size > 0 else 0
        loan_premium = loan_size * 0.003     # Standard Charge Rate.
        friction_fee_for_honest_user = (slippage_tax * 1000) + (float(block_timelock) * 200)
        capital_preserved = current_liquidity_pool_size - treasury_exploited
        # Adding Reward Parameters.
        attacker_reward = self.attacker_reward_parameters(treasury_exploited, gas_penalties, loan_premium)
        defender_reward = self.defender_reward_parameters(capital_preserved, friction_fee_for_honest_user)
        updated_liquidity_pool_size = np.clip(current_liquidity_pool_size - treasury_exploited, 1000, 100000)

# MATRIX INTERFACE
        self.state = np.array([
            updated_liquidity_pool_size,
            spike_in_governance_token,
            oracle_price_discrepancy,
        ], dtype = np.float32)

        # Termination Rules.
        game_terminated = bool(updated_liquidity_pool_size <= 1005)
        game_truncated = False

        # Data Dictionary.
        information = {
            "attacker_loan_deployed": loan_size,
            "defender_tax_applied": slippage_tax,
            "defender_block_timelock": block_timelock,
            "gross_treasury_drained": treasury_exploited,
        }
        return self.state, float(attacker_reward), float(defender_reward),game_terminated, game_truncated, information

    def render(self):
        pass

    def close(self):
        pass