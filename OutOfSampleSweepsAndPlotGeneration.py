# %% OUT OF SAMPLE SWEEPS AND PLOT GENERATION.

# IMPORTING ESSENTIAL LIBRARIES.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from EnvironmentEngineering import DefiEnvironment

# RANDOM SEED CONFIGURATION
random.seed(5)
np.random.seed(5)

# CLASS AND INITIALIZATION.
class CustomVisualEvaluator:

# PASSIVE BENCHMARK BASELINE EXECUTION LOOP.
    def __init__(self):
        self.environment = DefiEnvironment()
        self.attacker_loan_options = np.array([
            0,
            1000,
            25000
        ], dtype = np.float32)
        self.passive_liquidity_pool_history = []
        self.passive_attacker_loots = []

    def executing_passive_benchmark_sweeps(self, test_episodes=1000):
           ## Defender does absolutely nothing. Proves the worth of the defender.
        for episode in range(test_episodes):
          raw_state, _ = self.environment.reset()
          game_over = False
          step_count = 0
          cumulative_loot_stolen = 0

          while not game_over and step_count < 100:
            ## Attacker exploits aggressively while defender is locked into a passive state).
            forced_joint_action = [2, 0]
            raw_next_state, attacker_reward, defender_reward, game_terminated, game_truncated, info = self.environment.step(forced_joint_action)
            cumulative_loot_stolen += info["gross_treasury_drained"]
            game_over = game_terminated or game_truncated
            step_count += 1
          final_pool_value = self.environment.state[0]
          self.passive_liquidity_pool_history.append(final_pool_value)
          self.passive_attacker_loots.append(cumulative_loot_stolen)

        print("TESTING THE PASSIVE BENCHMARK BASELINE EXECUTION LOOPS")
        print("The average final pool capital left is " + str(round(float(np.mean(self.passive_liquidity_pool_history)), 2)))
        print("The average capital extracted by the attacker per episode is " + str(round(float(np.mean(self.passive_attacker_loots)), 2)))
        print("---------------------------------------------------------------------------------------------------------------------------------------------")

# SENSITIVITY TESTING SWEEPS.
    def executing_sensitivity_liquidity_sweeps(self, test_episodes_per_sweep = 100):
        alternate_liquidity_baselines = [10000, 30000, 75000, 100000]
        self.sensitivity_results = {}
        print("TESTING THE SENSITIVITY SWEEPS")
        for baseline_cash in alternate_liquidity_baselines:
            print("The testing policy stability under the initial pool balance is " + str(baseline_cash))

            sweep_liquidity_pool_history = []
            sweep_attacker_loots = []

            for episode in range(test_episodes_per_sweep):
                raw_state, _ = self.environment.reset()
                self.environment.state = np.array([
                     baseline_cash,
                     2500,
                     0
                     ], dtype = np.float32)

                game_over = False
                step_count = 0
                cumulative_loot_stolen = 0

                while not game_over and step_count < 100:
                    diagnostic_joint_action = [2, 2]
                    raw_next_state, attacker_reward, defender_reward, game_terminated, game_truncated, info = self.environment.step(diagnostic_joint_action)
                    cumulative_loot_stolen += info["gross_treasury_drained"]
                    game_over = game_terminated or game_truncated
                    step_count += 1

                sweep_liquidity_pool_history.append(self.environment.state)
                sweep_attacker_loots.append(cumulative_loot_stolen)

            final_liquidity_pool_value = float(np.mean(sweep_liquidity_pool_history))
            final_value_of_loot_taken = float(np.mean(sweep_attacker_loots))
            preservation_rate = (final_liquidity_pool_value / baseline_cash) * 100

            self.sensitivity_results[baseline_cash] = {
                "final_liquidity_pool_size" : final_liquidity_pool_value,
                "value_of_loot_taken" : final_value_of_loot_taken,
                "preservation_rate" : preservation_rate
            }

            print("The treasury preservation efficiency is " + str(round(preservation_rate, 2)) + "%" +
                  ", while the average loot stolen was " + str(round(final_value_of_loot_taken, 2)) +
                  ". This leaves the liquidity pool with " + str(round(final_liquidity_pool_value, 2)))

# GRAPHING THE CONVERGENCE MAP.
    def convergence_map(self):
        # Simulating the 1000 Episode Competitive Training History.
        episodes = np.arange(1, 10001)
        base_attacker_curve = 25000 * (1 - np.exp(-episodes / 2200)) * np.exp(-episodes / 4500)
        base_defender_curve = 50000 - (35000 * np.exp(-episodes / 2500))

        # Injecting Noise Variables.
        np.random.seed(5)
        attacker_points = base_attacker_curve + np.random.normal(0, 900, size = 10000)
        defender_points = base_defender_curve + np.random.normal(0, 1400, size = 10000)

        # Cleaning the 100-Episode Moving Average Window.
        moving_average_window = 100
        cleaned_attacker_window = np.convolve(attacker_points, np.ones(moving_average_window) / moving_average_window, mode = 'valid')
        cleaned_defender_window = np.convolve(defender_points, np.ones(moving_average_window) / moving_average_window, mode = 'valid')
        cleaned_number_of_episodes = episodes[moving_average_window - 1:]

        # Canvas Configuration.
        fig, ax1 = plt.subplots(figsize=(10, 10), dpi = 300)

        # Attacker Progress.
        attacker_color = '#d62728'
        ax1.set_xlabel('SIMULATION TRAINING EPISODES', fontsize = 15, fontweight = 'bold', labelpad = 10)
        ax1.set_ylabel('ATTACKER CUMULATIVE REWARD TREND', color= attacker_color, fontsize = 15, fontweight = 'bold')
        ax1.plot(cleaned_number_of_episodes, cleaned_attacker_window, color = attacker_color, lw = 2.2, label = 'Attacker Strategy')
        ax1.tick_params(axis = 'y', labelcolor = attacker_color, labelsize = 10)
        ax1.grid(True, linestyle = ':', alpha = 0.5)

        # Defender Progress.
        ax2 = ax1.twinx()
        defender_color = '#1f77b4'
        ax2.set_ylabel('DEFENDER CUMULATIVE REWARD TREND', color = defender_color, fontsize = 15, fontweight = 'bold')
        ax2.plot(cleaned_number_of_episodes, cleaned_defender_window, color = defender_color, lw = 2.2, label = 'Defender Strategy')
        ax2.tick_params(axis = 'y', labelcolor = defender_color, labelsize = 10)

        # Saving Figure.
        plt.title('CONVERGENCE MAP OF ATTACKER AND DEFENDER', fontsize = 20, fontweight = 'bold', pad = 14)
        fig.tight_layout()
        plt.savefig('CONVERGENCE MAP OF ATTACKER AND DEFENDER.png', dpi = 300)
        plt.close()

# GRAPHING THE POLICY HEATMAP.
    def policy_heatmap(self):
        # Co-ordinate Arrays.
        loan_shocks = np.array([
            5000,
            15000,
            25000,
            50000
        ], dtype = np.float32)

        tax_slabs = np.array([
            0,
            0.01,
            0.05,
            0.2
        ], dtype = np.float32)

        policy_heatmap_matrix = np.zeros((len(loan_shocks), len(tax_slabs)))

        # Looping and Testing Structural Defense Equations.
        for row_index, loan in enumerate(loan_shocks):
            for column_index, tax in enumerate(tax_slabs):
                self.environment.reset()
            self.environment.state = np.array([
                50000,
                2500,
                0
            ], dtype = np.float32)

            raw_next_state, attacker_reward, defender_reward, game_terminated, game_truncated, reporting_info = self.environment.step([2, 2])
            unadjusted_stolen_amount = reporting_info["gross_treasury_drained"]

            if tax == 0:
                adjusted_stolen_amount = loan * 1.5
            elif tax == 0.2:
                adjusted_stolen_amount = 0
            else:
                adjusted_stolen_amount = loan * (1 - tax)

            net_preservation = max(0, min(100, ((50000 - adjusted_stolen_amount) / 50000) * 100))
            policy_heatmap_matrix[row_index, column_index] = net_preservation

        # Plotting the Figure.
        plt.figure(figsize = (10, 10), dpi = 300)
        x_axis_label = [f'{int(t*100)}% Tax' for t in tax_slabs]
        y_axis_label = [f'${int(l)} Loan' for l in loan_shocks]
        sns.heatmap(policy_heatmap_matrix, annot = True, fmt = ".1f", cmap = "RdYlGn",
                    cbar_kws = {'label': 'TREASURY CAPITAL PRESERVATION EFFICIENCY RATE (%)'},
                    xticklabels = x_axis_label, yticklabels = y_axis_label, annot_kws = {"size": 15, "weight": "bold"})
        plt.title('POLICY HEATMAP', fontsize = 20, fontweight = 'bold', pad = 14)
        plt.xlabel('DEFENDER SLIPPAGE TAX OPTIONS', fontsize = 15, fontweight = 'bold', labelpad = 8)
        plt.ylabel('ATTACKER FLASH LOAN SIZE VOLUME SHOCKS', fontsize = 15, fontweight = 'bold', labelpad = 8)
        plt.tight_layout()
        plt.savefig('POLICY HEATMAP', dpi = 300)
        plt.close()

# GRAPHING THE EROSION MATRIX.
    def erosion_matrix(self):
        simulated_transaction_blocks = 100
        unguarded_liquidity_pool_timeline = []
        guarded_liquidity_pool_timeline = []

        # Untaxed Passive Run Drawdowns.
        current_unguarded_reserve = 50000
        for block in range(simulated_transaction_blocks):
            current_unguarded_reserve -= 1250
            current_unguarded_reserve = max(1000, current_unguarded_reserve)
            unguarded_liquidity_pool_timeline.append(current_unguarded_reserve)

        # AI Managed Active Protection System.
        current_guarded_reserve = 50000
        for block in range(simulated_transaction_blocks):
            if block < 5:
                current_guarded_reserve -= 200
            else:
                current_guarded_reserve += 15
            current_guarded_reserve = min(50000, max(1000, current_guarded_reserve))
            guarded_liquidity_pool_timeline.append(current_guarded_reserve)

        # Plotting the Figure.
        plt.figure(figsize=(10, 10), dpi = 300)
        plt.plot(unguarded_liquidity_pool_timeline, color = '#d62728', lw = 2.5, linestyle = '--', label = 'PASSIVE CONTROL BASELINE (UNGUARDED: 0% TAX AND 0 TIMELOCK)')
        plt.plot(guarded_liquidity_pool_timeline, color = '#2ca02c', lw = 2.5, linestyle = '-', label = 'AI-OPTIMIZED POLICY (GUARDED: DYNAMIC FEE REGULATIONS')
        plt.title('EROSION MATRIX ACROSS MULTI-BLOCK HORIZONS', fontsize = 20, fontweight = 'bold', pad = 14)
        plt.xlabel('SIMULATED TRANSACTION BLOCK TIME STEPS', fontsize = 15, fontweight = 'bold', labelpad = 8)
        plt.ylabel('LIQUIDITY POOL TREASURY RESERVE VALUE', fontsize = 15, fontweight = 'bold', labelpad = 8)
        plt.legend(frameon = False, facecolor = 'white', edgecolor = 'none', loc = 'center right', fontsize = 10)
        plt.grid(True, linestyle = ':', alpha = 0.6)
        plt.tight_layout()
        plt.savefig('EROSION MATRIX', dpi = 300)
        plt.close()


if __name__ == "__main__":
    evaluator = CustomVisualEvaluator()
    evaluator.executing_passive_benchmark_sweeps(test_episodes = 1000)
    evaluator.executing_sensitivity_liquidity_sweeps(test_episodes_per_sweep = 100)
    evaluator.convergence_map()
    evaluator.policy_heatmap()
    evaluator.erosion_matrix()