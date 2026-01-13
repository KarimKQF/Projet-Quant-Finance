
import numpy as np
import pandas as pd

class TradingEnv:
    """
    A simplified trading environment for Reinforcement Learning.
    The agent interacts with the market by Buying, Selling, or Holding.
    The goal is to maximize portfolio value.
    """
    def __init__(self, initial_balance=10000, max_steps=500):
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.prices = self._generate_prices()
        self.done = False
        
        # State: [Price Trend (0:Down, 1:Flat, 2:Up), Position (0:No, 1:Yes)]
        self.state_space_size = 6 
        self.action_space_size = 3 # 0: Hold, 1: Buy, 2: Sell


    def _generate_prices(self):
        """
        Generates a synthetic price series using the Merton Jump-Diffusion Model.
        Parameters:
            S0: Initial Price
            mu: Drift
            sigma: Volatility
            lambda_j: Jump intensity
            mu_j: Mean of jump size
            sigma_j: Std of jump size
        """
        S0 = 100
        mu = 0.05
        sigma = 0.2
        lambda_j = 0.5
        mu_j = -0.05
        sigma_j = 0.1
        
        dt = 1/252 # Daily steps
        T = self.max_steps / 252
        N = self.max_steps
        
        prices = np.zeros(N + 1)
        prices[0] = S0
        
        for t in range(1, N + 1):
            # Geometric Brownian Motion component
            z1 = np.random.normal()
            diffusion = sigma * np.sqrt(dt) * z1
            
            # Jump component
            # Number of jumps in dt (approximate with Poisson)
            n_jumps = np.random.poisson(lambda_j * dt)
            jump_factor = 0
            if n_jumps > 0:
                # Sum of log-normal jumps
                jump_sum = 0
                for _ in range(n_jumps):
                    jump_sum += np.random.normal(mu_j, sigma_j)
                jump_factor = jump_sum
            
            # Euler-Maruyama discretization for log-price
            # d(ln S) = (mu - 0.5*sigma^2)*dt + sigma*dW + dJ
            drift = (mu - 0.5 * sigma**2) * dt
            
            prices[t] = prices[t-1] * np.exp(drift + diffusion + jump_factor)
            
        return prices

    def reset(self):
        """Resets the environment to the initial state."""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.prices = self._generate_prices()
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        Returns the current state.
        State definition: (Price Trend relative to previous, In Position)
        Trend: 0 if Price < Prev_Price, 1 if Price >= Prev_Price
        Position: 0 if shares_held == 0, 1 if shares_held > 0
        """
        if self.current_step == 0:
            trend = 1 # Assume flat/up at start
        else:
            trend = 1 if self.prices[self.current_step] >= self.prices[self.current_step - 1] else 0
            
        position = 1 if self.shares_held > 0 else 0
        
        return (trend, position)

    def step(self, action):
        """
        Executes an action.
        Action 0: Hold (Do nothing)
        Action 1: Buy (Buy max possible with available balance)
        Action 2: Sell (Sell all shares)
        Includes Transaction Costs (0.1% per trade).
        """
        current_price = self.prices[self.current_step]
        reward = 0
        transaction_cost_pct = 0.001 # 0.1% fee
        
        if self.done:
            return self._get_observation(), 0, True, {}

        # Execute action
        if action == 1: # Buy
            if self.balance >= current_price:
                # Calculate max shares we can buy including fee
                # Cost = Price * Shares * (1 + Fee)
                max_buy_cost = self.balance
                price_with_fee = current_price * (1 + transaction_cost_pct)
                num_to_buy = int(max_buy_cost // price_with_fee)
                
                if num_to_buy > 0:
                    cost = num_to_buy * current_price
                    fee = cost * transaction_cost_pct
                    self.balance -= (cost + fee)
                    self.shares_held += num_to_buy
                
        elif action == 2: # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                fee = revenue * transaction_cost_pct
                self.balance += (revenue - fee)
                self.shares_held = 0
                
        # Calculate Net Worth for reward function (Net Worth = Cash + Shares * Price)
        # Note: We don't deduct liquidation fees in Net Worth calc for simplicity unless liquidating
        prev_net_worth = self._calculate_net_worth(self.current_step - 1) if self.current_step > 0 else self.initial_balance
        current_net_worth = self._calculate_net_worth(self.current_step)
        
        # Reward is the change in net worth (step-based reward)
        reward = current_net_worth - prev_net_worth

        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            self.done = True
            
        next_state = self._get_observation()
        
        info = {'net_worth': current_net_worth, 'price': current_price}
        
        return next_state, reward, self.done, info

    def _calculate_net_worth(self, step):
        if step < 0: return self.initial_balance
        price = self.prices[step]
        val = self.balance + (self.shares_held * price)
        return val
