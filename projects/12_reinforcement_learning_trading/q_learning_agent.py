
import numpy as np
import random

class QLearningAgent:
    """
    A simple Q-Learning Agent.
    Unlikely to beat the market in reality, but demonstrates the concept.
    """
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size # Not strictly used with dict q_table but good for ref
        self.action_size = action_size
        self.alpha = alpha # Learning Rate
        self.gamma = gamma # Discount Factor
        self.epsilon = epsilon # Exploration Rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {} # Dictionary mapping (state_tuple) -> [q_values]

    def get_q_values(self, state):
        """Returns Q-values for a state, initializing if unknown."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        """Updates the Q-table based on the Bellman equation."""
        current_q = self.get_q_values(state)[action]
        
        if done:
            target = reward # No future value if done
        else:
            next_q_values = self.get_q_values(next_state)
            max_future_q = np.max(next_q_values)
            target = reward + self.gamma * max_future_q
            
        # Update rule: Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action] = new_q
        
    def decay_epsilon(self):
        """Reduces exploration rate over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
