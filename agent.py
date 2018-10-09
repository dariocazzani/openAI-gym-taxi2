import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.9
        self.gamma = 0.9
        self.epsilon = 5E-4
        self.policy_s = 1. / nA

    def compute_epsilon_policy(self, state):
        # the action probabilities
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        return policy_s

    def update_Q(self, Qsa, Qsa_next, reward,):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy_s = self.compute_epsilon_policy(state)
        action = np.random.choice(np.arange(self.nA), p=self.policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # update Q
        expected_reward = np.dot(self.policy_s, self.Q[next_state])
        self.Q[state][action] = self.update_Q(self.Q[state][action], expected_reward, reward)
