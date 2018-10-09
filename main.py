from agent import Agent
from monitor import interact
import gym
import numpy as np

def play(agent, env):
    answer = input("Wanna play? [Y/N]")
    while answer == 'y':
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            env.render()
            input("Press to step")
            if done:
                print("Final reward: {}".format(samp_reward))
                # save final sampled reward
                break
        answer = input("Wanna play? [Y/N]")

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
play(agent, env)
