from re import T
import gym
import math
import random
import numpy as np
from absl import app
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p
import tensorflow as tf

env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)

from typing import Any, List, Sequence, Tuple
from models import ActorCriticPolicy
from agents import ActorCriticAgent

# Create the environment
min_episodes_criterion = 100
max_episodes = 5000
steps_per_episode = 500
gamma = 0.9

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

def plot_rewards(running_rewards):
    ########## Code starts here ##########
    # Plot rewards per episode.
    plt.plot(running_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.savefig('rewards.png')
    ########## Code ends here ##########
    pass

def normalize(state):
    state = state / 255.
    return state

def main(argv):

    model = ActorCriticPolicy(input_shape=env.observation_space.shape, action_dim=env.action_space.n)
    agent = ActorCriticAgent(model=model)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    episodes_reward = []

    for i in tf.range(max_episodes):
        # This is your outer training loop for each episode. You will run a `steps` number of steps in each episode.

        done = False
        state = env.reset()
        episode_reward = 0
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        with tf.GradientTape() as tape:
            for j in tf.range(steps_per_episode):
            # This is your main training loop per step of each episode
            # Hint: Don't forget to normalize your state before using it. A function is provided for you.

            ########## Code starts here ##########
                action, log_prob, value = agent.act(normalize(state)) # choose action and get model feedback
                next_state, reward, done, _ = env.step(action) # apply action in env
                state = next_state # advance state for next iteration
                
                rewards = rewards.write(j, reward) # store rewards
                values = values.write(j, value) # store critic values
                log_probs = log_probs.write(j, log_prob) # store log probability of chosen action

                # end loop when done!
                if done:
                    break
            
            # pack into a single tensor
            log_probs = log_probs.stack()
            values = values.stack()
            rewards = rewards.stack()

            # calculate expected returns and loss
            returns = agent.compute_expected_return(rewards, gamma)
            log_probs, values, returns = [tf.expand_dims(x, 1) for x in [log_probs, values, returns]] 
            loss = agent.compute_loss(log_probs, returns, values)
            ########## Code ends here ##########

        # Update your gradients and optimizer.
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        # Sum up your episode rewards.
        episode_reward = tf.math.reduce_sum(rewards)

        episodes_reward.append(float(episode_reward))

        # Calculate your running reward average.
        running_reward = tf.math.reduce_mean(episodes_reward)

        # Print your running reward every 10 episodes.
        if i % 10 == 0:
            tf.print('Running reward after {} episodes is {}'.format(i, running_reward))

        # Plot rewards per episode.
        # Hint: you will want to aggregate a `running_rewards` list of episode running rewards to pass in.
        plot_rewards(running_reward)

if __name__ == '__main__':
  app.run(main)
