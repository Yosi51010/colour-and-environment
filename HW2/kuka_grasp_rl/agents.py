import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class ActorCriticAgent():

    def __init__(self, model):
        super(ActorCriticAgent, self).__init__()
        self.model = model
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    def act(self, state):
        """Select an action based on the policy output.
        Return: the action, the action log probabilities of this action sample
        and the value from the policy output"""

        # Hint: we are working with a discrete version of the Kuka grasping env
        # so you can use a Categorical distribution.

        ########## Code starts here ##########
        # Run the model and to get action logits (they can be negative!!) and critic values
        action_logits, value = self.model(np.array([state])) 
        dist = tfp.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        # convert logits to probability
        action_probs = tf.nn.softmax(action_logits) 
        # log probabilities
        action_log_probs = tf.math.log(action_probs)
        ########## Code ends here ##########
        return int(np.argmax(action)), action_log_probs, value

    def compute_loss(self, log_probs, returns, values):
        """Compute the actor loss and the critic loss separately and add them together."""

        ########## Code starts here ##########
        # actor loss
        advantage = returns - values 
        actor_loss = -tf.math.reduce_sum(log_probs * advantage)

        # critic loss
        critic_loss = self.huber_loss(values, returns)

        total_loss = actor_loss + critic_loss
        ########## Code ends here ##########
        return total_loss

    def compute_expected_return(self, rewards, gamma):
        """Compute the expected returns."""

        # Hint #1: Start from the rewards in reverse, it'll make your calculation easier to code.
        # Hint #2: Don't forget to normalize your returns in the end by subtracting the mean and dividing by the std plus some epsilon to avoid any divisions by 0.
        
        ########## Code starts here ##########
        eps = np.finfo(np.float32).eps.item() # Small epsilon value for stabilizing division operations

        n = tf.shape(rewards)[0] # rewards is a Tensor
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Reversed rewards and accumulating reward sums into the returns array

        rewards = tf.cast(rewards[::-1], dtype=tf.float32) # casting to remove error 
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        # standardization
        returns = ((returns - tf.math.reduce_mean(returns))/(tf.math.reduce_std(returns) + eps))
        ########## Code ends here ##########

        return returns
