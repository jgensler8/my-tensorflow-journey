import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

print env.action_space
print env.observation_space
print env.observation_space.high
print env.observation_space.low

max_reward = 0
max_policy = tf.zeros([4, 1])
NUM_EPISODES = 1000

with tf.Session() as sess:
    for i in range(NUM_EPISODES):
        policy = tf.random_uniform([1, 4], dtype=tf.float64) * 2 - 1

        observation = env.reset()
        total_reward = 0
        for j in range(200):
            mul = tf.matmul(policy, np.reshape(observation, (4, 1)))
            greater = tf.greater(mul, 0)
            result = sess.run(greater)
            if result:
                action = 1
            else:
                action = 0

            observation, reward, done, info = env.step(action)

            total_reward += reward
            if done:
                break

        if total_reward > max_reward:
            pt = sess.run(policy)
            print "Episode {}, Total Reward {}, Policy Tensor {}".format(i, total_reward, pt)
            max_reward = total_reward
            max_policy = policy
