import gym
import gym.wrappers as wrappers
import tensorflow as tf
import numpy as np
import os

output_dir = '/tmp/cartpole-random-hill-climbing'
env = gym.make('CartPole-v0')
UPLOAD = os.environ.get("UPLOAD", False)

if UPLOAD:
    env = wrappers.Monitor(env, output_dir, force=True)
    api_key = os.environ.get("APIKEY")

DEBUG = os.environ.get("DEBUG", False)

if DEBUG:
    print env.action_space
    print env.observation_space
    print env.observation_space.high
    print env.observation_space.low

max_reward = 0
max_policy = tf.zeros([1, 4], dtype=tf.float64)
# max_policy = tf.constant([[ 0.0042116,  -0.00097569,  0.00027187,  0.01684492]], dtype=tf.float64)
NUM_EPISODES = 500
shrink = .05

with tf.Session() as sess:

    writer = tf.summary.FileWriter('/tmp/tf/logs', sess.graph)
    reward_p = tf.Variable(0.)
    tf.summary.scalar("Reward", reward_p)
    merged = tf.summary.merge_all()
    # summary = sess.run(merged)

    for i in range(NUM_EPISODES):
        print i

        observation = env.reset()
        total_reward = 0
        should_perturb = True

        if should_perturb:
            axis = i % 4
            step = (tf.random_uniform([1], dtype=tf.float64) - .5) * shrink
            perturb = np.zeros(4)
            mul_op = max_policy[0][axis] + step
            perturb[axis] = sess.run(mul_op)
            policy_op = tf.add(max_policy, perturb)

            ob_placeholder = tf.placeholder(tf.float64, [4, 1])
            mul = tf.matmul(policy_op, ob_placeholder)
            greater = tf.greater(mul, 0)

        if DEBUG:
            # print perturb
            policy = sess.run(policy_op)
            print policy
            m_p = sess.run(max_policy)
            print m_p

        for j in range(200):
            if DEBUG:
                print i, j

            result = sess.run(greater, feed_dict={ob_placeholder: np.reshape(observation, (4, 1))})
            if result:
                action = 1
            else:
                action = 0

            observation, reward, done, info = env.step(action)

            total_reward += reward
            if done:
                if DEBUG:
                    print total_reward
                break

            # env.render()

        summary = sess.run(merged, feed_dict={reward_p: total_reward})
        writer.add_summary(summary, global_step=i)

        if i % 10:
            writer.flush()

        if total_reward > max_reward:
            pt = sess.run(policy_op)
            print "Episode {}, Total Reward {}, Policy Tensor {}".format(i, total_reward, pt)
            max_reward = total_reward
            max_policy = policy_op
            if total_reward > 190:
                should_perturb = False

if UPLOAD:
    env.close()
    gym.upload(output_dir, api_key=api_key)