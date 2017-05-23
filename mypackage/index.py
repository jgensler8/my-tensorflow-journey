import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import tflearn
import random
from collections import deque
from replay_buffer import ReplayBuffer
from os import environ

class Actor(object):
    def __init__(self, session, env, action_repeat):
        self.session = session
        self.env = env
        self.action_repeat = action_repeat
        self.state_dim = self.env.observation_space.shape[0]
        # self.action_dim = self.env.action_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.learning_rate = 0.0001
        self.tau = 0.001

        # self.inputs = tflearn.input_data(shape=[action_repeat, self.state_dim], name="inputs")
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # net = tflearn.fully_connected(self.inputs, 4, activation='relu', name="FirstLayer", weights_init=w_init)
        # w_init_2 = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # self.out = tflearn.fully_connected(net, self.action_dim, name="FinalLayer", weights_init=w_init_2)

        self.inputs = tflearn.input_data(shape=[None, self.state_dim], name="Actor-Inputs")
        net = tflearn.fully_connected(self.inputs, 50, activation='relu', name="Actor-fc1")
        net = tflearn.fully_connected(net, 100, activation='relu', name="Actor-fc2")
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        self.out = tflearn.fully_connected(
            net, self.action_dim, activation='tanh', weights_init=w_init, name="Actor-output")

        # model = tflearn.regression(self.out, optimizer='sgd', loss='categorical_crossentropy', name="Regression")
        # self.model = tflearn.DNN(model, session=self.session)

        self.network_params = tf.trainable_variables()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        self.critic_input = tf.placeholder(tf.float32, [None, self.action_dim])

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau, name="update-target-with-online") +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau, name="update-target-with-online"))
                for i in range(len(self.target_network_params))]

        for i in range(len(self.network_params)):
            self.network_params[i].assign(self.network_params[i])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, grad_ys=-self.critic_input, name="CombineGradients")

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def train(self, inputs, action_gradient):
        self.session.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.critic_input: action_gradient
        })

    def predict(self, inputs):
        return self.session.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.session.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def update_target_network(self):
        self.session.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    def __init__(self, session, env, action_repeat, actor_num_trainable_variables):
        self.session = session
        self.env = env
        self.action_repeat = action_repeat
        self.state_dim = self.env.observation_space.shape[0]
        # self.action_dim = self.env.action_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.learning_rate = 0.0001
        self.tau = 001

        self.inputs = tflearn.input_data(shape=[None, self.state_dim], name="Critic-InputSpace")
        self.action = tflearn.input_data(shape=[None, self.action_dim], name="Critic-ActionSpace")
        net = tflearn.fully_connected(self.inputs, 4, activation='relu', name="Critic-fc1")

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 50)
        t2 = tflearn.fully_connected(self.action, 50)

        net = tflearn.activation(
            tf.matmul(net, t1.W, name="Critic-Activation-1") + tf.matmul(self.action, t2.W) + t2.b, activation='relu', name="Critic-Activation-2")

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        self.out = tflearn.fully_connected(net, 1, weights_init=w_init, name="Critic-fc2")

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        self.network_params = tf.trainable_variables()[actor_num_trainable_variables:]

        self.target_network_params = tf.trainable_variables()[
                                     (len(self.network_params)+16):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau)
                + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        self.action_grads = tf.gradients(self.out, self.action)

    def train(self, inputs, action, predicted_q_value):
        return self.session.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.session.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.session.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def action_gradients(self, inputs, actions):
        return self.session.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.session.run(self.update_target_network_params)

output_dir='/tmp/cartpole-actor-critic-failure-1'
env = gym.make('CartPole-v0')
if environ.get("UPLOAD", None):
    env = wrappers.Monitor(env, output_dir, force=True)
env.reset()

action_repeat = 2
# MINIBATCH_SIZE = 64
MINIBATCH_SIZE = 64
GAMMA = 0.99
lr = .85
y = .99
# num_episodes = 2000
num_episodes = 1000

with tf.Session() as sess:

    writer = tf.summary.FileWriter('/tmp/tf/logs', sess.graph)

    buf = ReplayBuffer(MINIBATCH_SIZE*num_episodes*2)

    # a_g = tf.Graph()
    # with a_g.as_default() as g:
    #     actor = Actor(sess, env, action_repeat)
    #     writer.add_graph(g)
    #     actor.update_target_network()
    # c_g = tf.Graph()
    # with c_g.as_default() as g:
    #     critic = Critic(sess, env, action_repeat, actor.get_num_trainable_vars())
    #     writer.add_graph(g)
    #     critic.update_target_network()

    actor = Actor(sess, env, action_repeat)
    actor.update_target_network()
    critic = Critic(sess, env, action_repeat, actor.get_num_trainable_vars())
    critic.update_target_network()

    sess.run(tf.global_variables_initializer())

    for i in range(num_episodes):
        print "Episode: {}".format(i)
        #Reset environment and get first new observation
        s = env.reset()
        d = False
        total_reward = 0
        ep_ave_max_q = 0
        epsilon = .4

        # queue = []
        #
        # for _ in range(action_repeat):
        #     queue.append(s)

        for j in range(100):

            # stack = np.stack(queue, axis=0)
            # with a_g.as_default():
            readout_t = actor.predict(np.reshape(s, (1, 4)))

            if random.random() <= epsilon:
                action_index = random.randrange(env.action_space.n)
                epsilon = epsilon * lr
            else:
                action_index = np.argmax(readout_t)

            s2, r, terminal, _ = env.step(action_index)

            # buf.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), r,
            #         terminal, np.reshape(s2, (actor.state_dim,)))

            # queue.append(s2)
            # queue.pop(0)

            buf.add(s, readout_t, r, terminal, s2)

            total_reward += r

            # env.render()

            #  Keep adding experience to the memory until
            # there are at least minibatch size samples
            if buf.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    buf.sample_batch(MINIBATCH_SIZE)

                actor.predict_target(s2_batch)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                # predicted_q_value, _ = critic.train(
                #     s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                predicted_q_value, _ = critic.train(s_batch,
                                                    np.reshape(a_batch, (1, MINIBATCH_SIZE, 2))[0],
                                                    np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(np.reshape(s_batch, (1, MINIBATCH_SIZE, 4))[0])
                grads = critic.action_gradients(s_batch,
                                                a_outs)
                actor.train(s_batch,
                            grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2

            if terminal:
                # writer.add_summary(tf.summary.scalar("actor.out", actor.out))
                # x = tf.summary.scalar("readout.t", readout_t)
                # y = tf.summary.scalar("readout.t", predicted_q_value)
                # tf.summary.merge_all()
                # writer.add_summary(x, i)
                # writer.flush()

                print ("took %d iterations, total reward was %d" % (j, total_reward))
                break

        # merged = tf.summary.merge_all()
        # summary = sess.run(merged)
        # writer.add_summary(summary)
        # writer.flush()


env.close()
if environ.get("UPLOAD", None):
    gym.upload(output_dir, api_key='sk_Fjxb5zJHQkSil4KZgj35A')