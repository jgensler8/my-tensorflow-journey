import tensorflow as tf
import numpy as np

with tf.Session() as sess:

    writer = tf.summary.FileWriter('/tmp/tf/logs')

    # args = tf.placeholder(tf.int32, [4])
    train = tf.constant([4,2,3,4], shape=[4, 1])
    weights = tf.constant([2,2,2,2], shape=[1, 4])
    tf.global_variables_initializer().run()

    mul = tf.matmul(train, weights)
    op = tf.gradients(mul, weights)

    # max = tf.reduce_max(args, op)

    # with tf.name_scope('summaries'):
    #     # tf.summary.tensor_summary("train", train)
    #     for i in range(4):
    #         tf.summary.scalar("arg {}".format(i), op[0][i])

    # tf.summary.scalar("max", tf.reduce_max(op[0]))
    # tf.summary.histogram("all", op)

    res = sess.run(op, feed_dict={train: np.reshape([1,2,4,4], (4,1))})
    print res

    # merged = tf.summary.merge_all()
    # print merged
    # summ = sess.run(merged)
    # summary = sess.run(merged)
    # writer.add_summary(summ)

    writer.flush()
