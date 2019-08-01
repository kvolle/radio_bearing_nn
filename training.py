import tensorflow as tf
import numpy as np

class Network:
    def __init__(self, x):
        self.loss = self.loss_fcn()
        self.hidden1 = self.fcl(x, 128, "hidden_1")
        self.out = tf.nn.softmax(self.fcl(self.hidden1, 72, "out"))
        return self.out

    def fcl(self, input_layer, nodes, name, keep_rate=1.):
        # Pass th#rough to conv_layer. renamed function for easier readability
        layer = self.conv_layer(input_layer, [1, 1, input_layer.shape[3], nodes], name, padding='VALID', stride=1,
                                pooling=False)
        #        out = tf.nn.dropout(layer, rate=#drop_rate)
        out = tf.nn.dropout(layer, keep_prob=keep_rate)
        return out

    def conv_layer(self, input_layer, weights, name, padding, stride=1, pooling=True):
        # with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name+"_kernel", shape=weights, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        conv = self.conv2d(input_layer, kernel, padding, stride)
        init = tf.constant(1., shape=[weights[-1]], dtype=tf.float32)
        bias = tf.get_variable(name+"_bias",  dtype=tf.float32, initializer=init)
        preactivation = tf.nn.bias_add(conv, bias, name=name+"_bias_add")
        conv_relu = tf.nn.relu(preactivation, name=name)

        if pooling:
            out = self.create_max_pool_layer(conv_relu)
        else:
            out = conv_relu
        return out


input_layer = tf.placeholder(shape=[None,3], dtype=tf.float32)
truth = tf.placeholder(shape=[None], dtype=tf.float32)
network = Network(input_layer)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    tf_saver = tf.train.Saver(name="saver")
    if tf.train.checkpoint_exists("./model/Final"):
        print("Loading from model")
        tf_saver.restore(sess,'./model/Final')
    else:
        print("Training from scratch")

    writer = tf.summary.FileWriter("log/", sess.graph)

    start = 0
    data_in = np.asarray([0,0,0])
    angle = 7.

    N = 50000
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(network.loss)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    for step in range(start, N):
        _, loss_v = sess.run([train_step, network.loss], feed_dict={input_layer:data_in, truth:angle})
        if step % 100 == 0:
            print(str(step) + ", " +str(loss_v))
            # ll = sess.run(network.acc)
            # writer.add_summary(ll, step)
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            tf_saver.save(sess, 'model/Final')
            quit()
        if step % 20000 == 0:
            tf_saver.save(sess, 'model/intermediate', global_step=step)
    writer.close()
    for i in range(1500):
        _ = sess.run([update_op])
    tf_saver.save(sess, 'model/Final')
    print("Fin")
