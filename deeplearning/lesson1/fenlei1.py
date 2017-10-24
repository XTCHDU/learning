import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases,name='Wxplus_b')
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

xs = tf.placeholder(tf.float32,[None,784]) #28x28
ys = tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})

    return result

for i in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_x,ys:batch_y})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
saver = tf.train.Saver()
saver.save(sess,'my_net/fenlei1.ckpt')
