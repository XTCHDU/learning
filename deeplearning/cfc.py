import tensorflow as tf
import numpy as np
import scipy.io as sio
SNR = 1000
read_x_data = sio.loadmat('x_data.mat') #读取mat文件
read_y_data = sio.loadmat('y_data.mat')
x1 = read_x_data['radar_in'].reshape(-1)
x2 = read_y_data['y'].reshape(-1)
Pnoise = pow(10,-SNR/10.)
Noise = Pnoise * np.random.standard_normal(200)
in_samples = 1000
x_data = np.zeros((in_samples,200))
y_data = np.zeros((in_samples,2))
for i in range(int(in_samples/2)):
    temp_x1 = x1*np.random.uniform(1,3)
    temp_x2 = x2*np.random.uniform(1,3)
    start = np.random.randint(0,801)
    end = start+200
    x_data[i*2] = temp_x1[start:end]+Noise
    start = np.random.randint(0,801)
    end = start+200
    x_data[i*2+1] =temp_x2[start:end]+Noise
    y_data[i*2] = [0,1.]
    y_data[i*2+1] = [1.,0]
test_samples = 500
y_test = np.zeros((test_samples,2))
x_test = np.zeros((test_samples,200))
for i in range(int(test_samples/2)):
    temp_x1 = x1*np.random.uniform(1,2)
    temp_x2 = x2*np.random.uniform(1,2)
    start = np.random.randint(0, 801)
    end = start + 200
    x_test[i * 2] = temp_x1[start:end] + Noise
    start = np.random.randint(0, 801)
    end = start + 200
    x_test[i * 2 + 1] = temp_x2[start:end] + Noise
    y_test[i*2] = [0,1.]
    y_test[i*2+1] = [1.,0]



def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def biases_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv1d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_1x2(x):
    return tf.nn.max_pool(x,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')

xs = tf.placeholder(tf.float32,[None,200])
ys = tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,1,200,1])

W_conv1 = weight_variable([1,2,1,32])
b_conv1 = biases_variable([32])
h_conv1 = tf.nn.relu(conv1d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_1x2(h_conv1)

W_conv2 = weight_variable([1,2,32,64])
b_conv2 = biases_variable([64])
h_conv2 = tf.nn.relu(conv1d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_1x2(h_conv2)

W_fc1 = weight_variable([50*64,100])
b_fc1 = biases_variable([100])
h_pool2_flat = tf.reshape(h_pool2,[-1,50*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

W_fc2 = weight_variable([100,2])
b_fc2 = biases_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data, keep_prob:1.0})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_test, ys: y_test, keep_prob:1.0}))
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_test, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: x_test, ys: y_test, keep_prob:1.0})
        print(result)


