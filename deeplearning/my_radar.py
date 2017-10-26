import tensorflow as tf
import numpy as np
import scipy.io as sio
SNR = -20
read_x_data = sio.loadmat('x_data.mat')  # 读取mat文件
read_y_data = sio.loadmat('y_data.mat')
# print(data.keys())   # 查看mat文件中的所有变量
x1 = read_x_data['radar_in'].reshape(-1)
x2 = read_y_data['y'].reshape(-1)
Pnoise = pow(10,-SNR/10.)
Noise = Pnoise * np.random.standard_normal(x1.shape)
in_samples = 1000
x_data = np.zeros((in_samples,1001))
y_data = np.zeros((in_samples,2))
for i in range(int(in_samples/2)):
    temp_x1 = x1*np.random.uniform(1,3)
    temp_x2 = x2*np.random.uniform(1,3)
    x_data[i*2] = temp_x1+Noise
    x_data[i*2+1] =temp_x2+Noise
    y_data[i*2] = [0,1.]
    y_data[i*2+1] = [1.,0]
test_samples = 100
y_test = np.zeros((test_samples,2))
x_test = np.zeros((test_samples,1001))
for i in range(int(test_samples/2)):
    temp_x1 = x1*np.random.uniform(1,2)
    temp_x2 = x2*np.random.uniform(1,2)
    x_test[i*2] = temp_x1+Noise
    x_test[i*2+1] = temp_x2+Noise
    y_test[i*2] = [0,1.]
    y_test[i*2+1] = [1.,0]



xs = tf.placeholder(tf.float32,[None,1001])
ys = tf.placeholder(tf.float32,[None,2]) #输出为真假的概率


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

pre_1 = add_layer(xs,1001,20,activation_function=tf.nn.relu)
pre_2 = add_layer(pre_1,20,40,activation_function=tf.nn.sigmoid)
pre_3 = add_layer(pre_2,40,50,activation_function=tf.nn.relu)
prediction = add_layer(pre_3,50,2,activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i %50 == 0:
        sess.run(loss,feed_dict={xs:x_test,ys:y_test})
        sess.run(prediction,feed_dict={xs:x_test,ys:y_test})
        print(sess.run(loss,feed_dict={xs:x_test,ys:y_test}))
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_test,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy,feed_dict={xs:x_test,ys:y_test})
        print(result)






