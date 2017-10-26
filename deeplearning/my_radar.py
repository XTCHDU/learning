import tensorflow as tf
import numpy as np
import scipy.io as sio
SNR = 10
read_x_data = sio.loadmat('x_data.mat')  # 读取mat文件
read_y_data = sio.loadmat('y_data.mat')
# print(data.keys())   # 查看mat文件中的所有变量
x1 = read_x_data['radar_in']
x2 = read_y_data['y']
x_data_true = np.append(x1,x1).reshape(1,-1)
x_data_false = np.append(x1,x2).reshape(1,-1)

Pnoise = pow(10,-SNR/10.)
Noise = Pnoise * np.random.standard_normal(x_data_true.shape)
x_data = np.zeros((100,2002))
y_data = np.zeros((100,2))
for i in range(50):
    temp_x1 = x1*np.random.uniform(1,2)
    temp_x2 = x2*np.random.uniform(1,2)
    x_data[i*2] = np.append(x1,temp_x1)+Noise
    x_data[i*2+1] = np.append(x1,temp_x2)+Noise
    y_data[i*2] = [0,1.]
    y_data[i*2+1] = [1.,0]



xs = tf.placeholder(tf.float32,[None,2002])
ys = tf.placeholder(tf.float32,[None,2]) #输出为真假的概率

x_test = x_data
y_test = y_data

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

pre_1 = add_layer(xs,2002,8000,activation_function=tf.nn.relu)
pre_2 = add_layer(pre_1,8000,4000,activation_function=tf.nn.sigmoid)
pre_3 = add_layer(pre_2,4000,500,activation_function=tf.nn.relu)
prediction = add_layer(pre_3,500,2,activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    print(i)
    if i % 50 == 0:
        sess.run(loss,feed_dict={xs:x_test,ys:y_test})
        print(sess.run(prediction,feed_dict={xs:x_test,ys:y_test}))
        print(sess.run(loss,feed_dict={xs:x_test,ys:y_test}))
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_data,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy,feed_dict={xs:x_test,ys:y_test}))




