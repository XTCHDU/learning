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



xs = tf.placeholder(tf.float32,[None,200])
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

pre_1 = add_layer(xs,200,200,activation_function=tf.nn.relu)
pre_2 = add_layer(pre_1,200,400,activation_function=tf.nn.sigmoid)
pre_3 = add_layer(pre_2,400,50,activation_function=tf.nn.relu)
prediction = add_layer(pre_3,50,2,activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    saver.save(sess,'save/%dmodel.ckpt'% SNR)
    if i %50 == 0:
        sess.run(loss,feed_dict={xs:x_test,ys:y_test})
        sess.run(prediction,feed_dict={xs:x_test,ys:y_test})
        print(sess.run(loss,feed_dict={xs:x_test,ys:y_test}))
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_test,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy,feed_dict={xs:x_test,ys:y_test})
        print(result)






