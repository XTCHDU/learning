import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt



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

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'save/1000model.ckpt')
read_x_data = sio.loadmat('x_data.mat') #读取mat文件
read_y_data = sio.loadmat('y_data.mat')
x1 = read_x_data['radar_in'].reshape(-1)
x2 = read_y_data['y'].reshape(-1)
ans = []
for SNR in np.arange(-20,40,1):
    total = 0
    for k in range(100):
        print('Now SNR=%d and k=%d'% (SNR,k))
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


        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_test,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy,feed_dict={xs:x_test,ys:y_test})
        total += result

    ans.append([SNR,result/100])
plt.figure(1)
ans = np.array(ans)
plt.plot(ans[:,0],ans[:,1])
plt.show()







