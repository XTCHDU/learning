import tensorflow as tf
import numpy as np

xs = tf.placeholder(tf.float32,[None,100])
ys = tf.placeholder(tf.float32,[None,2]) #输出为真假的概率
x_data = np.array([np.random.normal(1,0.1,[100]) for i in range(1000)])
y_data = np.zeros((1000,2))
for i in range(1000):
    k = np.random.randint(0,2)
    y_data[i,0]=(k+1)%2
    y_data[i,1]=k%2
x_test = np.array([np.random.normal(1,0.1,[100]) for i in range(1000)])
y_test = np.zeros((1000,2))
for i in range(1000):
    k = np.random.randint(0,2)
    y_test[i,0]=(k+1)%2
    y_test[i,1]=k%2
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

pre_1 = add_layer(xs,100,500,activation_function=tf.nn.relu)
prediction = add_layer(pre_1,500,2,activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)))
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        sess.run(loss,feed_dict={xs:x_test,ys:y_test})
        print(sess.run(prediction,feed_dict={xs:x_test,ys:y_test}))
        print(sess.run(loss,feed_dict={xs:x_test,ys:y_test}))
        correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_data,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy,feed_dict={xs:x_test,ys:y_test}))




