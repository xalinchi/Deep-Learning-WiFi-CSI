import pandas as pd
import scipy.io as sio
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data') # 第一次启动会下载文本资料，放在/tmp/data文件夹下

# print(FLAGS.data_dir)
# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

filename='data.xlsx'
dx=pd.read_excel(filename)

#输入格式为：行为数据流个数，列为数据流长度
x_=np.array(dx)
x_row=np.shape(x_)[0]+1
x_col=np.shape(x_)[1]


def judge(x):
#计算处所给数字的最大开根整数
	for i in range(x):
		if (i*i>x):
			num=i-1
			break
	return num

x_1=judge(x_col)
x_2=x_1*x_1
x_test=np.resize(x_,(x_row,x_2))

def classification(size,n):
#构造编码矩阵
	matrix=np.zeros((size,n))
	k=int(x_row/n)
	for i in range(size):
		t=i//k
		matrix[k*t:k*(t+1),t]=1
	return matrix 
	
n_class=10
y_test=classification(x_row,n_class)
x_num=x_row




	
# x,x_test,y,y_test= train_test_split(x_,y_,test_size=0.01,)
# print(x_test)
# x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.01)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
						  
# def batch_iter(data, batch_size, num_epochs, shuffle=True):
	# data = np.array(data)
	# data_size = len(data)
	# num_batches_per_epoch = int(data_size / batch_size) + 1

	# for epoch in range(num_epochs):
		# if shuffle:
			# shuffle_indices = np.random.permutation(data)
			# shuffled_data = data[shuffle_indices]
		# else:
			# shuffled_data = data

		# for batch_num in range(num_batches_per_epoch):
			# start_index = batch_num * batch_size
			# end_index = min((batch_num + 1) * batch_size, data_size)
			# yield shuffled_data[start_index:end_index]
						  

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, x_2])
x_image = tf.reshape(x, [-1,x_1,x_1,1]) #将输入按照 conv2d中input的格式来reshape，reshape

W_conv1 = weight_variable([2, 4, 1, 2])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([2])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([2, 4, 2, 4])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv2 = bias_variable([4])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([2, 4, 4, 8])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv3 = bias_variable([8])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([2, 4, 8, 16])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv4 = bias_variable([16])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)


"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*32
# 也就是单个通道输出为28*28，共有32个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32
"""
W_conv5 = weight_variable([2, 4, 16, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv5 = bias_variable([32])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)

"""
# 第二层
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 ?*14*14*32， 卷积后为?*14*14*64
# 池化后，输出的图像尺寸为?*7*7*64
"""
W_conv6 = weight_variable([2, 4, 32, 64])
b_conv6 = bias_variable([64])
h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)

#接收上一层输出维数
h_1=h_pool6.get_shape()[1]
h_2=h_pool6.get_shape()[3]
print(h_1,h_2,h_pool6)

W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool6, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) 
keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



# 第四层，输入1024维，输出10维，也就是具体的0~9分类
W_fc2 = weight_variable([1024, n_class])
b_fc2 = bias_variable([n_class])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
y_ = tf.placeholder(tf.float32, [None, n_class])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1])) # 损失函数，交叉熵
train_step = tf.train.AdagradOptimizer(1e-2).minimize(cross_entropy) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer()) # 变量初始化
# # for i in range(20000):
    # # x,x_test= train_test_split(x_,test_size=0.01,)
    # # if i%100 == 0:
        # # # print(batch[1].shape)
        # # train_accuracy = accuracy.eval(feed_dict={
            # # x:batch[0], y_: batch[1], keep_prob: 1.0})
        # # print("step %d, training accuracy %g"%(i, train_accuracy))
    # # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# x,x_test= train_test_split(x_,test_size=0.01,)


n_epochs=5000
batch_size=20
for i in range(n_epochs):
	t=list(range(x_num-batch_size))
	random.shuffle(t)
	x_batch=x_test[t[0]:t[0]+batch_size,:]
	y_batch=y_test[t[0]:t[0]+batch_size,:]
	if (i%100==0):
		train_accuracy=accuracy.eval(feed_dict={x:x_batch,y:y_batch, keep_prob: 1.0})
		print("epoch %d, training accuracy %g"%(i,train_accuracy))
	train_step.run(feed_dict={x:x_batch,y:y_batch,keep_prob:1.0})
print("test_accuracy %g"%accuracy.eval(feed_dict={x:x_test[0:30,:],y:y_test[0:30,:],keep_prob: 1.0}))




# for epoch_i in range(n_epochs):
		# # i=0
	# t=list(range(x_num-batch_size))
	# random.shuffle(t)
	# for batch_i in t:
		 # train_accuracy = accuracy.eval(feed_dict={x:x_test[batch_i:batch_i+batch_size,:], y_: y_test[batch_i:batch_i+batch_size,:], keep_prob: 0.5})
		 # train_accuracy=sess.run([optimizer,accuracy],feed_dict={x:input[batch_i:batch_i+batch_size,:], y: y_test[batch_i:batch_i+batch_size,:]})[1]
		# # accuracy_all=np.append(accuracy_all,train_accuracy)
	# accuracy_all=np.append(accuracy_all,train_accuracy)
# print('the apoch',epoch_i,'accuracy is',np.max(accuracy_all))
	# plt.plot(epoch_i,np.max(accuracy_all),'r*')
# plt.title('CNN Training Result')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()



