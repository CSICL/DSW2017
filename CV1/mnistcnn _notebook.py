
# coding: utf-8

# In[1]:



#import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import math


# # Phase 1: Assemble our graph
# ## Step 1: read the data

# In[ ]:


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

sess = tf.InteractiveSession()


# ## Step 2: create palce holders for inputs and labels
# 

# In[ ]:


# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])


# ## Step3 : Create weights and bias

# In[ ]:


global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
name='global_step')


# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)


# ## Step4 : Build maodel to predict 

# In[ ]:


# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)


# ## Step 5 : Specify loss function

# In[ ]:


# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## Step6:Create Optimizer

# In[ ]:


# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)


# # Phase2 :Train our model

# In[ ]:


saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(20000):
  batch = mnist.train.next_batch(50)
  max_learning_rate = 0.003
  min_learning_rate = 0.0001
  decay_speed = 2000.0
  learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        X:batch[0], Y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={X: batch[0], Y_: batch[1],lr: learning_rate})
  if (i + 1) % 1000==0:
    saver.save(sess, './checkpoints/mnistcnn',
    global_step=global_step)
save_path = saver.save(sess, "modelcnn2.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g"%accuracy.eval(feed_dict={
    X: mnist.test.images, Y_: mnist.test.labels}))


# # See your model in Tensorboard

# ## Step1:
# 
# Writer = tf.summary.Filewriter(./graphs’,sess.graph)
# 
# ## Step2:After running the program run this in terminal
# $tensorboard –logdir=’./graphs’

# # Save our model every 1000 iterations

# In[ ]:


saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        X:batch[0], Y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={X: batch[0], Y_: batch[1]})
  if (i + 1) % 1000==0:
    saver.save(sess, 'mnistcnn',
    global_step=global_step)


# ## Restore our model weights

# In[ ]:


with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model2.ckpt")
        print ("Model restored.")


# ##  Restore the latest checkpoint

# In[ ]:


ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/mnistcnn'))
if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Latest model restored.")


# # Prediction

# In[ ]:


def predictint(imvalue):
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
   

    
    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28,1])
    #X = tf.reshape(X, shape=[-1,28,28,1])
    #X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 10])
    # variable learning rate
    lr = tf.placeholder(tf.float32)

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 4  # first convolutional layer output depth
    L = 8  # second convolutional layer output depth
    M = 12  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.ones([K])/10)
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L])/10)
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M])/10)

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N])/10)
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.ones([10])/10)

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    """
    Load the model.ckpt file
    file is stored in the same directory as this python script is started
    Use the model to predict the integer. Integer is returend as list.

    Based on the documentatoin at
    https://www.tensorflow.org/versions/master/how_tos/variables/index.html
    """
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "modelcnn1.ckpt")
        print ("Model restored.")
   
        prediction=tf.argmax(Y,1)
        imvalue = imvalue[ ... , np.newaxis]
        return prediction.eval(feed_dict={X: [imvalue]}, session=sess)


