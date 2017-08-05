
# coding: utf-8

# In[1]:



import tensorflow as tf
#import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)


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

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
O = 30
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


# ## Step4 : Build maodel to predict 

# In[ ]:


# The model
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
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


learning_rate = 0.003
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


# # Phase2 :Train our model

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
    saver.save(sess, './checkpoints/five_layers_sigmiod',
    global_step=global_step)
save_path = saver.save(sess, "model20.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g"%accuracy.eval(feed_dict={
    X: mnist.test.images, Y_: mnist.test.labels}))



# # See your model in Tensorboard

# ## Step1:
# 
# Writer = tf.summary.Filewriter('./graphs’,sess.graph)
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
    saver.save(sess, 'five_layers_sigmiod',
    global_step=global_step)


# ## Restore our model weights

# In[ ]:


with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model2.ckpt")
        print ("Model restored.")


# ##  Restore the latest checkpoint

# In[ ]:


ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/five_layers_sigmiod'))
if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Latest model restored.")


# # Prediction

# In[ ]:



"""Predict a handwritten integer (MNIST beginners).

Script requires
1) saved model (model.ckpt file) in the same location as the script is run from.
(requried a model created in the MNIST beginners tutorial)
2) one argument (png file location of a handwritten integer)

Documentation at:
http://niektemme.com/ @@to do
"""

#import modules
import sys
import tensorflow as tf
import PIL
from PIL import Image,ImageFilter
import PIL.ImageOps    
import cv2
import numpy as np
import math
from scipy import ndimage

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """
    
    # Define the model (same as when creating the model file)
   

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 784])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 10])

    # five layers and their number of neurons (tha last layer has 10 softmax neurons)
    L = 200 
    M = 100
    N = 60
    O = 30
    # Weights initialised with small random values between -0.2 and +0.2
    # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
    W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
    B1 = tf.Variable(tf.zeros([L]))
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    B2 = tf.Variable(tf.zeros([M]))
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    B3 = tf.Variable(tf.zeros([N]))
    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
    B4 = tf.Variable(tf.zeros([O])) 
    W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
    B5 = tf.Variable(tf.zeros([10]))

    # The model
    #XX = tf.reshape(X, [-1, 784])
    Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
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
        saver.restore(sess, "model2.ckpt")
        print ("Model restored.")
   
        prediction=tf.argmax(Y,1)
        return prediction.eval(feed_dict={X: [imvalue]}, session=sess)



