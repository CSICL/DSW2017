# Copyright 2016 Niek Temme. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('1')
    width = float(im.size[0])
    height = float(im.size[1])
    #newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    #wpercent = (basewidth / float(im.size[0]))
    #hsize = int((float(im.size[1]) * float(wpercent)))
    img = im.resize((28, 28), PIL.Image.ANTIALIAS) #.filter(ImageFilter.SHARPEN)
    #img1 = PIL.ImageOps.invert(img)
    img.save('resized_image.jpg')
    #print (type(img))
    '''
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    '''
    #newImage.save("sample.png")

    #tv = list(newImage.getdata()) #get pixel values
    tv = list(img.getdata())
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    #tva = [ (255-x)*1.0/255.0 for x in tv] 
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    gray = cv2.imread(argv, 0)

    # rescale it
    gray = cv2.resize(255-gray, (28, 28))
    # better black and white version
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    cv2.imwrite("sam3.jpg", gray)
    '''
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    '''
    # save the processed images
    #cv2.imwrite("sample3.jpg", gray)
    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = gray.flatten() / 255.0
    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """
    #mages = flatten
    #correct_val = np.zeros((10))
    #correct_val[no] = 1
    #correct_vals[i] = correct_val
    #i += 1

    """
    the prediction will be an array with four values,
    which show the predicted number
    """
    #prediction = tf.argmax(y,1)
    return flatten
    #print(tva)

def main(argv):
    """
    Main function.
    """
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    print (predint[0]) #first value in list
    
if __name__ == "__main__":
    main(sys.argv[1])
