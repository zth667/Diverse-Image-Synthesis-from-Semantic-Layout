#This is a model trained on GTA5. Assume training images are 00000001.png,...,00012403.png and test images are 001000001,...,00106382.png.
from __future__ import division
import os,cv2,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def lrelu(x):
    return tf.maximum(0.2*x,x)

#Define the main network
def recursive_generator(label,noise,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=tf.concat([label,noise],3)
    else:
        downsampled_label=tf.image.resize_area(label,(sp//2,sp),align_corners=False)
        downsampled_noise=tf.image.resize_bilinear(noise,(sp//2,sp),align_corners=False)
        input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled_label,downsampled_noise,sp//2),(sp,sp*2),align_corners=True),label,noise],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
    return net

#Define the Noise Encoder
def noise_encoder(label):
    net=slim.conv2d(label,100,[3,3],rate=1,normalizer_fn=slim.layer_norm,weights_initializer=
tf.initializers.truncated_normal(stddev=0.15),activation_fn=lrelu,scope='g_noise_conv1')
    net=slim.conv2d(net,100,[3,3],rate=1,normalizer_fn=slim.layer_norm,weights_initializer=
tf.initializers.truncated_normal(stddev=0.15),activation_fn=lrelu,scope='g_noise_conv2')
    net=slim.conv2d(net,10,[1,1],weights_initializer=
tf.initializers.truncated_normal(stddev=0.15),rate=1,activation_fn=None,scope='g_noise_conv3')
    return net

#Define the loss function
def compute_error(real,fake,label):
    return tf.reduce_mean(tf.multiply(tf.abs(fake-real),label))

sess=tf.Session()

#input resolution is 256x512
sp=256 

#Number of noise vector channels
num_noise=10

#Define the network
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,None,None,20+num_noise])
    purelabel = tf.slice(label,[0,0,0,0],[1,256,512,20])
    noise_encode = noise_encoder(label)
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator(purelabel,noise_encode,sp)

#Load Pre-trained model
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("gta_demo/final")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

#Find test semantic layouts
if not os.path.isdir("gta_demo/result"):
    os.makedirs("gta_demo/result")
lst = os.listdir("./datasets/GTA/Label256Full/")
lst = [x for x in lst if x.endswith(".png")]

#Testing
for ind in lst:
    semantic=helper.get_semantic_map("./datasets/GTA/Label256Full/%s"%ind)
    semantic=np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)
    output = np.empty([9,256,512,3],dtype=np.float32)
    for i in range(9):
        semantic1=np.concatenate((semantic,np.random.randn(semantic.shape[0],semantic.shape[1],semantic.shape[2],num_noise)),axis=3)
        output[i]=sess.run(generator,feed_dict={label:semantic1})
    output=np.minimum(np.maximum(output,0.0),255.0)
    upper=np.concatenate((output[0,:,:,:],output[1,:,:,:],output[2,:,:,:]),axis=1)
    middle=np.concatenate((output[3,:,:,:],output[4,:,:,:],output[5,:,:,:]),axis=1)
    bottom=np.concatenate((output[6,:,:,:],output[7,:,:,:],output[8,:,:,:]),axis=1)
    scipy.misc.toimage(np.concatenate((upper,middle,bottom),axis=0),cmin=0,cmax=255).save("gta_demo/result/%s"%ind)
