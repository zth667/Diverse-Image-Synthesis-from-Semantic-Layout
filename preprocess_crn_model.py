#This is a model trained on GTA5. Assume training images are 00000001.png,...,00012403.png and test images are 001000001,...,00106382.png.
from __future__ import division
import os,cv2,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np

def lrelu(x):
    return tf.maximum(0.2*x,x)

def recursive_generator(label,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=label
    else:
        downsampled=tf.image.resize_area(label,(sp//2,sp),align_corners=False)
        input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp*2),align_corners=True),label],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,27,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
        split0,split1,split2=tf.split(tf.transpose(net,perm=[3,1,2,0]),num_or_size_splits=3,axis=0)
        net=tf.concat([split0,split1,split2],3)
    return net

def recursive_generator_single(label,sp):
    dim=512 if sp>=128 else 1024
    if sp==4:
        input=label
    else:
        downsampled=tf.image.resize_area(label,(sp//2,sp),align_corners=False)
        input=tf.concat([tf.image.resize_bilinear(recursive_generator(downsampled,sp//2),(sp,sp*2),align_corners=True),label],3)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    if sp==256:
        net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
        net=(net+1.0)/2.0*255.0
        #split0,split1,split2=tf.split(tf.transpose(net,perm=[3,1,2,0]),num_or_size_splits=3,axis=0)
        #net=tf.concat([split0,split1,split2],3)
    return net

sess=tf.Session()
sp=256 #input resolution is 256x512
num_noise = 10
is_training=False
pre_root = "~/PhotographicImageSynthesis/result_GTA"
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,None,None,20])
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator(label,sp)
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(pre_root)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
graph = tf.get_default_graph()
last_w = graph.get_tensor_by_name("g_256_conv100/weights:0")
last_b = graph.get_tensor_by_name("g_256_conv100/biases:0")
lw = sess.run(last_w)[:,:,:,(0,9,18)]
lb = sess.run(last_b)[(0,9,18),]
print lw.shape,lb.shape

tf.reset_default_graph()
sess=tf.Session()
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,None,None,20])
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator_single(label,sp)
    t_vars=tf.trainable_variables()
saver=tf.train.Saver(max_to_keep=1000,var_list=[var for var in t_vars if not var.name.startswith('g_256_conv100')])
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(pre_root)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
graph = tf.get_default_graph()
last_w = graph.get_tensor_by_name("g_256_conv100/weights:0")
last_b = graph.get_tensor_by_name("g_256_conv100/biases:0")
assign_op = tf.assign(last_w,lw)
sess.run(assign_op)
assign_op = tf.assign(last_b,lb)
sess.run(assign_op)
filez = {}
j=2
for i in range(7):
    j*=2
    myweight = graph.get_tensor_by_name('g_%d_conv1/weights:0'%j)
    filez['g_%d_conv1/weights:0'%j] = sess.run(myweight)
if not os.path.isdir("gta_pretrained"):
    os.makedirs("gta_pretrained")
saver=tf.train.Saver(max_to_keep=1000)
saver.save(sess,"gta_pretrained/model.ckpt")

tf.reset_default_graph()
sess=tf.Session()
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,None,None,20+num_noise])
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator_single(label,sp)
    t_vars=tf.trainable_variables()
saver=tf.train.Saver(max_to_keep=1000,var_list=[var for var in t_vars if not (var.name.startswith('g_') and var.name.endswith('_conv1/weights:0'))])
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("gta_pretrained")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
graph = tf.get_default_graph()    
j=2
for i in range(7):
    j*=2
    myweight = graph.get_tensor_by_name('g_%d_conv1/weights:0'%j)
    myw = filez['g_%d_conv1/weights:0'%j]
    stv = np.sqrt(np.sum(myw**2)/(myw.shape[0]*myw.shape[1]*myw.shape[2]*myw.shape[3]))
    extraw = np.random.randn(myw.shape[0],myw.shape[1],num_noise,myw.shape[3])*stv*5
    fullw = np.concatenate((myw,extraw),axis=2)
    print(fullw.shape,np.mean(myw))
    assign_op = tf.assign(myweight,fullw)
    sess.run(assign_op)
saver=tf.train.Saver(max_to_keep=1000)
saver.save(sess,"gta_pretrained/model.ckpt")