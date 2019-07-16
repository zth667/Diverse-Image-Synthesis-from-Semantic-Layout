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

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

def build_vgg19(input,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net={}
    vgg_rawnet=scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
    vgg_layers=vgg_rawnet['layers'][0]
    net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    return net

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

def noise_encoder(label):
    net=slim.conv2d(label,100,[3,3],rate=1,normalizer_fn=slim.layer_norm,weights_initializer=
tf.initializers.truncated_normal(stddev=0.15),activation_fn=lrelu,scope='g_noise_conv1')
    net=slim.conv2d(net,100,[3,3],rate=1,normalizer_fn=slim.layer_norm,weights_initializer=
tf.initializers.truncated_normal(stddev=0.15),activation_fn=lrelu,scope='g_noise_conv2')
    net=slim.conv2d(net,10,[1,1],weights_initializer=
tf.initializers.truncated_normal(stddev=0.15),rate=1,activation_fn=None,scope='g_noise_conv3')
    return net

def compute_error(real,fake,label):
    return tf.reduce_mean(tf.multiply(tf.abs(fake-real),label))
sess=tf.Session()
sp=256 #input resolution is 256x512
is_training=True
num_noise=10
with tf.variable_scope(tf.get_variable_scope()):
    label=tf.placeholder(tf.float32,[None,None,None,20+num_noise])
    purelabel = tf.slice(label,[0,0,0,0],[1,256,512,20])
    noise_encode = noise_encoder(label)
    moments = tf.nn.moments(noise_encode,[0,1,2,3])
    #syn_label = tf.concat([purelabel,noise_encode],3)
    real_image=tf.placeholder(tf.float32,[None,None,None,3])
    fake_image=tf.placeholder(tf.float32,[None,None,None,3])
    generator=recursive_generator(purelabel,noise_encode,sp)
    weight=tf.placeholder(tf.float32)
    vgg_real=build_vgg19(real_image)
    vgg_fake=build_vgg19(generator,reuse=True)
    mask = tf.placeholder(tf.float32,[None,None,None,1])
    p0=compute_error(vgg_real['input'],vgg_fake['input'],mask)
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],mask)/1.4
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(mask,(sp//2,sp)))/1.8
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(mask,(sp//4,sp//2)))/1.3
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(mask,(sp//8,sp//4)))/2.2
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(mask,(sp//16,sp//8)))*10/0.62
    G_loss=p0+p1+p2+p3+p4+p5
    t_vars=tf.trainable_variables()
    lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("gta_demo/final")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if not os.path.isdir("gta_demo/result"):
    os.makedirs("gta_demo/result")
lst = os.listdir("./datasets/GTA/Label256Full/")
lst = [x for x in lst if x.endswith(".png")]
for ind in lst:
    #if not os.path.isfile("./datasets/GTA/Label256Full/%08d.png"%ind):#test label
    #    continue    
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
