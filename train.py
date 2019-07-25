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

#Define the VGG19 Net
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

#Define the main network recursively
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

#Define the Loss Function
def compute_error(real,fake,label):
    return tf.reduce_mean(tf.multiply(tf.abs(fake-real),label))


if __name__ == '__main__':
    sess=tf.Session()

    #Define the hyper-parameters
    
    #input resolution is 256x512
    sp=256 
    #Number of noise vector channels
    num_noise=10
    #Training data paths
    labelroot = "./datasets/GTA/Label256Full/"
    imageroot = "./datasets/GTA/RGB256Full/"
    #Number of images for each epoch
    K=400
    #Number of inner iterations (Training steps per epoch)
    L=10000
    #Batchsize
    batchsize=1
    #Initial Learning Rate
    curr_lr = 1e-4
    #Number of candidates for Nearest Neighbor search
    nn_num = 10
    # the no. of epoch when we decay the lr and reinitialize the noise encoder
    reinit_point=5

    #test both training and testing images
    testlist = [1,2,6,100001,100002,100003,100010]

    #Buffer for training data 
    input_images=np.empty([K,256,512,3],dtype=np.float32)
    label_images=np.empty([K,1,256,512,20],dtype=np.float32)
    loss_masks = np.empty([K,1,256,512,1],dtype=np.float32)
    minlabels = np.empty([K,256,512,20+num_noise],dtype=np.float32)

    #Define the Network and Loss function
    with tf.variable_scope(tf.get_variable_scope()):
        mask = tf.placeholder(tf.float32,[None,None,None,1])
        label=tf.placeholder(tf.float32,[None,None,None,20+num_noise])
        purelabel = tf.slice(label,[0,0,0,0],[1,256,512,20])
        noise_encode = noise_encoder(label)
        real_image=tf.placeholder(tf.float32,[None,None,None,3])
        fake_image=tf.placeholder(tf.float32,[None,None,None,3])
        generator=recursive_generator(purelabel,noise_encode,sp)

        #Generate VGG-19 feature
        vgg_real=build_vgg19(real_image)
        vgg_fake=build_vgg19(generator,reuse=True)

        #Define Loss Function
        p0=compute_error(vgg_real['input'],vgg_fake['input'],mask)
        p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'],mask)/1.4
        p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'],tf.image.resize_area(mask,(sp//2,sp)))/1.8
        p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'],tf.image.resize_area(mask,(sp//4,sp//2)))/1.3
        p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'],tf.image.resize_area(mask,(sp//8,sp//4)))/2.2
        p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'],tf.image.resize_area(mask,(sp//16,sp//8)))*10/0.62
        G_loss=p0+p1+p2+p3+p4+p5

        t_vars=tf.trainable_variables()
        #Learning rate
        lr=tf.placeholder(tf.float32)

    #Define the optimizer    
    G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])
    '''
    Please remove the 'var_list' argument in the next line if you have noise encoder parameters available in the loaded model (which is not the case for the model generated using 'preprocess_crn_model.py') 
    '''
    saver=tf.train.Saver(max_to_keep=1000,var_list=[var for var in t_vars if not var.name.startswith('g_noise')])
    sess.run(tf.global_variables_initializer())
    #Load the Pre-trained Model
    ckpt=tf.train.get_checkpoint_state("gta_pretrained/")
    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    
    #Load Rarity mask for loss rebalancing
    rarity_mask=np.load("rarity/GTA_weightedrarity_mask.npy",mmap_mode='r')

    #Load Rarity bin for dataset rebalancing
    objectlist = ['road','building','sky','other','sidewalk']
    objectid = [0,2,10,19,1]
    objectnum = 5
    rarity_bin=[]
    for i in range(objectnum):
        rarity_bin.append(np.load("rarity/kdecolor_rarity_bin_%d.npy"%objectid[i]))

    #Start training
    for ind in range(reinit_point+11):
        st=time.time()
        stind = ind*K%12403
        edind = stind+K
        for i in range(stind,edind):
            idx = np.searchsorted(rarity_bin[(i-stind)%objectnum],np.random.rand())+1
            semantic=helper.get_semantic_map(labelroot+"/%08d.png"%idx)#test label
            semantic=np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)
            label_images[i-stind]=semantic
            input_images[i-stind]=np.float32(scipy.misc.imread(imageroot+"/%08d.png"%idx))#training image
            loss_masks[i-stind]=rarity_mask[idx-1]
            mindist = np.inf
            for j in range(nn_num):
                semantic = label_images[i-stind]
                semantic=np.concatenate((semantic,np.random.randn(semantic.shape[0],semantic.shape[1],semantic.shape[2],num_noise)),axis=3)
                G_current = sess.run(G_loss,feed_dict={label:semantic,real_image:[input_images[i-stind]],mask:loss_masks[i-stind]})
                if G_current<mindist:
                    mindist=G_current
                    minlabels[i-stind]=semantic[0]
        
        for l in range(L):
            f=np.random.randint(K, size=batchsize)
            label_batch = minlabels[f]
            input_batch = input_images[f]
            mask_batch = loss_masks[f][0]
            _,G_current,l0,l1,l2,l3,l4,l5=sess.run([G_opt,G_loss,p0,p1,p2,p3,p4,p5],feed_dict={label:label_batch,real_image:input_batch,lr:curr_lr,mask:mask_batch})
            
            #Intermediate test
            if(l%1000==999):
                print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"%(ind,l,G_current,np.mean(l0),np.mean(l1),np.mean(l2),np.mean(l3),np.mean(l4),np.mean(l5)))
                os.makedirs("gta_demo/%04d_%06d"%(ind,l))
                for idx in testlist:
                    if not os.path.isfile(labelroot+"/%08d.png"%idx):#test label
                        continue
                    semantic=helper.get_semantic_map(labelroot+"/%08d.png"%idx)#test label
                    semantic=np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)
                    output = np.empty([9,256,512,3],dtype=np.float32)
                    for i in range(9):
                        semantic1=np.concatenate((semantic,np.random.randn(semantic.shape[0],semantic.shape[1],semantic.shape[2],num_noise)),axis=3)
                        output[i]=sess.run(generator,feed_dict={label:semantic1})
                    output=np.minimum(np.maximum(output,0.0),255.0)
                    upper=np.concatenate((output[0,:,:,:],output[1,:,:,:],output[2,:,:,:]),axis=1)
                    middle=np.concatenate((output[3,:,:,:],output[4,:,:,:],output[5,:,:,:]),axis=1)
                    bottom=np.concatenate((output[6,:,:,:],output[7,:,:,:],output[8,:,:,:]),axis=1)
                    scipy.misc.toimage(np.concatenate((upper,middle,bottom),axis=0),cmin=0,cmax=255).save("gta_demo/%04d_%06d/%06d_output.png"%(ind,l,idx))
        
        #Save model
        os.makedirs("gta_demo/%04d"%ind)
        saver.save(sess,"gta_demo/%04d/model.ckpt"%ind)

        #reinitialize the noise encoder to avoid overfitting    
        if ind==reinit_point:
            saver=tf.train.Saver(max_to_keep=1000,var_list=[var for var in tf.trainable_variables() if not var.name.startswith('g_noise')])
            sess.run(tf.global_variables_initializer())
            ckpt=tf.train.get_checkpoint_state("gta_demo/%04d"%ind)
            saver.restore(sess,ckpt.model_checkpoint_path)
            saver=tf.train.Saver(max_to_keep=1000)
            curr_lr /= 10.