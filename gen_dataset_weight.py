import numpy as np 
from scipy.misc import imread,imsave
import helper
import os
from scipy import stats
import sys
from shutil import copyfile
def one_hot(label):
    #print(label.shape)
    output=np.zeros((1,label.shape[0],label.shape[1],19),dtype=np.float32);
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j]<255:
                output[0,i,j,label[i,j]]=1
    return output
def kdeforvoid(a):
    return np.ones(a.shape[1])
labelroot = "./datasets/GTA/Label256Full/"
imageroot = "./datasets/GTA/RGB256Full/"
num_img = 12403
avgcolor = np.empty([num_img,20,3])
nums = np.zeros(20,dtype=np.int)
areas = np.empty([num_img,20])
for i in range(num_img):
    semantic=helper.get_semantic_map(labelroot+"/%08d.png"%(i+1))#test label
    #semantic = one_hot(imread(labelroot+"/%08d.png"%(i+1)))
    semantic=np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)
    image = imread(os.path.join(imageroot,"%08d.png"%(i+1)))
    areas[i] = np.sum(semantic,axis=(0,1,2))
    avgcolor[i]=np.sum(np.multiply(np.transpose(semantic,(3,1,2,0)),image),axis=(1,2))/np.expand_dims(areas[i],1)

kernels = []
invalidid=[]
#image 9061 is blank and hence we exclude it
avgcolor[9061] =  np.nan
for i in range(20):
    base = avgcolor[:,i,:][~np.any(np.isnan(avgcolor[:,i,:]),axis=1)]
    if base.shape[0]<=67:
        print "skip",i
        kernels.append(None)
        invalidid.append(i)
        continue
    values = np.transpose(base)
    kernels.append(stats.gaussian_kde(values))
    print i,base.shape#,kernels[i](values).max()
rarity = np.zeros([num_img,20],dtype=np.float64)
clusterres = np.zeros((num_img,20),dtype=np.int)
rarity_mask = np.empty([num_img,256,512,1],dtype=np.float32)
objectlist = ['road','building','vegetation','other','car','sidewalk']
objectid = range(20)#+[100]
for i in range(num_img):
    if i==9061:
        continue
    maxscore=0
    semantic=helper.get_semantic_map(labelroot+"/%08d.png"%(i+1))#test label
    #semantic = one_hot(imread(labelroot+"/%08d.png"%(i+1)))
    semantic=np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)
    scores = np.zeros([20],dtype=np.float32)
    for objid in range(20):
        if np.isnan(avgcolor[i,objid,0]):
            continue
        else:
            if objid in invalidid:
                prob=maxscore
            else:
                prob = kernels[objid](avgcolor[i,objid])
            rarity[i,objid] += 1./prob
            scores[objid]=1./prob
            maxscore = max(maxscore,scores[objid])
    rarity_mask[i] = np.expand_dims(np.sum(np.multiply(semantic,scores),axis=(0,3)),3)/maxscore

np.save("GTA_weightedrarity_mask.npy",rarity_mask)

if not os.path.isdir("rarity/"):
    os.makedirs("rarity/")
for objid in objectid:
    objname = str(objid)
    if not os.path.isdir("GTA_weight_check/%s"%objname):
        os.makedirs("GTA_weight_check/%s"%objname)
    rarity_bin = rarity[:,objid]/np.sum(rarity[:,objid])
    for i in range(1,num_img):
        rarity_bin[i]+=rarity_bin[i-1]
    np.save("rarity/kdecolor_rarity_bin_%d.npy"%objid,rarity_bin)
    order = np.argsort(rarity[:,objid])[::-1]
    for i in range(50):
        copyfile(imageroot+"%08d.png"%(order[i]+1),"GTA_weight_check/%s/top%d.png"%(objname,i))
