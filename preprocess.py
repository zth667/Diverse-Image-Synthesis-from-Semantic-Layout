from scipy.misc import imread,imresize,imsave
import numpy as np
import os
data_root="./"
lst=os.listdir(data_root+"train/image")
i=0
for b in lst:
    i+=1
    rawimg = imread(os.path.join(data_root,"train/image",b))
    rawimg = imresize(rawimg,(256,512))
    imsave("RGB256Full/%08d.png"%i,rawimg)
lst3=os.listdir(data_root+"val/image")
i=0
for b in lst3:
    i+=1
    rawimg = imread(os.path.join(data_root,"val/image",b))
    rawimg = imresize(rawimg,(256,512))
    imsave("RGB256Full/%08d.png"%(i+100000),rawimg)
i=0
for b in lst:
    i+=1
    rawimg = imread(os.path.join(data_root+"train/label",b))
    rawimg = imresize(rawimg,(256,512),'nearest')
    imsave("Label256Full/%08d.png"%i,rawimg)
i=0
for b in lst3:
    i+=1
    rawimg = imread(os.path.join(data_root,"val/label",b))
    rawimg = imresize(rawimg,(256,512),'nearest')
    imsave("Label256Full/%08d.png"%(i+100000),rawimg)
