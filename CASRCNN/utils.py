"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np
from functools import reduce
from itertools import repeat
import nibabel as nib
import tensorflow as tf
import sys


FLAGS = tf.app.flags.FLAGS
"""7-1-2 load h5"""
def read_data(path):
    """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
    """  
    #print('check1')
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'),dtype=np.float32)
        try:
            label = np.array(hf.get('label'),dtype=np.float32)
            return data, label
        except:
            return data

"""7-1-1-2"""
def preprocess(path, scale=3):
    """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
    """
    label_=np.array(nib.load(path).dataobj)
    
    image0=label_[:,:,:,0]
    image0=image0/np.percentile(image0,99)
    image0[image0>1]=1
    image0[image0<1]=sys.float_info.epsilon
    
    image2=label_[:,:,:,1]
    image2=image0/np.percentile(image2,99)
    image2[image2>1]=1
    image2[image2<1]=sys.float_info.epsilon
    
    image4=label_[:,:,:,2]
    image4=image0/np.percentile(image4,99)
    image4[image4>1]=1
    image4[image4<1]=sys.float_info.epsilon
    
    label_[:,:,:,0]=image0
    label_[:,:,:,1]=image2
    label_[:,:,:,2]=image4
    
    image0 = scipy.ndimage.interpolation.zoom(image0, (1./scale))#down-scale
    image0 = scipy.ndimage.interpolation.zoom(image0, (float(label_.shape[0])/image0.shape[0],float(label_.shape[1])/image0.shape[1],float(label_.shape[2])/image0.shape[2]))#up-scale
    image2 = scipy.ndimage.interpolation.zoom(image2, (1./scale))#down-scale
    image2 = scipy.ndimage.interpolation.zoom(image2, (float(label_.shape[0])/image2.shape[0],float(label_.shape[1])/image2.shape[1],float(label_.shape[2])/image2.shape[2]))#up-scale
    image4 = scipy.ndimage.interpolation.zoom(image4, (1./scale))#down-scale
    image4 = scipy.ndimage.interpolation.zoom(image4, (float(label_.shape[0])/image4.shape[0],float(label_.shape[1])/image4.shape[1],float(label_.shape[2])/image4.shape[2]))#up-scale
    
    input_=np.zeros(label_.shape)
    
    input_[:,:,:,0]=image0
    input_[:,:,:,1]=image2
    input_[:,:,:,2]=image4    
     
    return input_, label_

"""7-1-1-1 generating image path for training/testing"""

def prepare_data(sess, folderpath):
    """
  Args:
    folderpath: list of path/to/trainfolder and path/to/testfolder
    namelist: list of absolute path of trainImg names and testImg names
    """
    if FLAGS.is_train:#load traning and testing images
        assert(len(folderpath)==2)
        train_filenames = glob.glob(os.path.join(folderpath[0], "*.nii.gz"))
        test_filenames = sorted(glob.glob(os.path.join(folderpath[1], "*.nii.gz")))
        return [train_filenames,test_filenames]
    else:
        assert(len(folderpath)==1)
        train_filenames = glob.glob(os.path.join(folderpath[0], "*.nii.gz"))
        return [train_filenames]

"""7-1-1-3"""
def save_each(X,y,path):
    path_x=path+'.X'
    path_y=path+'.y'
    np.save(path_x,X)
    np.save(path_y,y)
    return True
    
def make_data(sess, data, label, folderpath, c_dim,mode='train'):
    print(data.shape)
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if mode=='train':
        savepath = os.path.join(folderpath,'train.c'+str(c_dim)+'.h5')
    elif mode=='test':
        savepath = os.path.join(folderpath, 'test.c'+str(c_dim)+'.h5')
    elif mode=='new':
        savepath = os.path.join(folderpath, 'new.c'+str(c_dim)+'.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        if label is not None:
            hf.create_dataset('label', data=label)
    return True
        
    

def imread(path, is_grayscale=True): 
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float32)/255.0
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float32)/255.0

"""7-1-1-2-1"""
def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def generate_patch(h,w,l,input_,label_,padding,config):
    #generate patches
    sub_input_sequence=list()
    sub_label_sequence=list()
    nx = 0
    ny = 0 
    nz=0
    for x in range(0, h-config.image_size+1, config.stride):
        nx+=1
        ny=0
        for y in range(0, w-config.image_size+1, config.stride):
            ny+=1
            nz=0
            for z in range(0,l-config.image_size+1, config.stride):
                nz+=1
                
                sub_input=input_[x:x+config.image_size, y:y+config.image_size, z:z+config.image_size, :]
                sub_label=label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size, z+int(padding):z+int(padding)+config.label_size, :]
                
                # append to list
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    return [nx,ny,nz,sub_input_sequence,sub_label_sequence]
    
"""7-1-1 input setup"""
def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    #if h5 exists, skip
    if not config.make_patch:
        target_path=os.path.join(config.checkpoint_dir,'test.c'+str(config.c_dim)+'.h5')
        if os.path.isfile(target_path):
            return False
            
    # Load data path
    data = prepare_data(sess, [config.trn_folderpath,config.tst_folderpath])#7-1-1-1
    padding =  abs(config.image_size - config.label_size) / 2 # 6
  
    #if training
    trn_sub_input_sequence = []
    trn_sub_label_sequence = []
    tst_sub_input_sequence = []
    tst_sub_label_sequence = []
    #nxny_list=list()
    print("making data...")
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            #preprocess each image
            for scale in ([1.2, 1.6, 2.]):
            
                input_, label_ = preprocess(data[i][j], scale)#7-1-1-2
                #get image size
                if len(input_.shape) == 4:
                    h, w, l,_ = input_.shape
                else:
                    h, w, l = input_.shape
                output=generate_patch(h,w,l,input_,label_,padding,config)
                print(data[i][j])
                if(i==0):#train
                    trn_sub_input_sequence.append(output[3])
                    trn_sub_label_sequence.append(output[4])
                else:#testing
                    tst_sub_input_sequence.append(output[3])
                    tst_sub_label_sequence.append(output[4])
                    #nxny_list.append((output[0],output[1]))
    #flatten list of lists 
    trn_sub_input_sequence=reduce(lambda x,y: x+y,trn_sub_input_sequence)
    trn_sub_label_sequence=reduce(lambda x,y: x+y,trn_sub_label_sequence)
    tst_sub_input_sequence=reduce(lambda x,y: x+y,tst_sub_input_sequence)
    tst_sub_label_sequence=reduce(lambda x,y: x+y,tst_sub_label_sequence)
    #list to numpy
    X_train=np.asarray(trn_sub_input_sequence)
    y_train=np.asarray(trn_sub_label_sequence)
    X_test=np.asarray(tst_sub_input_sequence)
    y_test=np.asarray(tst_sub_label_sequence)
    
    make_data(sess, X_train, y_train, config.checkpoint_dir, config.c_dim,mode='train')
    make_data(sess, X_test, y_test, config.checkpoint_dir, config.c_dim,mode='test')
    return True

def input_setup_test(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    #if h5 exists, skip
    if not config.make_patch:
        target_path=os.path.join(config.checkpoint_dir,'new.c'+str(config.c_dim)+'.h5')
        if os.path.isfile(target_path):
            return False
            
    # Load data path
    data = prepare_data(sess, [config.new_image_path])#7-1-1-1
    padding =  abs(config.image_size - config.label_size) / 2 # 6
  
    #if training
    tst_sub_input_sequence = []
    tst_sub_label_sequence = []
    nxny_list=list()
    for j in range(len(data[0])):
        #preprocess each image
        input_, label_ = preprocess(data[0][j], config.scale)#7-1-1-2
        #get image size
        if len(input_.shape) == 4:
            h, w, l, _ = input_.shape
        else:
            h, w ,l= input_.shape
        output=generate_patch(h,w,l,input_,label_,padding,config)

        tst_sub_input_sequence.append(output[3])
        tst_sub_label_sequence.append(output[4])
        nxny_list.append((output[0],output[1]))
    #flatten list of lists 
    tst_sub_input_sequence=reduce(lambda x,y: x+y,tst_sub_input_sequence)
    #tst_sub_label_sequence=reduce(lambda x,y: x+y,tst_sub_label_sequence)
    #list to numpy
    X_test=np.asarray(tst_sub_input_sequence)
    #y_test=np.asarray(tst_sub_label_sequence)
    
    make_data(sess, X_test,None, config.checkpoint_dir, config.c_dim,mode='new')
    return nxny_list,data[0]
def getXtest_each(imgName,c_dim):
    img=imread(imgName)
    output=np.zeros((img.shape[0],img.shape[1],c_dim),dtype=np.float32)
    for i in range(c_dim):
        output[:,:,i]=img
    return output
def getXtest(folderPath,c_dim):#for whole image based testing
    nameList=glob.glob(os.path.join(folderPath,'*.bmp')) 
    imgs=list(map(getXtest_each,nameList,repeat(c_dim)))
    sameSize=True
    imgSize=imgs[0].shape
    for i in imgs:
        if i.shape!=imgSize:
            sameSize=False
    return imgs,sameSize,nameList
def imsave(image, path):
    return scipy.misc.imsave(path, image,format='bmp')
#"""7-1-2 merge patches into an image"""
def merge(patches, nxny):
    patches=np.asarray(patches)
    print(patches.shape)
    h, w = patches.shape[1], patches.shape[2]
    img = np.zeros((h*nxny[0], w*nxny[1],1))
    for idx, image in enumerate(patches):
        #print(image.shape)
        i = idx % nxny[1]
        j = idx // nxny[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = image
    
    return np.squeeze(img)
