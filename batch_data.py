import numpy as np
import random
import h5py
import cv2
'''
边读取图像，边feed，读取时无法计算，浪费时间。推荐，可以使用多线程。
像tf.data模块，直接读取本地图像，但是是先读取到一个序列中，做到效率更高。
'''
class Data:
    ''' for label and image '''
    def __init__(self,dataset_path):
        self.hdf5=h5py.File(dataset_path,mode='r')
        self.image=self.hdf5['image'].value# not .value ,Memory will explode
        self.img_num=self.image.shape[0]
        self.label=self.hdf5['label'].value
        self.batch_offset = 0 
        self.epochs_completed = 0
    
    def data_augmentation(self,image,label):
        randint=random.randint(1,4)
        if randint==1:# left-right flip
            image=cv2.flip(image,1)
            label=cv2.flip(label,1)
        elif randint==2:# up-down-flip
            image=cv2.flip(image,0)
            label=cv2.flip(label,0)
        elif randint==3:# rotation 90
            M=cv2.getRotationMatrix2D((image.shape[1]//2,image.shape[0]//2),90,1.0)
            image=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
            label=cv2.warpAffine(label,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
        elif randint==4:# rotation 270
            M=cv2.getRotationMatrix2D((image.shape[1]//2,image.shape[0]//2),270,1.0)
            image=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
            label=cv2.warpAffine(label,M,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)
        return image,label
    
    def next_batch(self,batch_size,flag='train'):
        self.start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.img_num:
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            perm = np.arange(self.img_num)
            np.random.shuffle(perm)
            if flag=='train' or flag == 'valid':
                self.image=self.image[perm]
                self.label=self.label[perm]
            self.start = 0
            self.batch_offset = batch_size
        self.end = self.batch_offset
        if flag == 'train' or flag == 'valid':
            imgs=self.image[self.start:self.end]
            labs=self.label[self.start:self.end]

        perms=random.sample(range(batch_size),batch_size//2)
        for i in perms:
            img,lab=imgs[i],labs[i]
            img,lab=self.data_augmentation(img,lab)
            imgs[i],labs[i]=img,lab
            
        return imgs,labs
        
class data:
    ''' for no label '''
    def __init__(self,dataset_path):
        self.hdf5=h5py.File(dataset_path,mode='r')
        self.image=self.hdf5['image'].value
        self.img_num=self.image.shape[0]
        self.batch_offset = 0 
        self.epochs_completed = 0

    def next_batch(self,batch_size,flag='train'):
        self.start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.img_num:
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            perm = np.arange(self.img_num)
            np.random.shuffle(perm)
            if flag=='train' or flag == 'valid':
                self.image=self.image[perm]
            self.start = 0
            self.batch_offset = batch_size
        self.end = self.batch_offset
        if flag == 'train' or flag == 'valid':
            imgs=self.image[self.start:self.end]
        return imgs
