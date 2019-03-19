import numpy as np
import tensorflow as tf
import cv2
import os

from utils import confusionMatrix, compute_metrics
from models import build_G

# test
class_number=5
test_path=r'/home/*/test/data'
sess=tf.Session()

image = tf.placeholder(tf.float32, shape = [None,None,None,3],name='image')# have label or no label
#label = tf.placeholder(tf.float32, shape = [None,None,None,class_number],name='label')
is_training = tf.placeholder(tf.bool,name='is_training')

G_score, G_softmax, end_points = build_G(image,class_number,is_training,False,False)# trainable=False

saver=tf.train.Saver()
saver.restore(sess,r'/home/*/model/model.ckpt-250')# load saved model

names=sorted(os.listdir(os.path.join(test_path,'data')))
for i in range(len(names)):
    img=cv2.imread(os.path.join(test_path,'data',names[i]),-1)
    img=np.expand_dims(img,axis=0)# 1 H W 3
    pred=tf.argmax(G_softmax,axis=-1)
    predict=np.array(sess.run(pred,feed_dict={image:img,is_training:False}))[0]# H*W
    cv2.imwrite(os.path.join(test_path,'pred',names[i]),predict)

sess.close()    

confusionM=np.zeros([class_number,class_number]) # 混淆矩阵
for i in range(len(names)):
    predict=cv2.imread(os.path.join(test_path,'pred',names[i]),-1)
    lab=cv2.imread(os.path.join(test_path,'label',names[i]),-1)#0 1 2 3 4
    
    confusionM+=confusionMatrix(pred=predict,label=lab,class_number=class_number)

global_accuray,mean_precision,mean_recall,mean_iou=compute_metrics(confusionM)
print('global_accuray,mean_precision,mean_recall,mean_iou')
print(global_accuray,mean_precision,mean_recall,mean_iou)
