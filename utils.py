import numpy as np
import matplotlib.mlab as mm
import tensorflow as tf

''' train need '''

def sig_loss(logits,true_label=False):
    # binary cross-entropy
    if true_label:
        labels=tf.ones_like(logits,dtype=tf.float32)
    else:
        labels=tf.zeros_like(logits,dtype=tf.float32)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))

def nl_soft_loss(pred,confidence,class_number,mask_T):
    ''' no label softmax loss
    pred: NHWC
    confidence: NHW*1 , confidence is D's sigomid's , >= mask_T , is confidence
            then logits's argmax[confidence] ,finally one_hot
    '''
    #label: logits.argmax then one_hot ,finally reshape=>(-1,class_number)
    label=tf.argmax(pred,axis=-1)# NHWC=>NHW
    one_hot_label=tf.one_hot(label,class_number)# NHWC
    sig_c=[]
    for i in range(class_number):
        sig_c.append(confidence)
    sig_c=tf.concat(sig_c,axis=-1)
    
    index=tf.where(sig_c>=mask_T)# sig_c nhwc get index
    
    logits=tf.gather_nd(pred,index)
    logits=tf.reshape(logits,shape=[-1,class_number])# (nn,C) default axis=N+1
    labels=tf.gather_nd(one_hot_label,index)
    labels=tf.reshape(labels,shape=[-1,class_number])# (nn,C)
    
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits))
    
    return loss
    
def soft_loss(logits,labels):
    # multi-label-crosss-entropy
    labels=tf.to_float(labels)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits))
'''
because of G and semi need the same lr and weights ,so we modify 'train' func to 2 func : update_lr / update_optim
'''
def train(loss,learning_rate,learning_rate_decay_steps,learning_rate_decay_rate,global_step):
    ''' globale_step automatic update +1  and update lr '''
    decay_lr=tf.train.exponential_decay(learning_rate,global_step,learning_rate_decay_steps,
                                        learning_rate_decay_rate,staircase=True)
    # execute update_ops to update batch_norm weights, bn weights automatic add to GraphKeys but need get_collection
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer=tf.train.AdamOptimizer(decay_lr)
        train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op

def update_lr(learning_rate,learning_rate_decay_steps,learning_rate_decay_rate,global_step):
    ''' globale_step automatic update +1  and update lr '''
    decay_lr=tf.train.exponential_decay(learning_rate,global_step,learning_rate_decay_steps,
                                        learning_rate_decay_rate,staircase=True)
    return decay_lr

def update_optim(loss,decay_lr,var_list,global_step=None):
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer=tf.train.AdamOptimizer(decay_lr)
        # delete global_step,because global_step auto add 1,because we want to use G weight share and same lr
        train_op=optimizer.minimize(loss,var_list=var_list,global_step=global_step) 
    return train_op
    

''' test need '''
# compute accuracy
def confusionMatrix(pred,label,num_class):
    # pred [H,W] , label [H,W]
    HXJZ=np.zeros([num_class,num_class])
    for i in range(num_class):
        for j in range(num_class):
            Temp=pred[label==i]
            HXJZ[i,j]=len(mm.find(Temp==j))
    return HXJZ

def compute_metrics(confus_M):
    # one image or multi image
    H,W=confus_M.shape
    if H!=W:
        raise Exception('Error: num_class must be H=W')
    # compute accuray
    true_pred=0
    for i in range(H):
        true_pred+=confus_M[i,i]
    global_accuray=true_pred/np.sum(confus_M)
    # precision
    mean_precision=0.0
    for i in range(H):
        mean_precision+=confus_M[i,i]/np.sum(confus_M[:,i])
    mean_precision/=H
    # recall
    mean_recall=0.0
    for i in range(H):
        mean_recall+=confus_M[i,i]/np.sum(confus_M[i,:])
    mean_recall/=H
    # mean iou
    mean_iou=0.0
    for i in range(H):
        mean_iou+=confus_M[i,i]/(np.sum(confus_M[i,:])+np.sum(confus_M[:,i])-confus_M[i,i])
    mean_iou/=H
    return global_accuray,mean_precision,mean_recall,mean_iou




