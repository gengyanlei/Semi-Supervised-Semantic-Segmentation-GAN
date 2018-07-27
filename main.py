'''
main.py is for train ; and test is created in another .py file
'''
import tensorflow as tf
from tensorflow.contrib import slim

import batch_data
from models import build_G, build_D
from utils import sig_loss, soft_loss, nl_soft_loss, update_lr, update_optim

def main():
    dataset_path=r'/home/*/f3.hdf5'# 1 or 1/2 dataset
    dataset_nl_path=r'/home/*/hdf5/f2.hdf5'
    model_save_path=r'/home/*/model/model.ckpt'
    
    learning_rate_G=2.5e-4
    learning_rate_D=1e-4
    class_number=5
    
    batch_size=8
    max_iters=20000
    start_semi=5000
    mask_T=0.2
    lambda_abv=0.01
    lambde_semi=0.1
    
    dataset_param=batch_data.Data(dataset_path)
    iter_num=dataset_param.img_num//batch_size
    print(iter_num)
    dataset_nl_param=batch_data.data(dataset_nl_path)
    
    # G_net placeholder
    image = tf.placeholder(tf.float32, shape = [None,None,None,3],name='image')# have label or no label
    label = tf.placeholder(tf.float32, shape = [None,None,None,class_number],name='label')
    is_training = tf.placeholder(tf.bool,name='is_training')
    
    G_score, G_softmax, end_points = build_G(image,class_number,is_training,False,True)
    #print(end_points)
    L2_loss=tf.losses.get_regularization_loss()
#    init_fn=slim.assign_from_checkpoint_fn(r'/home/*/model.ckpt',slim.get_model_variables('u_net'))
    
    # build D two time
    D_score_fake, D_sigmoid_fake = build_D(G_softmax)
    ''' Core loss: Loss_Seg_Adv,  Loss_Semi, Loss_D '''
    
    with tf.name_scope('loss_g'):
        Loss_Seg_Adv=soft_loss(logits=G_score,labels=label)+lambda_abv*sig_loss(D_score_fake,True)+L2_loss#######
        tf.summary.scalar('loss_g',Loss_Seg_Adv)
    
    Loss_Semi=lambde_semi*nl_soft_loss(G_score,D_sigmoid_fake,class_number,mask_T)###################
        
    Loss_D_fake=sig_loss(D_score_fake,False)
    
    D_score_real, D_sigmoid_real = build_D(label,reuse=True)
    Loss_D_real=sig_loss(D_score_real,True)
    
    with tf.name_scope('loss_d'):
        Loss_D=Loss_D_fake+Loss_D_real#############
        tf.summary.scalar('loss_d',Loss_D)
    # get all g d vars list
    all_vars=tf.trainable_variables()
    g_vars=[var for var in all_vars if 'u_net' in var.name]
    d_vars=[var for var in all_vars if 'FCDiscriminator' in var.name]
    
    ''' adjust lr ; loss_semi with loss_seg_adv same lr 
        because it's a graph , global_step is a Variable ,update same time.
    '''
    global_step_G=tf.Variable(tf.constant(0))
    lr_g=update_lr(learning_rate_G,max_iters//4,0.1,global_step_G)
    train_op_G=update_optim(Loss_Seg_Adv,lr_g,g_vars,global_step_G)
    
    #lr_g=update_lr(learning_rate_G,max_iters//4,0.1,global_step_G)
    train_op_Semi=update_optim(Loss_Semi,lr_g,g_vars) # loss_semi's lr = loss_seg_adv's lr
    
    global_step_D=tf.Variable(tf.constant(0))
    lr_d=update_lr(learning_rate_D,max_iters//4,0.1,global_step_D)
    train_op_D=update_optim(Loss_D,lr_d,d_vars,global_step_D)
    
    sess=tf.Session()
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter('./model/',sess.graph)
    
    init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    saver=tf.train.Saver(max_to_keep=1)
    
#    init_fn(sess) # fine-tune
    continue_learning=False
    if continue_learning:
        ckpt=tf.train.get_checkpoint_state(r'/home/*/model')
        saver.restore(sess,ckpt.model_checkpoint_path)
        
    for iters in range(max_iters):
        loss_semi=0
        if iters >= start_semi:
            imgs_nl=dataset_nl_param.next_batch(batch_size)
            sess.run(train_op_Semi,feed_dict={image:imgs_nl,is_training:True})# no label ,generate by itself
            loss_semi += sess.run(Loss_Semi,feed_dict={image:imgs_nl,is_training:True})
        imgs,labs=dataset_param.next_batch(batch_size)
        sess.run(train_op_G,feed_dict={image:imgs,label:labs,is_training:True})
        sess.run(train_op_D,feed_dict={image:imgs,label:labs,is_training:True})
        # add loss to tensorboard
        result=sess.run(merged,feed_dict={image:imgs,label:labs,is_training:True})
        writer.add_summary(result,iters)
        
        loss_seg_adv, loss_d = sess.run([Loss_Seg_Adv,Loss_D],feed_dict={image:imgs,label:labs,is_training:True})
        
        #print('loss_seg_adv: %.2f ,loss_d: %.2f ,loss_semi: %.2f '%(loss_seg_adv,loss_d,loss_semi))
        print('\rloss_seg_adv: %.2f ,loss_d: %.2f ,loss_semi: %.2f '%(loss_seg_adv,loss_d,loss_semi),end='',flush=True)
        
        if iters%1000==0 and iters!=0:
            saver.save(sess,save_path=model_save_path,global_step=iters)

    
    
    
    
    
    
if __name__=='__main__':
    main()
