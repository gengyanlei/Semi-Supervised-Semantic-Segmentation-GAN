import tensorflow as tf
from tensorflow.contrib import slim

@slim.add_arg_scope
def max_pool(bottom,kernel,stride,name):
    out=slim.max_pool2d(bottom,kernel,stride,'SAME',scope=name)
    return out

@slim.add_arg_scope
def conv(bottom,num_out,kernel,stride,activation_fn,is_training,name):
    out=slim.conv2d(bottom,num_out,kernel,stride,'SAME',activation_fn=None,weights_regularizer=slim.l2_regularizer(5e-4),scope=name)
    if activation_fn is not None:
        out=slim.batch_norm(out,center=True,scale=True,activation_fn=activation_fn,is_training=is_training,scope=name+'_bn')
    return out

def upsample(bottom,size):
    out=tf.image.resize_bilinear(bottom,size=size)
    return out
'''
    G_Net
'''
def build_G(image,class_number,is_training,reuse=False,trainable=False):
    '''
    Args:
        image: cv2.imread BGR =>0-1
        class_number: ...
        is_training: train or test, batch_norm param
        reuse: Weight Sharing
        trainable: train->True ; and test->False
        #测试时，虽然trainable设置为True，没有优化器，计算loss，并不影响测试，但是逻辑不通。因此，设置一下这个参数。
        While testing, although trainable is set to True, there is no optimizer, calculating loss does not affect the test, 
        but it is not logical. Therefore, set this parameter.#
    '''
    with tf.name_scope('processing'):
            b,g,r=tf.split(image,3,axis=3)
            image=tf.concat([
                        b*0.00390625,
                        g*0.00390625,
                        r*0.00390625],axis=3)
    end_points_collections='GNet_End_Points'
    with tf.variable_scope('u_net'):
        with slim.arg_scope([conv],activation_fn=tf.nn.relu,is_training=is_training):
            with slim.arg_scope([slim.conv2d,slim.batch_norm,slim.max_pool2d],
                            outputs_collections=end_points_collections):
                with slim.arg_scope([slim.conv2d,slim.batch_norm],
                                    reuse=reuse,trainable=trainable):# need main add tf.losses.get_regularization_loss
                    conv1_1=conv(image,64,3,1,name='conv1_1')
                    conv1_2=conv(conv1_1,64,3,1,name='conv1_2')
                    pool1=max_pool(conv1_2,2,2,'pool1')
                    
                    conv2_1=conv(pool1,128,3,1,name='conv2_1')
                    conv2_2=conv(conv2_1,128,3,1,name='conv2_2')
                    pool2=max_pool(conv2_2,2,2,'pool2')
                    
                    conv3_1=conv(pool2,256,3,1,name='conv3_1')
                    conv3_2=conv(conv3_1,256,3,1,name='conv3_2')
                    pool3=max_pool(conv3_2,2,2,'pool3')
                    
                    conv4_1=conv(pool3,512,3,1,name='conv4_1')
                    conv4_2=conv(conv4_1,512,3,1,name='conv4_2')
                    pool4=max_pool(conv4_2,2,2,'pool4')
                    
                    conv5_1=conv(pool4,512,3,1,name='conv5_1')
                    conv5_2=conv(conv5_1,512,3,1,name='conv5_2')
                    
                    # upsample and decoder
                    # block 1
                    up6=upsample(conv5_2,tf.shape(conv4_2)[1:3])
                    concat6=tf.concat([up6,conv4_2],axis=-1)
                    conv6_1=conv(concat6,512,3,1,name='conv6_1')
                    conv6_2=conv(conv6_1,512,3,1,name='conv6_2')
                    
                    # block 2
                    up7=upsample(conv6_2,tf.shape(conv3_2)[1:3])
                    concat7=tf.concat([up7,conv3_2],axis=-1)
                    conv7_1=conv(concat7,256,3,1,name='conv7_1')
                    conv7_2=conv(conv7_1,256,3,1,name='conv7_2')
                    
                    # block 3
                    up8=upsample(conv7_2,tf.shape(conv2_1)[1:3])
                    concat8=tf.concat([up8,conv2_1],axis=-1)
                    conv8_1=conv(concat8,128,3,1,name='conv8_1')
                    conv8_2=conv(conv8_1,128,3,1,name='conv8_2')
                    
                    # block 4
                    up9=upsample(conv8_2,tf.shape(conv1_1)[1:3])
                    concat9=tf.concat([up9,conv1_1],axis=-1)
                    conv9_1=conv(concat9,64,3,1,name='conv9_1')
                    conv9_2=conv(conv9_1,64,3,1,name='conv9_2')
                    
                    score=conv(conv9_2,class_number,1,1,activation_fn=None,name='score')
                    
                    softmax=tf.nn.softmax(score,axis=-1)
                    softmax=slim.utils.collect_named_outputs(end_points_collections,'softmax',softmax)
                    pred=tf.argmax(softmax,axis=-1)
                    pred=slim.utils.collect_named_outputs(end_points_collections,'pred',pred)
                    
                    end_points=slim.utils.convert_collection_to_dict(end_points_collections)
            
    return score, softmax, end_points

def leaky_relu(x,alpha=0.2):
    return tf.maximum(x,alpha*x)

'''
    D_Net
'''
def build_D(inputs,reuse=False):
    '''
    inputs: G's output or label
    '''
    ndf=64
#    end_points_collections='DNet_End_Points'
    with tf.variable_scope('FCDiscriminator'):
        conv1=slim.conv2d(inputs,ndf,3,2,activation_fn=leaky_relu,reuse=reuse,scope='conv1')
        conv2=slim.conv2d(conv1,ndf*2,3,2,activation_fn=leaky_relu,reuse=reuse,scope='conv2')
        conv3=slim.conv2d(conv2,ndf*4,3,2,activation_fn=leaky_relu,reuse=reuse,scope='conv3')
        conv4=slim.conv2d(conv3,ndf*8,3,2,activation_fn=leaky_relu,reuse=reuse,scope='conv4')
        classifier=slim.conv2d(conv4,1,3,1,activation_fn=None,reuse=reuse,scope='classifier')
        
        score=upsample(classifier,tf.shape(inputs)[1:3])
        sigmoid=tf.nn.sigmoid(score)# NHW*1
        
    return score,sigmoid
