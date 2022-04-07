
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras as k
from ACDC import capsule_layers as CapNet


# from tensorflow.contrib.layers import xavier_initializer
# from tensorflow.contrib.layers import variance_scaling_initializer
# tf.keras.initializers.glorot_normal
# tf.keras.initializers.VarianceScaling
def FRNlayer(x,name):
    
    nu2 = tf.reduce_mean(tf.square(x), axis=[1,2],keepdims=True)
    
    x = x * tf.rsqrt(nu2 + tf.abs(1e-6))
    
    initial1 = tf.zeros(shape=[1,1,1,x.shape[-1].value])
    
    initial2 = tf.keras.initializers.glorot_normal()#xavier_initializer()
    
    gamma = tf.get_variable(name=name+"gamma", shape=[1,1,1,x.shape[-1].value],initializer=initial2)
    
    beta = tf.get_variable(name=name+"beta",initializer=initial1)
    
    tau = tf.get_variable(name=name+"tau",initializer=initial1)
    
    return tf.maximum(gamma * x + beta, tau)

def initweight(shape, stddev, name):
    
    initial = tf.keras.initializers.VarianceScaling()#variance_scaling_initializer()

#    initial = tf.truncated_normal(shape, mean = 0.0, stddev = stddev)

    return tf.get_variable(name = name,shape=shape, initializer = initial)

def initbias(shape, name):

    initial = tf.zeros(shape)
    
    return tf.get_variable(name = name, initializer = initial)

def convlayer(x, ws, bs, stddev, name):
    
    x = tf.reshape(x,[x.shape[0].value,x.shape[1].value,x.shape[2].value,x.shape[4].value])

    weight = initweight(ws, stddev, name + '_w')
    
    bias = initbias(bs, name + '_b')
    
    x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding = 'SAME')
    
    x = tf.nn.bias_add(x, bias)
    
    x = tf.reshape(x,[x.shape[0].value,x.shape[1].value,x.shape[2].value,1,x.shape[3].value])
    
    return x

def relulayer(x):
    
    return tf.nn.relu(x)

def poollayer(x):
    
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')

def deconvlayer(x, ws, stddev, os, strides, name):
    
    weight = initweight(ws, stddev, name + '_w')
    
    return tf.nn.conv2d_transpose(x, weight, os, strides, padding = 'SAME')

def droplayer(x, kprob):
    
    return tf.nn.dropout(x, kprob)

def batchnormallayer(x, training = True):
    
    return k.layers.BatchNormalization()(inputs=x,training=training)

def concatlayer(prex, posx):  
    
    return tf.add(prex,posx)

def concatlayerpro(prex, posx):
    
    
    return tf.concat([prex, posx], axis = -1)



def denselayer(x, laynum, gr, stddev, ks, kprob, training, name):
    
    tmp = batchnormallayer(x, training)
    
    tmp = relulayer(tmp)
    
    px = convlayer(tmp, [ks, ks, x.shape[-1].value, gr], [gr], stddev, name + name + '_lay_0')
    
#    px = relulayer(px)
    
    px = droplayer(px, kprob)
   # print(px.get_shape,"333")
    tmp = tf.concat([x, px], -1)
   # print(tmp.get_shape, "333")
    for ln in range(laynum - 1):
        
        tmp = batchnormallayer(tmp, training)
        
        tmp = relulayer(tmp)
        
        conv = convlayer(tmp, [ks, ks, tmp.shape[3].value, gr], [gr], stddev, name + '_lay_' + str(ln + 1))
        
        conv = droplayer(conv, kprob)
        
#        conv = relulayer(conv)
        
        tmp = tf.concat([tmp, conv], -1)

    tmp = convlayer(tmp, [ks, ks, tmp.shape[-1].value, int(tmp.shape[-1].value / 2)], [int(tmp.shape[-1].value / 2)], stddev, name + '_tmp_')

    return tmp


def denselayercap(x, laynum, gr, stddev, ks, kprob, training, seq):
    
    tmp = batchnormallayer(x, training)
    
    tmp = relulayer(tmp)
    
    px = CapNet.CapsuleBlock(tmp,seq+"0")
          
    px = relulayer(px)
    
    if training == True:
    
        px = droplayer(px, kprob)

    tmp = tf.concat([x, px], -1)
    """
    tmp x(第一个)与px连接
    """
    for ln in range(laynum - 1):
        
        tmp = batchnormallayer(tmp, training)
        
        tmp = relulayer(tmp)
        
        conv = CapNet.CapsuleBlock(tmp, seq + '_lay_' + str(ln + 1))
        
        if training == True:
        
            conv = droplayer(conv, kprob)
        
        tmp = tf.concat([tmp, conv], -1)
        """
        tmp tmp与conv连接
        """

    tmp = convlayer(tmp, [1, 1, tmp.shape[-1].value, gr], [gr], stddev,seq + '_tmp_')

    return tmp

def transdownlayer(x, training = True):
    
    x = tf.reshape(x,[x.shape[0].value,x.shape[1].value,x.shape[2].value,x.shape[4].value])
    
    x = batchnormallayer(x, training)
    
    x = relulayer(x)
    
    x = poollayer(x)
    
    x = tf.reshape(x,[x.shape[0].value,x.shape[1].value,x.shape[2].value,1,x.shape[3].value])
    
    return x

def transuplayer(x, ws, stddev, os, strides, name):    
    
     x = tf.reshape(x,[x.shape[0].value,x.shape[1].value,x.shape[2].value,x.shape[4].value])    
    
     os1 = [os.shape[0].value,os.shape[1].value,os.shape[2].value,os.shape[4].value]   

     x = deconvlayer(x, ws, stddev, os1, strides, name)

     x = relulayer(x)
    
     x = tf.reshape(x,[x.shape[0].value,x.shape[1].value,x.shape[2].value,1,x.shape[3].value])

     return x

def mtx_similar2(arr1, arr2):
    '''
    计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''

    differ = arr1 - arr2
    numera = tf.reduce_sum(differ**2)
    denom = tf.reduce_sum(arr1**2)
    similar = 1 - (numera / denom)

    return similar


def dual_attention(input, old_output):

    height, width, channel = input.get_shape()[1:4]

    x_1_1 = convlayer(x, [3, 3, channel, channel], [channel], [1, 2, 2, 1], "x_1_1")

    x_1_2 = tf.nn.max_pool(x_1_1, [1, height, width, 1], [1, height, width, 1], padding="VALID")

    x_1_3 = tf.sigmoid(x_1_2)

    x_2_1 = convlayer(x, [3, 3, channel, channel], [channel], [1, 2, 2, 1], "x_2_1")

    x_2_2 = tf.multiply(x_2_1, x_1_3)

    x_3_1 = convlayer(x, [3, 3, channel, channel], [channel], [1, 2, 2, 1], "x_3_1")

    x_3_2 = tf.nn.max_pool(x_3_1, [1,1, 1, channel], [1, 1, 1, channel], padding="VALID")

    x_4_1 = convlayer(old_output, [3, 3, channel, channel], [channel], [1, 2, 2, 1], "x_4_1")

    x_4_2 = tf.nn.max_pool(x_4_1, [1, 1, 1, channel], [1, 1, 1, channel], padding="VALID")

    cof = mtx_similar2(x_3_2, x_4_2)

    x_3_3 = x_3_2 + cof*(x_3_2-x_4_2)

    x_3_4 = tf.sigmoid(x_3_3)

    x_2_3 = tf.multiply(x_2_2, x_3_4)

    return x_2_3







if __name__ == '__main__':

    initial2 = tf.keras.initializers.glorot_normal()#xavier_initializer()

    x = tf.get_variable(name="x", shape=[1, 80, 80, 3], initializer=initial2)

    y = tf.get_variable(name="y", shape=[1, 80, 80, 3], initializer=initial2)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sess.run(dual_attention(x, y))

