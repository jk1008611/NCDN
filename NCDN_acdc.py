# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from ACDC import Simdata, Layerspropro as Layers
import numpy as np



def NCDN(x, cn, istraining):
    layer = {}

    stddev = 0.02

    num_layer = 3

    kprob = 0.5

    if istraining == 1:

        training = True

    else:

        training = False

    x = tf.reshape(x, [x.shape[0].value, x.shape[1].value, x.shape[2].value, 1, 1])

    x_1_1 = layer['conv1'] = Layers.convlayer(x, [3, 3, 1, 32], [32], stddev, 'conv1')

    x_1_2 = layer['cap1'] = Layers.denselayercap(x_1_1, num_layer, x_1_1.shape[-1].value, stddev, 3, kprob, training,
                                                 "1")

    x_1_3 = layer['concat1'] = Layers.concatlayer(x_1_1, x_1_2)

    x_1_4 = layer['transdown1'] = Layers.transdownlayer(x_1_3)

    x_1_5 = layer['Cap2'] = Layers.denselayercap(x_1_4, num_layer, x_1_4.shape[-1].value, stddev, 3, kprob, training,
                                                 "2")

    x_1_6 = layer['concat2'] = Layers.concatlayer(x_1_4, x_1_5)

    x_1_7 = layer['transdown2'] = Layers.transdownlayer(x_1_6, training)

    x_1_8 = layer['Cap3'] = Layers.denselayercap(x_1_7, num_layer, x_1_7.shape[-1].value, stddev, 3, kprob, training,
                                                 "3")

    x_1_9 = layer['concat3'] = Layers.concatlayer(x_1_7, x_1_8)

    x_1_10 = layer['transdown3'] = Layers.transdownlayer(x_1_9, training)

    x_1_11 = layer['Cap4'] = Layers.denselayercap(x_1_10, num_layer, x_1_10.shape[-1].value, stddev, 3, kprob, training,
                                                  "4")

    x_1_12 = layer['transup1'] = Layers.transuplayer(x_1_11,

                                                     [3, 3, x_1_11.shape[-1].value, x_1_9.shape[-1].value], stddev,

                                                     x_1_9, [1, 2, 2, 1], 'transup1')

    x_1_13 = layer['concat4'] = Layers.concatlayer(x_1_9, x_1_12)

    x_1_14 = layer['Cap5'] = Layers.denselayercap(x_1_13, num_layer, x_1_13.shape[-1].value, stddev, 3, kprob, training,
                                                  "5")

    x_1_15 = layer['transup2'] = Layers.transuplayer(x_1_14,

                                                     [3, 3, x_1_14.shape[-1].value, x_1_6.shape[-1].value], stddev,

                                                     x_1_6, [1, 2, 2, 1], 'transup2')

    x_1_16 = layer['concat5'] = Layers.concatlayer(x_1_6, x_1_15)

    x_1_17 = layer['Cap6'] = Layers.denselayercap(x_1_16, num_layer, x_1_16.shape[-1].value, stddev, 3, kprob, training,
                                                  "6")

    x_1_18 = layer['transup2'] = Layers.transuplayer(x_1_17,

                                                     [3, 3, x_1_17.shape[-1].value, x_1_3.shape[-1].value], stddev,

                                                     x_1_3, [1, 2, 2, 1], 'transup3')

    x_1_19 = layer['concat5'] = Layers.concatlayer(x_1_3, x_1_18)

    x_1_20 = layer['Cap7'] = Layers.denselayercap(x_1_19, num_layer, x_1_19.shape[-1].value, stddev, 3, kprob, training,
                                                  "7")

    x_3_2 = layer['conv2'] = Layers.convlayer(x_1_20,

                                              [1, 1, x_1_20.shape[-1].value, int(x_1_20.shape[-1].value / 2)],
                                              [int(x_1_16.shape[-1].value / 2)], stddev, 'conv2')

    x_3_3 = layer['conv3'] = Layers.convlayer(x_3_2,

                                              [1, 1, x_3_2.shape[-1].value, int(x_3_2.shape[-1].value / 2)],
                                              [int(x_3_2.shape[-1].value / 2)], stddev, 'conv3')

    x_3_4 = layer['conv4'] = Layers.convlayer(x_3_3,

                                              [1, 1, x_3_3.shape[-1].value, int(x_3_3.shape[-1].value / 2)],
                                              [int(x_3_3.shape[-1].value / 2)], stddev, 'conv4')

    x_3_5 = layer['conv5'] = Layers.convlayer(x_3_4, [1, 1, x_3_4.shape[-1].value, 4], [4], stddev, 'conv5')

    x_3_5 = tf.reshape(x_3_5, [x_3_5.shape[0].value, x_3_5.shape[1].value, x_3_5.shape[2].value, x_3_5.shape[4].value])

    return x_3_5


def paranum():
    total_parameters = 0

    for variable in tf.trainable_variables():

        shape = variable.get_shape()

        variable_parameters = 1

        for dim in shape:
            variable_parameters *= dim.value

        int(variable_parameters)

        total_parameters += variable_parameters

    # print(total_parameters)


def test():
    data = Simdata.siminput([128, 128, 1])

    data = np.expand_dims(data, 0)

    print(data.shape)

    data = tf.cast(data, tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # print(sess.run(tf.report_uninitialized_variables()))

        x = NCDN(data, 4, True)

        print(x.shape)

    paranum()


if __name__ == '__main__':
    test()
