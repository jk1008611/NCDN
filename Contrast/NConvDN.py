
import tensorflow.compat.v1 as tf
import Layerspropro as Layers
import Simdata
import numpy as np
import Parainit


def NConvDN(x, cn, istraining):
    layer = {}

    stddev = 0.02

    num_layer = 3

    kprob = 0.5

    if istraining == 1:

        training = True

    else:

        training = False

    dic = Parainit.BLUinit()

    x = tf.reshape(x, [x.shape[0].value, x.shape[1].value, x.shape[2].value, 1, x.shape[3].value])

    x_1_1 = layer['conv_1_1'] = Layers.convlayer(x, dic['conv_1_1']['ws'], dic['conv_1_1']['bs'], stddev, 'conv_1_1')

    x_1_3 = layer['Cap_1_1'] = Layers.denselayercap(x_1_1, num_layer, x_1_1.shape[-1].value, stddev, 3, kprob, training,
                                                    "c1")

    x_1_4 = layer['concat_1_1'] = Layers.concatlayer(x_1_3, x_1_1)

    x_1_5 = layer['transdown_1_1'] = Layers.transdownlayer(x_1_4)

    x_1_6 = layer['Cap_1_2'] = Layers.denselayercap(x_1_5, num_layer, x_1_5.shape[-1].value, stddev, 3, kprob, training,
                                                    "c2")

    x_1_7 = layer['concat_1_2'] = Layers.concatlayer(x_1_6, x_1_5)

    x_1_8 = layer['transdown_1_2'] = Layers.transdownlayer(x_1_7, training)

    x_1_9 = layer['Cap_1_3'] = Layers.denselayercap(x_1_8, num_layer, x_1_8.shape[-1].value, stddev, 3, kprob, training,
                                                    "c3")

    x_1_10 = layer['transup_1_1'] = Layers.transuplayer(x_1_9,

                                                        [3, 3, x_1_7.shape[-1].value, x_1_9.shape[-1].value], stddev,

                                                        x_1_7, [1, 2, 2, 1], 'transup_1_1')

    x_1_11 = layer['concat_1_3'] = Layers.concatlayer(x_1_10, x_1_7)

    x_1_12 = layer['Cap_1_4'] = Layers.denselayercap(x_1_11, num_layer, x_1_11.shape[-1].value, stddev, 3, kprob,
                                                     training, "c4")

    #
    #    x_2_1 = layer['transup_2_1'] = Layers.transuplayer(x_1_6,
    #
    #                 [3, 3, x_1_6.shape[-1].value, x_1_6.shape[-1].value], stddev,
    #
    #                x_1_4, [1, 2, 2, 1], 'transup_2_1')
    #
    #    x_2_2 = layer['concat_2_1'] = Layers.concatlayer(x_2_1, x_1_4)

    #    x_2_3 = layer['Cap_1_5'] = Layers.denselayercap(x_2_2, num_layer,x_2_2.shape[-1].value,stddev,3,kprob,training,"d1")

    x_1_13 = layer['transup_1_2'] = Layers.transuplayer(x_1_12,

                                                        [3, 3, x_1_4.shape[-1].value, int(x_1_12.shape[-1].value)],
                                                        stddev,

                                                        x_1_4, [1, 2, 2, 1], 'transup_1_2')

    #    x_2_4 = layer['conv_2_1'] = Layers.convlayer(x_2_3,
    #
    #                 [3, 3, x_2_3.shape[-1].value, int(x_2_3.shape[-1].value )], [int(x_2_3.shape[-1].value )], stddev, 'conv_2_1')

    x_1_14 = layer['concat_1_4'] = Layers.concatlayer(x_1_13, x_1_4)

    x_1_15 = layer['Cap_1_6'] = Layers.denselayercap(x_1_14, num_layer, x_1_14.shape[-1].value, stddev, 3, kprob,
                                                     training, "d2")

    x_1_16 = layer['conv_1_3'] = Layers.convlayer(x_1_15,

                                                  [3, 3, x_1_15.shape[-1].value, int(x_1_15.shape[-1].value / 2)],
                                                  [int(x_1_15.shape[-1].value / 2)], stddev, 'conv_1_3')

    #    x_3_1 = layer['concat_1_5'] = Layers.concatlayerpro(x_1_16, x_2_4)
    #
    #
    #
    #    x_3_1 = layer['concat_1_6'] = Layers.concatlayerpro(x_3_1, x_1_1)

    x_3_2 = layer['conv_3_1'] = Layers.convlayer(x_1_16,

                                                 [1, 1, x_1_16.shape[-1].value, int(x_1_16.shape[-1].value / 2)],
                                                 [int(x_1_16.shape[-1].value / 2)], stddev, 'conv_3_1')

    x_3_3 = layer['conv_3_2'] = Layers.convlayer(x_3_2,

                                                 [1, 1, x_3_2.shape[-1].value, int(x_3_2.shape[-1].value / 2)],
                                                 [int(x_3_2.shape[-1].value / 2)], stddev, 'conv_3_2')

    x_3_4 = layer['conv_3_3'] = Layers.convlayer(x_3_3,

                                                 [1, 1, x_3_3.shape[-1].value, cn], [cn], stddev, 'conv_3_3')

    x_3_4 = tf.reshape(x_3_4, [x_3_4.shape[0].value, x_3_4.shape[1].value, x_3_4.shape[2].value, x_3_4.shape[4].value])

    return x_3_4


def paranum():
    total_parameters = 0

    for variable in tf.trainable_variables():

        shape = variable.get_shape()

        variable_parameters = 1

        for dim in shape:
            variable_parameters *= dim.value

        int(variable_parameters)

        total_parameters += variable_parameters

    print(total_parameters)


def test():
    data = Simdata.siminput([80, 80, 3])

    data = np.expand_dims(data, 0)

    print(data.shape)

    data = tf.cast(data, tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(tf.report_uninitialized_variables()))

        x = NConvDN(data, 3, True)

        print(x.shape)

    paranum()


if __name__ == '__main__':
    test()