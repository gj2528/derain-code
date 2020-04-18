import tensorflow as tf
import dilated
from cnn_basenet import CNNBaseModel as cb

def SmoothDilatedResidualBlock(x, dilation=1,index=1, reuse=False):
    with tf.variable_scope("SmoothDilatedResidualBlock_%d"% index, reuse=reuse):
        y = dilated._smoothed_dilated_conv2d_SSC(x, 3, 64, dilation*2-1, name='dilated_conv_1')
        y = cb.instancenorm(y, name='norm_1')
        y1 = cb.lrelu(y, name='relu_1')

        y = dilated._smoothed_dilated_conv2d_SSC(y1, 3, 64, dilation*2-1, name='dilated_conv_2')
        y2 = cb.instancenorm(y, name='norm_2')
        y = cb.lrelu(y1 + y2, name='relu_2')
        return y

def ResidualBlock(x, dilation=1, reuse=False):
    with tf.variable_scope("ResidualBlock", reuse=reuse):
        y = cb.dilation_conv(x, 3, 64, dilation, name='diconv_1')
        y = cb.instancenorm(y, name='norm_1')
        y1 = cb.lrelu(y, name='relu_1')

        y = cb.dilation_conv(x, 3, 64, dilation, name='diconv_2')
        y2 = cb.instancenorm(y, name='norm_2')
        y = cb.lrelu(y1 + y2, name='relu_2')
        return y

def GCANet(x, out_c=3, only_residual=False, reuse=False):
    with tf.variable_scope("GCANet", reuse=reuse):
        y = cb.conv2d(x, 64, kernel_size=3, padding='SAME', stride=1, use_bias=False, name='conv_1')
        y = cb.instancenorm(y, name='norm_1')
        y = cb.lrelu(y, name='relu_1')

        y = cb.conv2d(y, 64, kernel_size=3, padding='SAME', stride=1, use_bias=False, name='conv_2')
        y = cb.instancenorm(y, name='norm_2')
        y = cb.lrelu(y, name='relu_2')

        y = cb.conv2d(y, 64, kernel_size=3, padding='SAME', stride=2, use_bias=False, name='conv_3')
        y = cb.instancenorm(y, name='norm_3')
        y1 = cb.lrelu(y, name='relu_3')

        y = SmoothDilatedResidualBlock(y1, 2, 1)
        y = SmoothDilatedResidualBlock(y, 2, 2)
        y = SmoothDilatedResidualBlock(y, 2, 3)
        y2 = SmoothDilatedResidualBlock(y, 4, 4)
        y = SmoothDilatedResidualBlock(y2, 4, 5)
        y = SmoothDilatedResidualBlock(y, 4, 6)
        y3 = ResidualBlock(y, 1)

        gates = tf.concat([y1, y2, y3], axis=-1)
        gates = cb.conv2d(gates, 3, kernel_size=3, padding='SAME', stride=1, use_bias=True, name='conv_4')
        gated_y = y1 * gates[:, :, :, 0:-2] + y2 * gates[:, :, :, 1:-1] + y3 * gates[:, :, :, 2:]

        y = cb.deconv2d(gated_y, out_channel=64, kernel_size=4, padding='SAME', stride=2, name='deconv1')
        y = cb.instancenorm(y, name='norm_4')
        y = cb.lrelu(y, name='relu_4')

        y = cb.deconv2d(y, out_channel=64, kernel_size=3, padding='SAME', stride=1, name='deconv2')
        y = cb.instancenorm(y, name='norm_5')
        y = cb.lrelu(y, name='relu_5')

        if only_residual:
            y = cb.deconv2d(y, out_channel=out_c, kernel_size=1, padding='SAME', stride=1, name='deconv3')
        else:
            y = cb.deconv2d(y, out_channel=out_c, kernel_size=1, padding='SAME', stride=1, name='deconv3')
            y = cb.lrelu(y, name='relu_6')

        return y
