
import os
import time
import numpy as np
import tensorflow as tf
from load_tfrecords import read_and_decode, read_and_decode_for_one_img
from GCANet_tensorflow import *
import datetime
from utils import *

################# Select GPU device ##################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
######################################################

tf.reset_default_graph()

################# Network parameters ##################
num_feature = 64  # number of feature maps
num_channels = 3  # number of inputs's channels
patch_height = 128  # training patch height
patch_width = 128  # training patch width
learning_rate = 0.001  # learning rate
iterations = 60000  # number of iterations
counter = 0  # start step of training
batch_size = 2  # number of batch
save_model_path = './checkpoint/GCANet_model/'  # path to save model
save_log_path = './log/'  # path to save log
save_sample_path = './sample/'  # path to save sample
model_name = 'GCANet_model'  # name of model
tfrec_filename_train = './TrainData/Rain_train.tfrecords'  # the path of tfrecords for training
tfrec_filename_val = './TrainData/Rain_val.tfrecords'  # the path of tfrecords for validation
load_model = True  # flag of load_model
checkpoint_dir = './checkpoint/GCANet_model/'
######################################################

if __name__ == '__main__':

    starttime = datetime.datetime.now()
    print("当前时间: ", str(starttime).split('.')[0])
    rain_img_train, clean_img_train = read_and_decode(tfrec_filename_train, patch_height, patch_width, batch_size)
    rain_img_val, clean_img_val = read_and_decode_for_one_img(tfrec_filename_val)
    print('load data success')
    # with tf.Session() as sess:
    #     rain_img_train = rain_img_train.eval()
    #     rain_img_val = rain_img_val.eval
    # img = np.array(rain_img_train).astype('float')
    # edge_img_train = edge_compute(img)
    # img_train = tf.concat([rain_img_train, edge_img_train], 3)
    # print(img_train, 'img_train')
    y = GCANet(rain_img_train)
    # edge_img_val = edge_compute(rain_img_val)
    # img_val = tf.concat([rain_img_val, edge_img_val], 3)
    y_val = GCANet(rain_img_val, reuse=True)

    loss_mse_train = tf.reduce_mean(tf.square(y - clean_img_train))
    loss_ssim_train = loss_ssim(y, clean_img_train, batch_size, num_channels)
    loss_train = loss_mse_train + loss_ssim_train

    loss_train_sum = tf.summary.scalar('loss_train', loss_train)

    global_step = tf.Variable(0, trainable=False)
    t_vars = tf.trainable_variables()
    rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=10000,
                                      decay_rate=0.1,
                                      staircase=True)
    optim = tf.train.RMSPropOptimizer(learning_rate=rate).minimize(loss_train, var_list=t_vars, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=5)

    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_sample_path):
        os.makedirs(save_sample_path)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter(save_log_path, sess.graph)

        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(save_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Loading success, global step is %s' % global_step)
                counter = int(global_step)

        for itr in range(counter, iterations + 1):
            start_time = time.time()
            _, summary_str_loss = sess.run([optim, loss_train_sum])
            counter += 1
            if itr == 0:
                print('No models, re-train')
            if np.mod(itr, 1) == 0:
                loss_train_value = sess.run(loss_train)
                print('iterations:[%4d/%4d], time:%4.4f \nloss_train:%.8f' % (
                    itr, iterations, time.time() - start_time, loss_train_value))
            if np.mod(itr, 200) == 0:
                rain_img_val_value, clean_img_val_value, y_val_value = sess.run(
                    [rain_img_val, clean_img_val, y_val])
                save_images_(rain_img_val_value, [1, 1], './{}/rain_img_val{:04d}.png'.format(save_sample_path, itr))
                save_images_(clean_img_val_value, [1, 1], './{}/clean_img_val{:04d}.png'.format(save_sample_path, itr))
                save_images_(y_val_value, [1, 1], './{}/derain_val{:04d}.png'.format(save_sample_path, itr))
                writer.add_summary(summary_str_loss, itr)
                saver.save(sess, os.path.join(save_model_path, model_name), global_step=itr)
        coord.request_stop()
        sess.close()
        var_list = tf.trainable_variables()
        print("Total parameters' number: %d"
              % (np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))
        endtime = datetime.datetime.now()
        print("结束时间: ", str(endtime).split('.')[0])
        print(u'相差：%s' % (endtime - starttime))