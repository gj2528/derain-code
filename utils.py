import os 
import torch
import tensorflow as tf
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def save_images_(images, size, image_path):
    images = images * 255.0
    images[np.where(images < 0)] = 0.
    images[np.where(images > 255)] = 255.
    images = images.astype(np.uint8)
    return scipy.misc.imsave(image_path, merge(images, size))

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=tf.float32))
    return numerator / denominator


def _tf_fspecial_gauss(size, sigma):
    """ Function to mimic the 'fspecial' gaussian MATLAB functino
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma * 2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image

def loss_ssim(img1, img2, batchsize, c_dims):
    ssim_value_sum = 0
    for i in range(batchsize):
        for j in range(c_dims):
            img1_tmp = img1[i, :, :, j]
            img1_tmp = image_to_4d(img1_tmp)
            img2_tmp = img2[i, :, :, j]
            img2_tmp = image_to_4d(img2_tmp)
            ssim_value_tmp = tf_ssim(img1_tmp, img2_tmp)
            ssim_value_sum += ssim_value_tmp
    ssim_value_ave = ssim_value_sum / (batchsize * c_dims)
    return log10(1.0 / (ssim_value_ave + 1e-4))

# def edge_compute(x):   #tensorflow
#     x_diffx = tf.abs(x[:,1:,:] - x[:,:-1,:])
#     x_diffy = tf.abs(x[1:,:,:] - x[:-1,:,:])
#     in_shape = x.shape
#     y = np.zeros(shape=in_shape, dtype=float)
#     y[:, 1:, :] = x_diffx
#     y[:, :-1, :] += x_diffx
#     y[1:, :, :] += x_diffy
#     y[:-1, :, :] += x_diffy
#     y = tf.reduce_sum(y, 0, keepdim=True) / 3
#     y /= 4
#     return y

def edge_compute(x):   #tensorflow
    print(x,'x')
    x_diffx = np.abs(x[:, 1:, :] - x[:, :-1, :])
    print(x_diffx,'x_diffx')
    x_diffy = np.abs(x[1:, :, :] - x[:-1, :, :])
    print(x_diffy, 'x_diffy')
    in_shape = x.shape
    print(in_shape,'in_shape')
    H, W, C = in_shape
    y = np.zeros(shape=in_shape, dtype=float)
    y[:, 1:, :] = x_diffx
    print(y, 'y[:, i+1, :]')
    y[:, :-1, :] += x_diffx
    print(y,'y[:, i, :]')
    y[1:, :, :] += x_diffy
    print(y,'y[1:, :, :]')
    y[:-1, :, :] += x_diffy
    print(y,'y[:-1, :, :]')
    y = np.sum(y, axis=-1, keepdims=True) / 3
    y /= 4
    return y

# def edge_compute(x):
#     x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
#     x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

#     y = x.new(x.size())
#     y.fill_(0)
#     y[:,:,1:] += x_diffx
#     y[:,:,:-1] += x_diffx
#     y[:,1:,:] += x_diffy
#     y[:,:-1,:] += x_diffy
#     y = torch.sum(y,0,keepdim=True)/3
#     y /= 4
#     return y

# def edge_compute(x):   #pytorch
#     print(x[:,:,:])
#     # print(x[:,:,1:],'x[:,:,1:]')
#     # print(x[:,:,1:].shape)
#     # print(x[:,:,:-1],'x[:,:,:-1]')
#     # print(x[:,:,:-1].shape)
#     x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
#     print(x_diffx,'x_diffy')
#     print(x_diffx.shape)
#     # print(x[:,1:,:],'x[:,1:,:]')
#     # print(x[:,1:,:].shape)
#     # print(x[:,:-1,:],'x[:,:-1,:]')
#     # print(x[:,:-1,:].shape)
#     x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])
#     print(x_diffy,'x_diffy')
#     print(x_diffy.shape)
#
#     y = x.new(x.size())
#     y.fill_(0)
#     y[:,:,1:] += x_diffx
#     # print(y[:,:,1:],'y[:,:,1:]')
#     # print(y[:,:,1:].shape)
#     y[:,:,:-1] += x_diffx
#     # print(y[:,:,:-1],'y[:,:,:-1]')
#     # print(y[:,:,:-1].shape)
#     y[:,1:,:] += x_diffy
#     # print(y[:,1:,:],'y[:,1:,:]')
#     # print(y[:,1:,:].shape)
#     y[:,:-1,:] += x_diffy
#     # print(y[:,:-1,:],'y[:,:-1,:]')
#     # print(y[:,:-1,:].shape)
#     # print(y[:,:,:],'y[:,:,:]')
#     # print(y[:,:,:].shape)
#     # print(torch.sum(y,0,keepdim=True),'torch.sum(y,0,keepdim=True)')
#     # print(torch.sum(y,0,keepdim=True).shape)
#     y = torch.sum(y,0,keepdim=True)/3
#     # print(y[:, :, :], 'y[:,:,:]')
#     # print(y[:, :, :].shape)
#     y /= 4
#     # print(y[:, :, :], 'y[:,:,:]')
#     # print(y[:, :, :].shape)
#     return y