""" layers """
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import control_flow_ops


def get_variable(name, shape, initializer, weight_decay=0.0, dtype='float',
                 trainable=True, freeze_weights=False):
    """Simple wrapper for get_variable.

    """
    if weight_decay > 0.:
        regularizer = tf.keras.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    variable = tf.get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, trainable=trainable, collections=collections)
    if freeze_weights:
        variable = tf.stop_gradient(variable)
    return variable


def int2list(shape):
    if isinstance(shape, int):
        shape = [shape, shape]
    return shape


def NHWC_to_NCHW(x, name='NHWC_to_NCHW'):
    """Convert data format from NHWC to NCHW.

    """
    x = tf.transpose(x, [0, 3, 1, 2])
    return x


def NCHW_to_NHWC(x, name='NCHW_to_NHWC'):
    """Convert data format from NCHW to NHWC.

    """
    x = tf.transpose(x, [0, 2, 3, 1])
    return x


def NHWC_to_HWNC(x, name='NHWC_to_HWNC'):
    """Convert data format from NHWC to HWNC, may be used for re-indexing.

    """
    x = tf.transpose(x, [1, 2, 0, 3])
    return x


def HWNC_to_NHWC(x, name='HWNC_to_NHWC'):
    """Convert data format from HWNC to NHWC, may be used for re-indexing.

    """
    x = tf.transpose(x, [2, 0, 1, 3])
    return x


def apply_activation(x, relu, activation_fn, name='activation'):
    """Wrapper for apply activation.

    **Note** activation_fn has higher execution level.
    """
    with tf.variable_scope(name):
        if activation_fn is not None:
            return activation_fn(x)
        elif relu:
            return tf.nn.relu(x)
        else:
            return x


def moving_average_var(x, decay=0.99, initial_value=0.,
                       name='moving_average_var'):
    """Moving_average_var.

    """
    moving_x = get_variable(
        name, x.get_shape(),
        initializer=tf.constant_initializer(initial_value), trainable=False)
    with tf.control_dependencies([assign_moving_average(moving_x, x, decay)]):
        moving_x = tf.identity(moving_x)
    return moving_x


def depthwise_conv2d(x, ksize=3, stride=1, decay=0.0, biased=True, relu=False,
         activation_fn=None, w_init=tf.keras.initializers.glorot_normal,
         padding='SAME', name='depthwise_conv2d'):
    """Simple wrapper for convolution layer.
    Padding can be 'SAME', 'VALID', 'REFLECT', 'SYMMETRIC'
    """
    ksize = int2list(ksize)
    stride = int2list(stride)
    filters_in = x.get_shape()[-1]
    with tf.variable_scope(name):
        if padding == 'SYMMETRIC' or padding == 'REFELECT':
            x = tf.pad(x, [[0,0], [int((ksize[0]-1)/2), int((ksize[0]-1)/2)], [int((ksize[1]-1)/2), int((ksize[1]-1)/2)], [0,0]], mode=padding)
            padding = 'VALID'
        weights = get_variable(
            'weights', [ksize[0], ksize[1], filters_in, 1],
            initializer=w_init, weight_decay=decay)
        conv_data = tf.nn.depthwise_conv2d(
            x, weights, strides=[1, stride[0], stride[1], 1], padding=padding)
        if biased:
            biases = get_variable(
                'biases', [filters_in], initializer=tf.zeros_initializer(),
                weight_decay=decay)
            conv_data = conv_data + biases
        conv_data = apply_activation(conv_data, relu, activation_fn)
    return conv_data


def max_pool(x, ksize=2, stride=2, padding='SAME', name='max_pool'):
    """Max pooling wrapper.

    """
    k = int2list(ksize)
    s = int2list(stride)
    with tf.variable_scope(name):
        return tf.nn.max_pool(
            x, [1, k[0], k[1], 1], [1, s[0], s[1], 1], padding)


def avg_pool(x, ksize=2, stride=2, padding='SAME', name='avg_pool'):
    """Average pooling wrapper.

    """
    k = int2list(ksize)
    s = int2list(stride)
    with tf.variable_scope(name):
        return tf.nn.avg_pool(
            x, [1, k[0], k[1], 1], [1, s[0], s[1], 1], padding)


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


def bilinear_upsample(x, scale=2):
    """Bilinear upsample.

    Caffe bilinear upsample forked from
    https://github.com/ppwwyyxx/tensorpack
    Deterministic bilinearly-upsample the input images.

    Args:
        x (tf.Tensor): a NHWC tensor
        scale (int): the upsample factor

    Returns:
        tf.Tensor: a NHWC tensor.

    """
    def bilinear_conv_filler(s):
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret

    inp_shape = x.get_shape().as_list()
    # inp_shape = tf.shape(x)
    ch = inp_shape[3]
    assert ch is not None
    filter_shape = 2 * scale
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))
    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    # pad = min(scale - 1, inp_shape[1])
    pad = scale - 1
    x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='SYMMETRIC')
    # if inp_shape[1] < scale:
        # # may cause problem?
        # pad = scale - 1 - inp_shape[1]
        # x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]],
                   # mode='CONSTANT')
    out_shape = tf.shape(x) * tf.constant([1, scale, scale, 1], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, scale, scale, 1], 'SAME')
    edge = scale * (scale - 1)
    deconv = deconv[:, edge:-edge, edge:-edge, :]
    if inp_shape[1]:
        inp_shape[1] *= scale
    if inp_shape[2]:
        inp_shape[2] *= scale
    deconv.set_shape(inp_shape)
    return deconv


def transformer(U, theta, out_size=None, name='SpatialTransformer'):
    """Spatial Transformer Layer.

    Forked from tensorflow/models transformer.

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(
                    tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(
                tf.ones(shape=tf.stack([height, 1])),
                tf.transpose(
                    tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed,
                tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        if out_size is None:
            out_size = tf.shape(U)[1:3]
        output = _transform(theta, U, out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer

    Args:
        U (float):
        tensor of inputs [num_batch,height,width,num_channels]
        thetas (float):
        a set of transformations for each input [num_batch,num_transforms,6]
        out_size (int):
        the size of the output [out_height,out_width]

    Returns: float
        Tensor of size
        [num_batch*num_transforms,out_height,out_width,num_channels]

    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)


def pixel_flow(x, offset, interpolation='bilinear', name='pixel_flow'):
    """pixel_flow: an operation to reorder pixels according to offsets.

    Args:
        x (tf.Tensor): NHWC
        offset (tf.Tensor): NHW2, 2 indicates (h, w) coordinates offset
        interpolation: bilinear, softmax
        name: name of module

    References
    ----------
    [1] Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
    [2] https://github.com/ppwwyyxx/tensorpack

    """
    def reindex(x, offset):
        offset = tf.cast(offset, tf.int32)
        xs = tf.shape(x)
        ofs = tf.shape(offset)
        n_add = tf.tile(tf.reshape(
            tf.range(xs[0]), [xs[0], 1, 1, 1]), [1, xs[1], xs[2], 1])
        h_add = tf.tile(tf.reshape(
            tf.range(xs[1]), [1, xs[1], 1, 1]), [xs[0], 1, xs[2], 1])
        w_add = tf.tile(tf.reshape(
            tf.range(xs[2]), [1, 1, xs[2], 1]), [xs[0], xs[1], 1, 1])
        coords = offset + tf.concat([h_add, w_add], axis=3)
        coords = tf.clip_by_value(coords, 0, [xs[1] - 1, xs[2] - 1])
        coords = tf.concat([n_add, coords], axis=3)
        # TODO(Jiahui): gather nd is also too slow.
        sampled = tf.gather_nd(x, coords)
        return sampled

    def reindex_slow(x, offset):
        offset = tf.cast(offset, tf.int32)
        xs = tf.shape(x)
        ofs = tf.shape(offset)
        n_add = tf.tile(tf.reshape(
            tf.range(xs[0]), [xs[0], 1, 1, 1]), [1, xs[1], xs[2], 1])
        h_add = tf.tile(tf.reshape(
            tf.range(xs[1]), [1, xs[1], 1, 1]), [xs[0], 1, xs[2], 1])
        w_add = tf.tile(tf.reshape(
            tf.range(xs[2]), [1, 1, xs[2], 1]), [xs[0], xs[1], 1, 1])
        coords = offset + tf.concat([h_add, w_add], axis=3)
        coords = tf.clip_by_value(coords, 0, [xs[1] - 1, xs[2] - 1])
        coords = tf.concat([n_add, coords], axis=3)
        x = tf.reshape(x, [-1, xs[3]])
        coords_flat = tf.reshape(coords, [-1, 3])  # (batch, height, width)
        coords_flat = (coords_flat[:, 0] * xs[0] * xs[1] +
                       coords_flat[:, 1] * xs[1] + coords_flat[:, 2])
        sampled = tf.gather(x, coords_flat)
        sampled = tf.reshape(sampled, xs)
        return sampled

    with tf.variable_scope(name):
        assert x.get_shape().ndims == 4 and offset.get_shape().ndims == 4

        l = tf.floor(offset)  # lower
        u = l + 1  # upper
        diff = offset - l
        neg_diff = 1.0 - diff

        lh, lw = tf.split(l, 2, axis=3)
        uh, uw = tf.split(u, 2, axis=3)

        lhuw = tf.concat([lh, uw], axis=3)
        uhlw = tf.concat([uh, lw], axis=3)

        diffh, diffw = tf.split(diff, 2, axis=3)
        neg_diffh, neg_diffw = tf.split(neg_diff, 2, axis=3)
        if interpolation == 'bilinear':
            pass
        elif interpolation == 'softmax':
            scale = 10.
            diffh = tf.sigmoid(scale*(diffh-0.5))
            diffw = tf.sigmoid(scale*(diffw-0.5))
            neg_diffh = tf.sigmoid(scale*(neg_diffh-0.5))
            neg_diffw = tf.sigmoid(scale*(neg_diffw-0.5))
        else:
            assert NotImplementedError(
                "interpolation method: {} is not implemented."
                .format(interpolation))

        sampled = tf.add_n(
            [reindex(x, l) * neg_diffw * neg_diffh,
             reindex(x, u) * diffw * diffh,
             reindex(x, lhuw) * neg_diffh * diffw,
             reindex(x, uhlw) * diffh * neg_diffw],
            name='sampled')
        return sampled


def concatenated_relu(x, name='concatenated_relu'):
    """Concatenated relu wrapper.

    """
    with tf.variable_scope(name):
        pos = tf.nn.relu(x)
        neg = tf.nn.relu(-x)
        return tf.concat([pos, neg], -1)


def scaled_elu(x, name='scaled_elu'):
    """Scaled elu wrapper.

    """
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def flatten(x, name='flatten'):
    """Flatten wrapper.

    """
    with tf.variable_scope(name):
        return tf.keras.layers.flatten(x)
