import tensorflow._api.v2.compat.v1 as tf

from .summary_ops import scalar_summary


def gan_log_loss(pos, neg, name='gan_log_loss'):
    """
    log loss function for GANs.
    - Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
    """
    with tf.variable_scope(name):
        # generative model G
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.ones_like(neg)))
        # discriminative model D
        d_loss_pos = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos, labels=tf.ones_like(pos)))
        d_loss_neg = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.zeros_like(neg)))
        pos_acc = tf.reduce_mean(tf.sigmoid(pos))
        neg_acc = tf.reduce_mean(tf.sigmoid(neg))
        scalar_summary('d_scores/pos_mean', pos_acc)
        scalar_summary('d_scores/neg_mean', neg_acc)
        # loss
        d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg)
        scalar_summary('losses/d_loss', d_loss)
        scalar_summary('losses/g_loss', g_loss)
    return g_loss, d_loss


def gan_ls_loss(pos, neg, value=1., name='gan_ls_loss'):
    """
    gan with least-square loss
    """
    with tf.variable_scope(name):
        l2_pos = tf.reduce_mean(tf.squared_difference(pos, value))
        l2_neg = tf.reduce_mean(tf.square(neg))
        scalar_summary('pos_l2_avg', l2_pos)
        scalar_summary('neg_l2_avg', l2_neg)
        d_loss = tf.add(.5 * l2_pos, .5 * l2_neg)
        g_loss = tf.reduce_mean(tf.squared_difference(neg, value))
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
    return g_loss, d_loss


def gan_hinge_loss(pos, neg, value=1., name='gan_hinge_loss'):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    with tf.variable_scope(name):
        hinge_pos = tf.reduce_mean(tf.nn.relu(1-pos))
        hinge_neg = tf.reduce_mean(tf.nn.relu(1+neg))
        scalar_summary('pos_hinge_avg', hinge_pos)
        scalar_summary('neg_hinge_avg', hinge_neg)
        d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
        g_loss = -tf.reduce_mean(neg)
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
    return g_loss, d_loss


def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.

    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
        scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss


def random_interpolates(x, y, alpha=None, dtype=tf.float32):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1], dtype=dtype)
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs

    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


def kernel_spectral_norm(kernel, iteration=1, name='kernel_sn'):
    # spectral_norm
    def l2_norm(input_x, epsilon=1e-12):
        input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
        return input_x_norm
    with tf.variable_scope(name) as scope:
        w_shape = kernel.get_shape().as_list()
        w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
        u = tf.get_variable(
            'u', shape=[1, w_shape[-1]],
            initializer=tf.truncated_normal_initializer(),
            trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1

        u_hat, v_hat,_ = power_iteration(u, iteration)
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


class Conv2DSepctralNorm(tf.layers.Conv2D):
    def build(self, input_shape):
        super(Conv2DSepctralNorm, self).build(input_shape)
        self.kernel = kernel_spectral_norm(self.kernel)


def conv2d_spectral_norm(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    layer = Conv2DSepctralNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)
