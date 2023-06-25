import tensorflow._api.v2.compat.v1 as tf


__all__ = ['get_sess']


def get_sess(sess=None):
    """Get default session if sess is None.

    Args:
        sess: Valid sess or None.

    Returns:
        Valid sess or get default sess.

    """
    if sess is None:
        sess = tf.get_default_session()
    assert sess, 'sess should not be None.'
    return sess
