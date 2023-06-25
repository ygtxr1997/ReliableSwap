import tensorflow._api.v2.compat.v1 as tf


def average_gradients(tower_grads):
    """ Calculate the average gradient for each shared variable across
    all towers.

    **Note** that this function provides a synchronization point
    across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples.
            The outer list is over individual gradients. The inner list is
            over the gradient calculation for each tower.

    Returns:
        List of pairs of (gradient, variable) where the gradient
            has been averaged across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        v = grad_and_vars[0][1]
        # sum
        grad = tf.add_n([x[0] for x in grad_and_vars])
        # average
        grad = grad / float(len(tower_grads))
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def process_gradients(grads, gradient_processor):
    """Process gradients with func.

    Args:
        grads (list): List of (grad, var).
        gradient_processor (function): Function to processs gradients.

    Returns:
        list: grads.

    """
    if gradient_processor is not None:
        grads = [grad for grad in grads if grad[0] is not None]
        grads = [gradient_processor(grad) for grad in grads]
    return grads
