
import tensorflow as tf
from tensorflow.python import keras
from tqdm import tqdm


# -----------------------------------------------------------------------
# See https://github.com/openai/weightnorm/tree/master/keras
# See https://github.com/krasserm/weightnorm/tree/master/keras_2
# See https://arxiv.org/abs/1602.07868
# -----------------------------------------------------------------------


class AdamWithWeightnorm(keras.optimizers.Adam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [keras.backend.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay *
                         keras.backend.cast(self.iterations, keras.backend.floatx())))

        t = keras.backend.cast(self.iterations + 1, keras.backend.floatx())
        lr_t = lr * keras.backend.sqrt(1. - keras.backend.pow(
            self.beta_2, t)) / (1. - keras.backend.pow(self.beta_1, t))
        shapes = [keras.backend.int_shape(p) for p in params]
        ms = [keras.backend.zeros(shape) for shape in shapes]
        vs = [keras.backend.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Adam
            ps = keras.backend.int_shape(p)
            if len(ps) > 1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(
                    p, g)

                # Adam containers for the 'g' parameter
                V_scaler_shape = keras.backend.int_shape(V_scaler)
                m_g = keras.backend.zeros(V_scaler_shape)
                v_g = keras.backend.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1. - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1. - self.beta_2) * \
                    keras.backend.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / \
                    (keras.backend.sqrt(v_g_t) + self.epsilon)
                self.updates.append(keras.backend.update(m_g, m_g_t))
                self.updates.append(keras.backend.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * \
                    keras.backend.square(grad_V)
                new_V_param = V - lr_t * m_t / \
                    (keras.backend.sqrt(v_t) + self.epsilon)
                self.updates.append(keras.backend.update(m, m_t))
                self.updates.append(keras.backend.update(v, v_t))

                # if there are constraints we apply them to V, not W
                if getattr(p, 'constraint', None) is not None:
                    new_V_param = p.constraint(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(
                    self.updates, new_V_param, new_g_param, p, V_scaler)

            else:  # do optimization normally
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * \
                    keras.backend.square(g)
                p_t = p - lr_t * m_t / (keras.backend.sqrt(v_t) + self.epsilon)

                self.updates.append(keras.backend.update(m, m_t))
                self.updates.append(keras.backend.update(v, v_t))

                new_p = p_t
                # apply constraints
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)
                self.updates.append(keras.backend.update(p, new_p))
        return self.updates


def get_weightnorm_params_and_grads(p, g):
    ps = keras.backend.int_shape(p)

    # construct weight scaler: V_scaler = g/||V||
    V_scaler_shape = (ps[-1],)  # assumes we're using tensorflow!
    # init to ones, so effective parameters don't change
    V_scaler = keras.backend.ones(V_scaler_shape)

    # get V parameters = ||V||/g * W
    norm_axes = [i for i in range(len(ps) - 1)]
    V = p / tf.reshape(V_scaler, [1] * len(norm_axes) + [-1])

    # split V_scaler into ||V|| and g parameters
    V_norm = tf.sqrt(tf.reduce_sum(tf.square(V), norm_axes))
    g_param = V_scaler * V_norm

    # get grad in V,g parameters
    grad_g = tf.reduce_sum(g * V, norm_axes) / V_norm
    grad_V = tf.reshape(V_scaler, [1] * len(norm_axes) + [-1]) * \
        (g - tf.reshape(grad_g / V_norm, [1] * len(norm_axes) + [-1]) * V)

    return V, V_norm, V_scaler, g_param, grad_g, grad_V


def add_weightnorm_param_updates(updates, new_V_param, new_g_param, W, V_scaler):
    ps = keras.backend.int_shape(new_V_param)
    norm_axes = [i for i in range(len(ps) - 1)]

    # update W and V_scaler
    new_V_norm = tf.sqrt(tf.reduce_sum(tf.square(new_V_param), norm_axes))
    new_V_scaler = new_g_param / new_V_norm
    new_W = tf.reshape(new_V_scaler, [1] * len(norm_axes) + [-1]) * new_V_param
    updates.append(keras.backend.update(W, new_W))
    updates.append(keras.backend.update(V_scaler, new_V_scaler))

