#!/usr/bin/env python

import theano
import theano.tensor as tensor
import numpy as np

profile = False


def sgd(tparams, grads, inp, cost, lr0):
    """Stochastic Gradient Descent optimizer."""
    # define the update step rule
    updates = []
    for p, g in zip(tparams.values(), grads):
        updates.append((p, p - lr0 * g))

    return updates


def rmsprop(tparams, grads, inp, cost, lr0=0.01, rho=0.95, eps=1e-6):
    """RMSProp optimizer."""
    # define the update step rule
    updates = []
    one = tensor.constant(1.)
    for p, g in zip(tparams.values(), grads):
        # Accumulate gradient squares
        v = theano.shared(np.zeros(p.get_value().shape).astype('float32'))
        # rho * past + (1 - rho) * current
        v_new = (rho * v) + (one - rho) * g**2
        updates.append((v, v_new))
        updates.append((p, p - (lr0 * g / tensor.sqrt(v_new + eps))))

    return updates


def adam(tparams, grads, inp, cost, lr0=0.0001, b1=0.9, b2=0.999, eps=1e-8):
    """ADAM optimizer."""
    i = theano.shared(np.float32(0.))
    i_t = i + 1.

    # Running learning-rate
    lr_t = lr0 * (tensor.sqrt(1. - b2**i_t) / (1. - b1**i_t))

    updates = []

    for p, g in zip(tparams.values(), grads):
        m = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_mu')
        v = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_var')

        m_t = (b1 * m) + ((1. - b1) * g)
        v_t = (b2 * v) + ((1. - b2) * g**2)
        p_t = p - (lr_t * (m_t / (tensor.sqrt(v_t) + eps)))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    return updates


def adam_multi(tparams_list, grads_list, lr0_list, b1=0.9, b2=0.999, eps=1e-8):
    """ADAM optimizer."""
    i = theano.shared(np.float32(0.))
    i_t = i + 1.

    # Running learning-rate
    lr_t_list = []
    for lr0 in lr0_list:
        lr_t_list.append(lr0 * (tensor.sqrt(1. - b2**i_t) / (1. - b1**i_t)))

    updates = []
    for index, p_g in enumerate(zip(tparams_list, grads_list)):
        tparams, grads = p_g
        lr_t = lr_t_list[index]
        for p, g in zip(tparams.values(), grads):
            m = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_mu')
            v = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_var')

            m_t = (b1 * m) + ((1. - b1) * g)
            v_t = (b2 * v) + ((1. - b2) * g ** 2)
            p_t = p - (lr_t * (m_t / (tensor.sqrt(v_t) + eps)))
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))

    updates.append((i, i_t))
    return updates


def adam_warm(tparams, grads, warmup_steps, decay_steps, lr0, b1=0.9, b2=0.999, eps=1e-8):
    """ADAM optimizer."""
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    decay_steps = np.float32(decay_steps)
    warmup_steps = np.float32(warmup_steps)

    # Running learning-rate
    # warm_up
    lr = tensor.switch(
        i_t <= warmup_steps,
        lr0 * (i / warmup_steps),
        lr0 * (1. - tensor.minimum(i_t, decay_steps) / decay_steps)
    )

    lr_t = lr * (tensor.sqrt(1. - b2**i_t) / (1. - b1**i_t))

    updates = []

    for p, g in zip(tparams.values(), grads):
        m = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_mu')
        v = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_var')

        m_t = (b1 * m) + ((1. - b1) * g)
        v_t = (b2 * v) + ((1. - b2) * g**2)
        p_t = p - (lr_t * (m_t / (tensor.sqrt(v_t) + eps)))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    return updates


def amsgrad(tparams, grads, inp, cost, lr0=0.0001, b1=0.9, b2=0.999, eps=1e-8):
    """ADAM optimizer."""
    i = theano.shared(np.float32(0.))
    i_t = i + 1.

    # Running learning-rate
    lr_t = lr0 * (tensor.sqrt(1. - b2**i_t) / (1. - b1**i_t))

    updates = []

    for p, g in zip(tparams.values(), grads):
        m = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_mu')
        v = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_var')
        vhat = theano.shared(np.zeros(p.get_value().shape).astype('float32'), p.name + '_vhat')
        m_t = (b1 * m) + ((1. - b1) * g)
        v_t = (b2 * v) + ((1. - b2) * g**2)
        vhat_t = tensor.maximum(vhat, v_t)
        p_t = p - (lr_t * (m_t / (tensor.sqrt(vhat_t) + eps)))
        updates.append((vhat, vhat_t))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    return updates