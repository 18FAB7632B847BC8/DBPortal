#!/usr/bin/env python

import theano.tensor as tensor
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy
import theano

from utils import norm_weight, ortho_weight
import numbers

# Shorthands for activations
# linear = lambda x: x

def linear(x): return x
sigmoid = tensor.nnet.sigmoid
tanh = tensor.tanh
relu = tensor.nnet.relu
softmax = tensor.nnet.softmax
softplus = tensor.nnet.softplus
default_rng = RandomStreams(1235)


class BatchNormLayer(object):
    def __init__(self, axes='auto', input_shape=None, epsilon=1e-4, alpha=0.1,
                 beta=0., gamma=1., mean=0., inv_std=1.,
                 target='dev0', name='bn'):
        self.input_shape = input_shape
        self.name = name
        self.target = target
        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, numbers.Integral):
            axes = (axes,)

        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        self.tparams = OrderedDict()
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is not None:
            beta = (numpy.ones(shape) * beta).astype('float32')
        if gamma is not None:
            gamma = (numpy.ones(shape) * gamma).astype('float32')
        mean = (numpy.ones(shape) * mean).astype('float32')
        inv_std = (numpy.ones(shape) * inv_std).astype('float32')

        self.tparams['beta'] = theano.shared(beta, name=self.name + '_beta')
        self.tparams['gamma'] = theano.shared(gamma, name=self.name + '_gamma')
        self.tparams['mean'] = theano.shared(mean, name=self.name + '_mean')
        self.tparams['inv_std'] = theano.shared(inv_std, name=self.name + '_std')

    def get_output(self, input_x, use_averages=False, update_averages=True):
        input_mean = input_x.mean(self.axes)
        input_inv_std = tensor.inv(tensor.sqrt(input_x.var(self.axes) + self.epsilon))

        assert use_averages != update_averages
        if use_averages:
            mean = self.tparams['mean']
            inv_std = self.tparams['inv_std']
        else:
            mean = input_mean
            inv_std = input_inv_std

        if update_averages:
            running_mean = theano.clone(self.tparams['mean'], share_inputs=False)
            running_inv_std = theano.clone(self.tparams['inv_std'], share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        param_axes = iter(range(input_x.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input_x.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = self.tparams['beta'].dimshuffle(pattern)
        gamma = self.tparams['gamma'].dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)
        # normalize
        normalized = (input_x - mean) * (gamma * inv_std) + beta
        return normalized


class PointerNetwork(object):
    def __init__(self, n_in, n_ctx, n_hid, scale="xavier", name="PointerNetwork"):
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        params['W'] = norm_weight(n_in, n_hid, scale=scale)
        params['b1'] = numpy.zeros((n_hid,)).astype('float32')

        params['U'] = norm_weight(n_ctx, n_hid, scale=scale)
        params['b2'] = numpy.zeros((n_hid,)).astype('float32')

        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, ctx_q, ctx_h):
        # ctx_q: n_samples, n_q, n_in
        # ctx_h: n_samples, n_h, n_ctx
        preact = tensor.dot(ctx_q, self.tparams['W']) + self.tparams['b1']  # n_sampels, n_q, n_hid
        ctx = tensor.dot(ctx_h, self.tparams['U']) + self.tparams['b2'] # n_samples, n_h, n_hid
        out = tensor.batched_dot(preact, ctx.dimshuffle(0, 2, 1))  # n_samples, n_q, n_h

        return out
