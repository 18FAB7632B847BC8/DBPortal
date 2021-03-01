#!/usr/bin/env python

import theano.tensor as tensor
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy
import math
import theano

from utils import norm_weight
import numbers


TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


def linear(x): return x
sigmoid = tensor.nnet.sigmoid
tanh = tensor.tanh
relu = tensor.nnet.relu
softmax = tensor.nnet.softmax
softplus = tensor.nnet.softplus
default_rng = RandomStreams(1235)


def all_cosine(v1, v2):
    # v: n_batch_size * dim
    numerator = tensor.dot(v1, v2.T)  # n_batch_size * n_batch_size
    denominator = tensor.sqrt(tensor.dot((v1 ** 2).sum(1)[:, None],
                                         (v2 ** 2).sum(1)[None, :]))  # n_batch_size * n_batch_size

    return numerator / denominator


def cosine(v1, v2):
    # v: n_batch_size * dim
    numerator = (v1 * v2).sum(1)  # n_samples
    denominator = tensor.sqrt((v1 ** 2).sum(1) * (v2 ** 2).sum(1))  # n_samples

    return numerator / denominator


def norm(v):
    # v: n_batch_size * dim
    denominator = tensor.sqrt((v ** 2).sum(1))  # n_samples

    return v / denominator[:, None]


def gelu(x):
    return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tensor.pow(x, 3))))


def dropout(x, rng, p, in_train):
    if p <= 0.:
        # No dropout, return x intact
        return x

    success = 1. - p
    return tensor.switch(in_train,  # If True, next one else last one
                         x * rng.binomial(x.shape, p=success, dtype=x.dtype) / success,
                         x)


def layer_norm(x, b, s, eps=1e-12):
    if x.ndim == 3:
        output = (x - x.mean(2)[:, :, None]) / tensor.sqrt((x.var(2)[:, :, None] + eps))
        output = s[None, None, :] * output + b[None, None, :]
    else:
        output = (x - x.mean(1)[:, None]) / tensor.sqrt(x.var(1)[:, None] + eps)
        output = s[None, :] * output + b[None, :]
    return output


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


class LayerNormLayer(object):
    def __init__(self, n_out, name='ln', epsilon=1e-6, scale_add=0.0, scale_mul=1.0):
        self.tparams = dict()
        self.name = name

        self.epsilon = epsilon
        gamma = scale_mul * numpy.ones((1 * n_out)).astype('float32')
        beta = scale_add * numpy.ones((1 * n_out)).astype('float32')
        self.tparams['gamma'] = theano.shared(gamma, name=self.name + '_gamma')
        self.tparams['beta'] = theano.shared(beta, name=self.name + '_beta')

    def get_output(self, input_x):
        pattern = ['x'] * input_x.ndim
        pattern[-1] = 0
        mean = input_x.mean(-1, keepdims=True)
        std = input_x.std(-1, keepdims=True)
        beta = self.tparams['beta'].dimshuffle(pattern)
        gamma = self.tparams['gamma'].dimshuffle(pattern)
        return gamma * (input_x - mean) / (std + self.epsilon) + beta


class EmbeddingLayer(object):
    def __init__(self, n_words_src, embedding_dim, input_embdding=None, scale=0.01, name=''):
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        self.n_words_src = n_words_src
        self.embedding_dim = embedding_dim
        if input_embdding is None:
            params['Wemb'] = norm_weight(self.n_words_src, self.embedding_dim, scale=scale)
        else:
            assert type(input_embdding) == numpy.ndarray
            assert input_embdding.shape[0] == n_words_src and input_embdding.shape[1] == embedding_dim
            params['Wemb'] = input_embdding
        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, input_x):
        return self.tparams['Wemb'][input_x.flatten()]


class PositionEmbeddingLayer(object):
    def __init__(self, n_pos, embedding_dim, trainable=True, initializer='uniform', name='pos_emb'):
        self.tparams = OrderedDict()
        self.name = name
        self.n_pos = n_pos
        self.embedding_dim = embedding_dim
        self.trainable = trainable
        self.initializer = initializer

        if self.initializer == 'pos':
            pos_emb = numpy.array([
                [pos / numpy.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
                if pos != 0 else numpy.zeros(embedding_dim)
                for pos in range(n_pos)
            ])
            pos_emb[1:, 0::2] = numpy.sin(pos_emb[1:, 0::2])  # dim 2i
            pos_emb[1:, 1::2] = numpy.cos(pos_emb[1:, 1::2])  # dim 2i+1
            pos_emb = pos_emb.astype('float32')
        else:
            pos_emb = norm_weight(n_pos, embedding_dim, scale=0.02)
        self.tparams['Wemb'] = theano.shared(pos_emb, name=self.name + '_Wemb')

    def get_output(self, input_x):
        return self.tparams['Wemb'][input_x.flatten()]


def scaled_dot_attention(query, key, value, history_only, mask=None):
    dim = tensor.cast(query.shape[-1], dtype='float32')
    energy = tensor.batched_tensordot(query, key, axes=(2, 2))  # batch_size, n_timestep_q, n_timestep_k
    energy /= tensor.sqrt(dim)
    energy = tensor.exp(energy - energy.max(-1, keepdims=True))
    if history_only:
        energy += 1e-12
        query_len = query.shape[1]
        key_len = key.shape[1]
        indices = tensor.arange(0, key_len)[None, :]
        upper = tensor.arange(0, query_len)[:, None]
        energy *= tensor.cast(indices <= upper, 'float32')[None, :, :]
        if mask is not None:
            energy = energy * mask[:, None, :]
        alpha = energy / (energy.sum(-1, keepdims=True))  # batch_size, n_timestep_q, n_timestep_k
    else:
        if mask is not None:
            energy = energy * mask[:, None, :]
        alpha = energy / (energy.sum(-1, keepdims=True) + 1e-12)  # batch_size, n_timestep_q, n_timestep_k
    ctx = tensor.batched_dot(alpha, value)  # batch_size, n_timestep_q, n_dim

    return ctx


class MultiHeadAttentionLayer(object):
    def __init__(self, n_in_q, n_in_k, n_in_v, n_out, n_head, history_only, scale=0.01, ortho=True,
                 name='MultiHeadAttention'):
        self.n_in_q = n_in_q
        self.n_in_k = n_in_k
        self.n_in_v = n_in_v
        self.n_out = n_out
        self.n_head = n_head
        self.name = name
        self.history_only = history_only
        self.tparams = OrderedDict()
        params = dict()

        params['Wq'] = norm_weight(n_in_q, n_out, scale=scale, ortho=ortho)
        params['bq'] = numpy.zeros((n_out,)).astype('float32')
        params['Wk'] = norm_weight(n_in_k, n_out, scale=scale, ortho=ortho)
        params['bk'] = numpy.zeros((n_out,)).astype('float32')
        params['Wv'] = norm_weight(n_in_v, n_out, scale=scale, ortho=ortho)
        params['bv'] = numpy.zeros((n_out,)).astype('float32')
        params['Wo'] = norm_weight(n_out, n_out, scale=scale, ortho=ortho)
        params['bo'] = numpy.zeros((n_out,)).astype('float32')

        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, q, k, v, k_mask, activ='relu'):
        batch_size = q.shape[0]
        len_q = q.shape[1]
        len_k = k.shape[1]
        len_v = v.shape[1]
        preact_q = eval(activ)(tensor.dot(q, self.tparams['Wq']) + self.tparams['bq'])
        preact_k = eval(activ)(tensor.dot(k, self.tparams['Wk']) + self.tparams['bk'])
        preact_v = eval(activ)(tensor.dot(v, self.tparams['Wv']) + self.tparams['bv'])
        head_dim = self.n_out // self.n_head

        preact_q = preact_q.reshape([batch_size, len_q, self.n_head, head_dim])
        preact_k = preact_k.reshape([batch_size, len_k, self.n_head, head_dim])
        preact_v = preact_v.reshape([batch_size, len_v, self.n_head, head_dim])

        preact_q = preact_q.dimshuffle(0, 2, 1, 3)
        preact_k = preact_k.dimshuffle(0, 2, 1, 3)
        preact_v = preact_v.dimshuffle(0, 2, 1, 3)

        preact_q = preact_q.reshape([batch_size * self.n_head, len_q, head_dim])
        preact_k = preact_k.reshape([batch_size * self.n_head, len_k, head_dim])
        preact_v = preact_v.reshape([batch_size * self.n_head, len_v, head_dim])

        mask = tensor.tile(k_mask[:, None, :], [1, self.n_head, 1]).reshape([batch_size * self.n_head, len_k])
        ctx = scaled_dot_attention(preact_q, preact_k, preact_v, self.history_only, mask=mask)
        ctx = ctx.reshape([batch_size, self.n_head, len_q, head_dim])
        ctx = ctx.dimshuffle(0, 2, 1, 3)
        ctx = ctx.reshape([batch_size, len_q, self.n_out])

        ctx = eval(activ)(tensor.dot(ctx, self.tparams['Wo']) + self.tparams['bo'])

        return ctx


class FeedForwardLayer(object):
    def __init__(self, n_in, n_out, scale=0.01, ortho=True, name='', layer_nm=False):
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.layer_norm = layer_nm

        params['W'] = norm_weight(n_in, n_out, scale=scale, ortho=ortho)
        params['b'] = numpy.zeros((n_out,)).astype('float32')

        if self.layer_norm:
            scale_add = 0.0
            scale_mul = 1.0
            params['ln_b'] = scale_add * numpy.ones((1 * n_out)).astype('float32')
            params['ln_s'] = scale_mul * numpy.ones((1 * n_out)).astype('float32')

        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, input_x, activ='linear'):
        preact = tensor.dot(input_x, self.tparams['W']) + self.tparams['b']
        if self.layer_norm:
            preact = layer_norm(preact, self.tparams['ln_b'], self.tparams['ln_s'])

        return eval(activ)(preact)


class EmbeddingSimLayer(object):
    def __init__(self, n_out, name=""):
        self.n_out = n_out
        self.name = name

        self.tparmas = OrderedDict()
        self.tparmas['b'] = theano.shared(numpy.zeros((n_out,)).astype('float32'),
                                          name=self.name + '_b')

    def get_output(self, input_x, w_emb):
        # W_emb: n_vocabulary, n_emb
        # input_x: n_samples, n_timestep, n_emb
        preact = tensor.dot(input_x, w_emb.T) + self.tparmas['b']  # n_samples, n_timestep, n_vocabulary

        return preact


class AttentionPoolingLayer(object):
    def __init__(self, n_in, n_hid, name, scale=0.02):
        # n_in:      input dim (e.g. embedding dim in the case of NMT)
        # n_hid:      gru_dim   (e.g. 1000)
        # n_ctx:   2*gru_dim (e.g. 2000)
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        self.n_in = n_in
        self.n_hid = n_hid

        params['Ws'] = norm_weight(n_in, n_hid, scale=scale)
        params['bs'] = numpy.zeros((n_hid,)).astype('float32')
        params['Wq'] = norm_weight(n_in, n_hid, scale=scale)
        params['bq'] = numpy.zeros((n_hid,)).astype('float32')
        params['U_att'] = norm_weight(n_hid, 1, scale=scale)
        params['c_att'] = numpy.zeros((1,)).astype('float32')

        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, query, input_context, context_mask=None):
        assert input_context.ndim == 3, 'Context must be 3-d: #sample x #annotation x #dim'
        assert query.ndim == 2, 'query must be 2-d: #sample x #dim'
        pctx_ = tensor.dot(input_context, self.tparams['Ws']) + self.tparams['bs']
        # pctx_: #sample * #annotation * #ctx
        q_ = tensor.dot(query, self.tparams['Wq']) + self.tparams['bq']
        pctx__ = tanh(pctx_ + q_[:, None, :])
        alpha = tensor.dot(pctx__, self.tparams['U_att']) + self.tparams['c_att']
        # alpha: #sample * #annotaion * 1
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1]))
        # alpha: #sample * #annotaion
        alpha = tensor.exp(alpha - alpha.max(1, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask

        alpha = alpha / alpha.sum(1, keepdims=True)
        # alpha: #sample * #annotaion
        # input_context: #sample * #annotaion * #ctx
        ctx_ = (alpha[:, :, None] * input_context).sum(1)
        # ctx_: #sample * #ctx

        return ctx_


class PointerNetwork(object):
    def __init__(self, n_in, n_hid, scale=0.02, name="PointerNetwork"):
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        self.INF = 1e+10
        params['W'] = norm_weight(n_hid, n_in, scale=scale)
        params['b'] = numpy.zeros((n_in,)).astype('float32')

        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, ctx_q, ctx_h, ctx_mask=None):
        # ctx_q: n_samples, n_q, n_emb
        # ctx_h: n_samples, n_h, n_hid
        # ctx_mask: n_samples, n_h
        preact = tensor.dot(ctx_q, self.tparams['W']) + self.tparams['b']  # n_sampels, n_q, n_emb
        logit = tensor.batched_dot(preact, ctx_h.dimshuffle(0, 2, 1))  # n_samples, n_q, n_h

        if ctx_mask:
            logit = logit * ctx_mask[:, None, :] + (1. - ctx_mask)[:, None, :] * (-self.INF)

        return logit


class CombineLayer(object):
    def __init__(self, n_in, n_hid, scale=0.01, name="CombineLayer"):
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        self.hid_dim = n_hid
        params['W'] = norm_weight(n_in, n_hid, scale=scale)
        params['U'] = norm_weight(n_in, n_hid, scale=scale)
        params['b'] = numpy.zeros((n_hid,)).astype('float32')
        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, x1, x2, activ="linear"):
        assert x1.ndim == 2, x1.ndim
        assert x2.ndim == 3, x2.ndim
        # x1: [n_samples, n_in]
        # x2: [n_samples, n_dim1, n_in]
        preactx1 = tensor.dot(x1, self.tparams['W'])
        preactx2 = tensor.dot(x2, self.tparams['U'])
        preact = preactx1[:, None, :] + preactx2 + self.tparams['b']

        return eval(activ)(preact)


class SimpleCombineLayer(object):
    def __init__(self, n_in, n_hid, scale=0.01, name="SimpleCombineLayer"):
        params = dict()
        self.tparams = OrderedDict()
        self.name = name
        params['W'] = norm_weight(n_in, n_hid, scale=scale)
        params['U'] = norm_weight(n_in, n_hid, scale=scale)
        params['b'] = numpy.zeros((n_hid,)).astype('float32')
        for k, v in params.iteritems():
            self.tparams[k] = theano.shared(v, name=self.name + '_' + k)

    def get_output(self, x1, x2, activ="linear"):
        assert x1.ndim == x2.ndim, "%d,%d" % (x1.ndim, x2.ndim)
        # x1: [n_samples, n_dim1, n_in]
        # x2: [n_samples, n_dim1, n_in]
        preactx1 = tensor.dot(x1, self.tparams['W'])
        preactx2 = tensor.dot(x2, self.tparams['U'])
        preact = preactx1 + preactx2 + self.tparams['b']

        return eval(activ)(preact)
