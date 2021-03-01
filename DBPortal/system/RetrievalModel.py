#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from basemodel import BaseModel
from Bert import Bert
from layers_bert import AttentionPoolingLayer, FeedForwardLayer, dropout, all_cosine

import theano
from theano import tensor


class RetrievalModel(BaseModel):
    def __init__(self, config):
        super(RetrievalModel, self).__init__(config)
        self.config = config
        self.token_num = config.get('token_num', 21128)
        self.bert = None
        self.att_q = None
        self.pooler_q = None
        self.att_c = None
        self.pooler_c = None
        self.query_q = None
        self.query_c = None
        self.margin = config.get('margin', 0.3)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.dropout_rate_bert = config.get('dropout_rate_bert', 0.1)
        self.use_dropout_bert = theano.shared(numpy.float32(config.get('use_dropout_bert', True)))
        self.eps = 1e-12

    def set_dropout_bert(self, val):
        self.use_dropout_bert.set_value(numpy.float32(val))

    def init_model(self):
        self.bert = Bert(self.token_num, type_num=2, trng=self.trng, dropout_rate=self.dropout_rate_bert, name="bert")

        self.att_q = AttentionPoolingLayer(self.bert.hidden_num, self.bert.hidden_num, name="att_q", scale=0.01)
        self.pooler_q = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="pooler_q")
        self.att_c = AttentionPoolingLayer(self.bert.hidden_num, self.bert.hidden_num, name="att_c", scale=0.01)
        self.pooler_c = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="pooler_c")
        self.query_q = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="query_q")
        self.query_c = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="query_c")

        self.layers = [
            self.att_q, self.pooler_q, self.att_c, self.pooler_c, self.query_q,
            self.query_c,
        ]

        for layer in self.layers:
            if layer is None:
                raise KeyError("layer is None")
            for k, v in layer.tparams.iteritems():
                p_name = layer.name + '_' + k
                assert p_name not in self.tparams, "Duplicated parameter: %s" % p_name
                self.tparams[p_name] = v

        for k, v in self.bert.tparams.iteritems():
            assert k not in self.tparams, "Duplicated parameter: %s" % k
            self.tparams[k] = v

    def build_model(self):
        x_q = tensor.matrix('x_q', dtype='int64')  # n_samples, n_timestep_q
        x_q_mask = tensor.matrix('x_q_mask', dtype='float32')   # n_samples, n_timestep_q
        x_c = tensor.matrix('x_c', dtype='int64')   # n_samples, n_timestep_c
        x_c_mask = tensor.matrix('x_c_mask', dtype='float32')   # n_samples, n_timestep_c

        self.inputs['x_q'] = x_q
        self.inputs['x_q_mask'] = x_q_mask
        self.inputs['x_c'] = x_c
        self.inputs['x_c_mask'] = x_c_mask

        x_q_type = tensor.zeros_like(x_q)
        x_c_type = tensor.ones_like(x_c)

        encoded_q = self.bert.get_output(x_q, x_q_type, mask=x_q_mask,
                                         in_train=self.use_dropout_bert)  # n_samples, n_timestep_q, 786
        encoded_c = self.bert.get_output(x_c, x_c_type, mask=x_c_mask,
                                         in_train=self.use_dropout_bert)  # n_samples, n_timestep_c, 786

        encoded_q = dropout(encoded_q, self.trng, self.dropout_rate, in_train=self.use_dropout)
        encoded_c = dropout(encoded_c, self.trng, self.dropout_rate, in_train=self.use_dropout)

        encoded_q = encoded_q.dimshuffle(1, 0, 2)  # n_timestep_q, n_samples, 786
        encoded_c = encoded_c.dimshuffle(1, 0, 2)  # n_timestep_c, n_samples, 786

        cls_q = encoded_q[0, :, :]
        cls_c = encoded_c[0, :, :]

        encoded_q = encoded_q[1:, :, :]
        encoded_c = encoded_c[1:, :, :]

        query_q = self.query_q.get_output(cls_q)
        query_c = self.query_c.get_output(cls_c)

        x_q_mask = x_q_mask.dimshuffle(1, 0)[1:, :]
        x_c_mask = x_c_mask.dimshuffle(1, 0)[1:, :]

        preact_q = self.att_q.get_output(query_q, encoded_q, x_q_mask)  # n_samples, 786
        emb_q = self.pooler_q.get_output(preact_q, activ='tanh')

        preact_c = self.att_c.get_output(query_c, encoded_c, x_c_mask)  # n_samples, 786
        emb_c = self.pooler_c.get_output(preact_c, activ='tanh')

        ###################################################
        #                  Retrieval Loss                 #
        ###################################################
        # emb_q: n_samples * 786
        # emb_c: n_samples * 786

        n_samples = emb_q.shape[0]

        all_cos = all_cosine(emb_q, emb_c)  # n_samples * n_samples
        pos_cos = all_cos.diagonal()  # n_samples
        diag_mask = 1. - theano.tensor.eye(n_samples, dtype='float32')
        all_cos_T = all_cos.T

        im2re_tri_loss = (all_cos - pos_cos[:, None] + self.margin) * diag_mask
        im2re_tri_loss = tensor.clip(im2re_tri_loss, 0., im2re_tri_loss.max())  # n_samples * n_samples
        mask = tensor.zeros_like(im2re_tri_loss)
        nonzero_ids = tensor.nonzero(im2re_tri_loss)
        mask = tensor.set_subtensor(mask[nonzero_ids], 1.)
        im2re_tri_loss = im2re_tri_loss.sum(1) / (mask.sum(1) + self.eps)

        re2im_tri_loss = (all_cos_T - pos_cos[None, :] + self.margin) * diag_mask
        re2im_tri_loss = tensor.clip(re2im_tri_loss, 0., re2im_tri_loss.max())  # n_samples * n_samples
        mask = tensor.zeros_like(re2im_tri_loss)
        nonzero_ids = tensor.nonzero(re2im_tri_loss)
        mask = tensor.set_subtensor(mask[nonzero_ids], 1.)
        re2im_tri_loss = re2im_tri_loss.sum(1) / (mask.sum(1) + self.eps)

        retrieval_loss = im2re_tri_loss + re2im_tri_loss

        return retrieval_loss

    def build_valid(self):
        x_q = tensor.matrix('x_q', dtype='int64')  # n_samples, n_timestep_q
        x_q_mask = tensor.matrix('x_q_mask', dtype='float32')  # n_samples, n_timestep_q
        x_c = tensor.matrix('x_c', dtype='int64')  # n_samples, n_timestep_c
        x_c_mask = tensor.matrix('x_c_mask', dtype='float32')  # n_samples, n_timestep_c

        inputs = [
            x_q, x_q_mask, x_c, x_c_mask
        ]

        x_q_type = tensor.zeros_like(x_q)
        x_c_type = tensor.ones_like(x_c)

        use_dropout_bert = theano.shared(numpy.float32(False))
        encoded_q = self.bert.get_output(x_q, x_q_type, mask=x_q_mask,
                                         in_train=use_dropout_bert)  # n_samples, n_timestep_q, 786
        encoded_c = self.bert.get_output(x_c, x_c_type, mask=x_c_mask,
                                         in_train=use_dropout_bert)  # n_samples, n_timestep_c, 786

        encoded_q = dropout(encoded_q, self.trng, self.dropout_rate, in_train=self.use_dropout)
        encoded_c = dropout(encoded_c, self.trng, self.dropout_rate, in_train=self.use_dropout)

        encoded_q = encoded_q.dimshuffle(1, 0, 2)  # n_timestep_q, n_samples, 786
        encoded_c = encoded_c.dimshuffle(1, 0, 2)  # n_timestep_c, n_samples, 786

        cls_q = encoded_q[0, :, :]
        cls_c = encoded_c[0, :, :]

        encoded_q = encoded_q[1:, :, :]
        encoded_c = encoded_c[1:, :, :]

        query_q = self.query_q.get_output(cls_q)
        query_c = self.query_c.get_output(cls_c)

        x_q_mask = x_q_mask.dimshuffle(1, 0)[1:, :]
        x_c_mask = x_c_mask.dimshuffle(1, 0)[1:, :]

        preact_q = self.att_q.get_output(query_q, encoded_q, x_q_mask)  # n_samples, 786
        emb_q = self.pooler_q.get_output(preact_q, activ='tanh')

        preact_c = self.att_c.get_output(query_c, encoded_c, x_c_mask)  # n_samples, 786
        emb_c = self.pooler_c.get_output(preact_c, activ='tanh')

        self.f_log_probs = theano.function(inputs, outputs={'emb_1': emb_q, 'emb_2': emb_c})

    def build_valid_single(self):
        x_q = tensor.matrix('x_q', dtype='int64')  # n_samples, n_timestep_q
        x_q_mask = tensor.matrix('x_q_mask', dtype='float32')  # n_samples, n_timestep_q
        x_q_type = tensor.matrix('x_q_type', dtype='int64')

        inputs = [
            x_q, x_q_mask, x_q_type
        ]

        use_dropout_bert = theano.shared(numpy.float32(False))
        encoded_q = self.bert.get_output(x_q, x_q_type, mask=x_q_mask,
                                         in_train=use_dropout_bert)  # n_samples, n_timestep_q, 786

        encoded_q = encoded_q.dimshuffle(1, 0, 2)  # n_timestep_q, n_samples, 786

        cls_q = encoded_q[0, :, :]

        encoded_q = encoded_q[1:, :, :]

        query_q = self.query_q.get_output(cls_q)

        x_q_mask = x_q_mask.dimshuffle(1, 0)[1:, :]

        preact_q = self.att_q.get_output(query_q, encoded_q, x_q_mask)  # n_samples, 786
        emb_q = self.pooler_q.get_output(preact_q, activ='tanh')

        self.f_val_s = theano.function(inputs, emb_q)
