#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from basemodel_retrieval import BaseModel
from RetrievalBert import Bert
from layers_bert import AttentionPoolingLayer, FeedForwardLayer, dropout, relu

import theano
from theano import tensor


class RetrievalModel(BaseModel):
    def __init__(self, config):
        super(RetrievalModel, self).__init__(config)
        self.config = config
        self.token_num = config.get('token_num', 21128)
        self.emb_dim = config.get('emb_dim', 768)
        self.bert_last_num = config.get('bert_last_num', 1)
        self.bert_layer_num = config.get('bert_layer_num', 12)
        self.bert_head_num = config.get('bert_head_num', 12)
        self.bert = None
        self.att_q = None
        self.pooler_q = None
        self.att_c = None
        self.pooler_c = None
        self.query_q = None
        self.query_c = None
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.dropout_rate_bert = config.get("dropout_rate_bert", 0.1)
        self.use_dropout_bert = theano.shared(numpy.float32(config.get('use_dropout_bert', True)))
        self.eps = 1e-12

    def set_dropout_bert(self, val):
        self.use_dropout_bert.set_value(numpy.float32(val))

    def init_model(self):
        self.bert = Bert(token_num=self.token_num, embed_dim=self.emb_dim, bert_last_num=self.bert_last_num,
                         encoder_num=self.bert_layer_num, head_num=self.bert_head_num, hidden_num=self.emb_dim,
                         dropout_rate=self.dropout_rate_bert)

        self.att_q = AttentionPoolingLayer(self.bert.hidden_num, self.bert.hidden_num, name="att_q", scale="xavier")
        self.pooler_q = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="pooler_q", scale="xavier")
        self.att_c = AttentionPoolingLayer(self.bert.hidden_num, self.bert.hidden_num, name="att_c", scale="xavier")
        self.pooler_c = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="pooler_c", scale="xavier")
        self.query_q = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="query_q", scale="xavier")
        self.query_c = FeedForwardLayer(self.bert.hidden_num, self.bert.hidden_num, name="query_c", scale="xavier")

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
        x_q_mask = tensor.matrix('x_q_mask', dtype='float32')  # n_samples, n_timestep_q
        f_mask = tensor.tensor3('f_mask', dtype='float32')  # n_samples, n_table, n_table
        n_samples = x_q.shape[0]
        if self.train_mode == "inner":
            x_c = tensor.tensor3('x_c', dtype='int64')   # n_samples, n_table,  n_timestep_c
            x_c_mask = tensor.tensor3('x_c_mask', dtype='float32')   # n_samples, n_table, n_timestep_c
            self.inputs['x_q'] = x_q
            self.inputs['x_q_mask'] = x_q_mask
            self.inputs['x_c'] = x_c
            self.inputs['x_c_mask'] = x_c_mask
            self.inputs['f_mask'] = f_mask

            n_tables = x_c.shape[1]
            n_tok_t = x_c.shape[2]
            x_c = x_c.reshape([n_samples * n_tables, n_tok_t])
            x_c_mask = x_c_mask.reshape([n_samples * n_tables, n_tok_t])
        else:
            assert self.train_mode == "all"
            x_c = tensor.matrix('x_c', dtype='int64')  # n_table, n_timestep_c
            x_c_mask = tensor.matrix('x_c_mask', dtype='float32')  # n_table, n_timestep_c

            self.inputs['x_q'] = x_q
            self.inputs['x_q_mask'] = x_q_mask
            self.inputs['x_c'] = x_c
            self.inputs['x_c_mask'] = x_c_mask
            self.inputs['f_mask'] = f_mask
            n_tables = x_c.shape[0]

        x_q_type = tensor.zeros_like(x_q)
        x_c_type = tensor.ones_like(x_c)

        encoded_q = self.bert.get_output(x_q, x_q_type, mask=x_q_mask,
                                         in_train=self.use_dropout_bert)  # n_samples, n_tok_q, 786
        encoded_c = self.bert.get_output(x_c, x_c_type, mask=x_c_mask,
                                         in_train=self.use_dropout_bert)  # n_samples, n_tok_t, 786

        encoded_q = dropout(encoded_q, self.trng, self.dropout_rate, in_train=self.use_dropout)
        encoded_c = dropout(encoded_c, self.trng, self.dropout_rate, in_train=self.use_dropout)

        # encoded_q = encoded_q.dimshuffle(1, 0, 2)  # n_timestep_q, n_samples, 786
        # encoded_c = encoded_c.dimshuffle(1, 0, 2)  # n_timestep_c, n_samples, 786

        cls_q = encoded_q[:, 0, :]
        cls_c = encoded_c[:, 0, :]

        encoded_q = encoded_q[:, 1:, :]
        encoded_c = encoded_c[:, 1:, :]

        query_q = self.query_q.get_output(cls_q)
        query_c = self.query_c.get_output(cls_c)

        # x_q_mask = x_q_mask.dimshuffle(1, 0)[1:, :]
        # x_c_mask = x_c_mask.dimshuffle(1, 0)[1:, :]

        preact_q = self.att_q.get_output(query_q, encoded_q, x_q_mask[:, 1:])  # n_samples, 786
        emb_q = self.pooler_q.get_output(preact_q, activ='tanh')

        preact_c = self.att_c.get_output(query_c, encoded_c, x_c_mask[:, 1:])  # n_samples, 786
        emb_c = self.pooler_c.get_output(preact_c, activ='tanh')

        if self.train_mode == "inner":
            emb_c = emb_c.reshape([n_samples, n_tables, self.bert.hidden_num])

            ###################################################
            #                  Retrieval Loss                 #
            ###################################################
            # emb_q: n_samples * 786
            # emb_c: n_samples * n_table * 786
            emb_q_norm = emb_q / tensor.sqrt((emb_q ** 2).sum(-1, keepdims=True))
            emb_c_norm = emb_c / tensor.sqrt((emb_c ** 2).sum(-1, keepdims=True))

            all_cosine = (emb_q_norm[:, None, :] * emb_c_norm).sum(-1)   # n_samples, n_table

        else:
            emb_q_norm = emb_q / tensor.sqrt((emb_q ** 2).sum(-1, keepdims=True))  # n_q, dim
            emb_c_norm = emb_c / tensor.sqrt((emb_c ** 2).sum(-1, keepdims=True))  # n_t, dim

            all_cosine = tensor.dot(emb_q_norm, emb_c_norm.T)  # n_q, n_t

        retrieval_loss = relu(all_cosine[:, None, :] - all_cosine[:, :, None] + self.margin) * f_mask

        return retrieval_loss

    def build_valid(self):
        x_q = tensor.matrix('x_q', dtype='int64')  # n_samples, n_timestep_q
        x_q_mask = tensor.matrix('x_q_mask', dtype='float32')  # n_samples, n_timestep_q

        x_c = tensor.tensor3('x_c', dtype='int64')  # n_samples, n_table,  n_timestep_c
        x_c_mask = tensor.tensor3('x_c_mask', dtype='float32')  # n_samples, n_table, n_timestep_c

        inputs = [x_q, x_q_mask, x_c, x_c_mask]

        n_samples = x_q.shape[0]

        n_tables = x_c.shape[1]
        n_tok_t = x_c.shape[2]
        x_c = x_c.reshape([n_samples * n_tables, n_tok_t])
        x_c_mask = x_c_mask.reshape([n_samples * n_tables, n_tok_t])

        x_q_type = tensor.zeros_like(x_q)
        x_c_type = tensor.ones_like(x_c)

        use_dropout_bert = theano.shared(numpy.float32(False))
        encoded_q = self.bert.get_output(x_q, x_q_type, mask=x_q_mask,
                                         in_train=use_dropout_bert)  # n_samples, n_timestep_q, 786
        encoded_c = self.bert.get_output(x_c, x_c_type, mask=x_c_mask,
                                         in_train=use_dropout_bert)  # n_samples, n_timestep_c, 786

        encoded_q = dropout(encoded_q, self.trng, self.dropout_rate, in_train=self.use_dropout)
        encoded_c = dropout(encoded_c, self.trng, self.dropout_rate, in_train=self.use_dropout)

        # encoded_q = encoded_q.dimshuffle(1, 0, 2)  # n_timestep_q, n_samples, 786
        # encoded_c = encoded_c.dimshuffle(1, 0, 2)  # n_timestep_c, n_samples, 786

        cls_q = encoded_q[:, 0, :]
        cls_c = encoded_c[:, 0, :]

        encoded_q = encoded_q[:, 1:, :]
        encoded_c = encoded_c[:, 1:, :]

        query_q = self.query_q.get_output(cls_q)
        query_c = self.query_c.get_output(cls_c)

        # x_q_mask = x_q_mask.dimshuffle(1, 0)[1:, :]
        # x_c_mask = x_c_mask.dimshuffle(1, 0)[1:, :]

        preact_q = self.att_q.get_output(query_q, encoded_q, x_q_mask[:, 1:])  # n_samples, 786
        emb_q = self.pooler_q.get_output(preact_q, activ='tanh')

        preact_c = self.att_c.get_output(query_c, encoded_c, x_c_mask[:, 1:])  # n_samples, 786
        emb_c = self.pooler_c.get_output(preact_c, activ='tanh')

        emb_c = emb_c.reshape([n_samples, n_tables, self.bert.hidden_num])

        if self.train_mode == "inner":
            emb_q_norm = emb_q / tensor.sqrt((emb_q ** 2).sum(-1, keepdims=True))
            emb_c_norm = emb_c / tensor.sqrt((emb_c ** 2).sum(-1, keepdims=True))

            all_cosine = (emb_q_norm[:, None, :] * emb_c_norm).sum(-1)  # n_samples, n_table
            self.f_log_probs = theano.function(inputs, outputs=all_cosine)
        else:
            self.f_log_probs = theano.function(inputs, outputs={'emb_q': emb_q, 'emb_c': emb_c})




