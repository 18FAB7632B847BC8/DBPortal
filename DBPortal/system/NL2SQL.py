#!/usr/bin/env python
# coding: utf-8
from basemodel import BaseModel
from Bert import Bert
from NL2SQLEncoder import NL2SQLEncoder
from SimpleDecoder import SimpleDecoder
from layers_bert import EmbeddingLayer, PointerNetwork, FeedForwardLayer, CombineLayer
from layers_bert import dropout
from theano import tensor
from theano.ifelse import ifelse
import theano
import numpy
from utils import get_plat_sql
from data import DataHolder
from TableParsing import parse_table


class NL2SQL(BaseModel):
    def __init__(self, config):
        super(NL2SQL, self).__init__(config)
        self.token_num = config.get('token_num', 4064)
        self.type_0_num = config.get('type_0_num', 4)
        self.type_1_num = config.get('type_0_num', 13)
        self.type_2_num = config.get('type_0_num', 21)

        self.emb_dim = config.get('emb_dim', 768)
        self.hid_dim = config.get('hid_dim', 512)
        self.foreign_dim = config.get('foreign_dim', 128)
        self.ctrl_dim = config.get('ctrl_dim', 32)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        self.dropout_rate_bert = config.get("dropout_rate_bert", 0.1)
        self.agg_num = config.get('agg_num', 6)
        self.op_num = config.get('op_num', 14)
        self.bert_last_num = config.get('bert_last_num', 1)
        self.eps = 1e-12
        self.bert_layer_num = config.get('bert_layer_num', 12)
        self.bert_head_num = config.get('bert_head_num', 12)
        self.scale = config.get('scale', 'xavier')
        self.decoder_num = config.get('decoder_num', 4)
        self.decoder_head_num = config.get('decoder_head_num', 8)

        self.bert = None
        self.encoder = None
        self.decoder = None
        self.type_0_emb_layer = None
        self.type_1_emb_layer = None
        self.type_2_emb_layer = None
        self.op_emb_layer = None
        self.agg_emb_layer = None
        self.linear_init_h = None
        self.linear_init_t = None

        self.output_column = None
        self.output_agg = None
        self.output_op = None
        self.output_vls = None
        self.output_vle = None
        self.output_vrs = None
        self.output_vre = None
        self.output_cd = None
        self.output_vn = None
        self.output_t = None

        self.output_co = None
        self.output_d = None
        self.output_o = None
        self.output_c = None
        self.output_l = None
        self.output_nf = None

        self.linear_q = None
        self.linear_h = None
        self.linear_t = None

        self.layers = []

        self.f_init = None
        self.f_next_t = None
        self.f_next_h = None

    def init_model(self):
        self.bert = Bert(token_num=self.token_num, embed_dim=self.emb_dim, bert_last_num=self.bert_last_num,
                         encoder_num=self.bert_layer_num, head_num=self.bert_head_num, hidden_num=self.emb_dim,
                         dropout_rate=self.dropout_rate_bert)

        self.encoder = NL2SQLEncoder(self.emb_dim, self.hid_dim, 32, self.foreign_dim, self.scale, trng=self.trng,
                                     name="encoder")

        self.decoder = SimpleDecoder(self.hid_dim, self.emb_dim, self.decoder_num, self.decoder_head_num,
                                     self.foreign_dim, self.scale, self.dropout_rate, name="decoder", trng=self.trng)

        self.type_0_emb_layer = EmbeddingLayer(self.type_0_num, self.emb_dim - self.emb_dim / 3 - self.hid_dim / 2,
                                               scale=self.scale,
                                               name="type_0_emb_layer")

        self.type_1_emb_layer = EmbeddingLayer(self.type_1_num, self.emb_dim / 3, scale=self.scale,
                                               name="type_1_emb_layer")

        self.type_2_emb_layer = EmbeddingLayer(self.type_2_num, self.hid_dim / 2, scale=self.scale,
                                               name="type_2_emb_layer")

        self.op_emb_layer = EmbeddingLayer(self.op_num, self.hid_dim / 4, scale=self.scale,
                                           name="op_emb_layer")
        self.agg_emb_layer = EmbeddingLayer(self.agg_num, self.hid_dim / 4, scale=self.scale,
                                            name="agg_emb_layer")

        self.linear_init_h = FeedForwardLayer(3 * self.emb_dim + self.emb_dim / 3, self.hid_dim, self.scale,
                                              name="linear_init_header")
        self.linear_init_t = FeedForwardLayer(3 * self.emb_dim + self.emb_dim / 3, self.hid_dim, self.scale,
                                              name="linear_init_table")

        self.output_column = PointerNetwork(self.hid_dim, self.hid_dim, self.scale, name="column_output_pointer")
        self.output_agg = FeedForwardLayer(self.hid_dim, self.agg_num, self.scale, name="agg_output_linear")
        self.output_op = FeedForwardLayer(self.hid_dim, self.op_num, self.scale, name="op_output_linear")
        self.output_vls = PointerNetwork(self.hid_dim, self.hid_dim, self.scale, name="vls_output_pointer")
        self.output_vle = PointerNetwork(self.hid_dim, self.hid_dim, self.scale, name="vle_output_pointer")
        self.output_vrs = PointerNetwork(self.hid_dim, self.hid_dim, self.scale, name="vrs_output_pointer")
        self.output_vre = PointerNetwork(self.hid_dim, self.hid_dim, self.scale, name="vre_output_pointer")
        self.output_cd = FeedForwardLayer(self.hid_dim, 2, self.scale, name="cd_output_linear")
        self.output_vn = FeedForwardLayer(self.hid_dim, 2, self.scale, name="vn_output_linear")
        self.output_t = PointerNetwork(self.hid_dim + self.foreign_dim, self.hid_dim, self.scale,
                                       name="table_output_pointer")

        self.output_co = FeedForwardLayer(self.emb_dim * 3, 2, self.scale, name="co_output_linear")
        self.output_d = FeedForwardLayer(self.emb_dim * 3, 2, self.scale, name="d_output_linear")
        self.output_o = FeedForwardLayer(self.emb_dim * 3, 2, self.scale, name="o_output_linear")
        self.output_c = FeedForwardLayer(self.emb_dim * 3, 4, self.scale, name="c_output_linear")
        self.output_l = FeedForwardLayer(self.emb_dim * 3, 11, self.scale, name="l_output_linear")
        self.output_nf = FeedForwardLayer(self.emb_dim * 3, 2, self.scale, name="nf_output_linear")

        self.linear_q = CombineLayer(self.emb_dim, self.hid_dim, self.scale, name="outer_combine_question")
        self.linear_h = CombineLayer(self.emb_dim, self.hid_dim, self.scale, name="outer_combine_header")
        self.linear_t = CombineLayer(self.emb_dim, self.hid_dim, self.scale, name="outer_combine_table")

        self.layers = [
            self.type_0_emb_layer, self.type_1_emb_layer, self.type_2_emb_layer, self.op_emb_layer, self.agg_emb_layer,
            self.linear_init_h, self.linear_init_t, self.output_column,
            self.output_agg, self.output_op, self.output_vls, self.output_vle, self.output_vrs, self.output_vre,
            self.output_cd, self.output_vn, self.output_t, self.output_co, self.output_d, self.output_o, self.output_c,
            self.output_l, self.output_nf, self.linear_q, self.linear_h, self.linear_t
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

        for k, v in self.encoder.tparams.iteritems():
            assert k not in self.tparams, "Duplicated parameter: %s" % k
            self.tparams[k] = v

        for k, v in self.decoder.tparams.iteritems():
            assert k not in self.tparams, "Duplicated parameter: %s" % k
            self.tparams[k] = v

    @staticmethod
    def get_shifted_emb(y, enc_h, emb_init):
        y_flat = y.flatten()  # n_samples * n_timestep
        n_samples = enc_h.shape[0]
        n_h = enc_h.shape[1]
        n_timestep_y = y.shape[1]
        emb_dim = enc_h.shape[2]
        y_flat_idx = tensor.tile(tensor.arange(n_samples)[:, None],
                                 [1, n_timestep_y]).reshape(y_flat.shape) * n_h + y_flat
        emb_y = enc_h.reshape([n_samples * n_h, emb_dim])[y_flat_idx]
        emb_y = emb_y.reshape([n_samples, n_timestep_y, emb_dim])
        emb_y_shifted = tensor.zeros_like(emb_y)
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 1:, :], emb_y[:, :-1, :])
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 0, :], emb_init)

        return emb_y_shifted

    @staticmethod
    def get_shifted_emb_sample(y, enc_h, emb_init):
        y_flat = y.flatten()  # n_samples * (n_timestep - 1)
        n_samples = enc_h.shape[0]
        n_h = enc_h.shape[1]
        n_timestep_y = y.shape[1]
        emb_dim = enc_h.shape[2]
        y_flat_idx = tensor.tile(tensor.arange(n_samples)[:, None],
                                 [1, n_timestep_y]).reshape(y_flat.shape) * n_h + y_flat
        emb_y = enc_h.reshape([n_samples * n_h, emb_dim])[y_flat_idx]
        emb_y = emb_y.reshape([n_samples, n_timestep_y, emb_dim])
        emb_y_shifted = tensor.zeros((n_samples, n_timestep_y, emb_dim), dtype="float32")
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 1:, :], emb_y[:, 1:, :])
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 0, :], emb_init)

        return emb_y_shifted

    def get_shifted_app_emb(self, y, emb_layer):
        n_samples = y.shape[0]
        n_timestep_y = y.shape[1]

        emb_y = emb_layer.get_output(y).reshape([n_samples, n_timestep_y, self.hid_dim / 4])
        emb_y_shifted = tensor.zeros((n_samples, n_timestep_y, self.hid_dim / 4), dtype="float32")
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 1:, :], emb_y[:, 1:, :])

        return emb_y_shifted


    @staticmethod
    def cross_entropy_loss(logit, y, y_mask=None):
        logit_shape = logit.shape
        y_flat = y.flatten()
        if logit.ndim == 3:
            # logit: n_batch, n_step, n_class
            # y: n_batch, n_step
            log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shape[0] * logit_shape[1], logit_shape[2]]))
            y_flat_idx = tensor.arange(y_flat.shape[0]) * logit_shape[2] + y_flat

            cost_y = log_probs.flatten()[y_flat_idx]
            cost_y = cost_y.reshape([logit_shape[0], logit_shape[1]])
            cost_y = (cost_y * y_mask).sum(1)

        else:
            # logit: n_batch, n_class
            # y: n_batch
            log_probs = -tensor.nnet.logsoftmax(logit)

            y_flat_idx = tensor.arange(y_flat.shape[0]) * logit_shape[1] + y_flat
            cost_y = log_probs.flatten()[y_flat_idx]  # n_samples

        return cost_y

    @staticmethod
    def one_hot(x, n_class):
        # x: n
        assert x.ndim == 1
        n = x.shape[0]
        y = tensor.zeros((n, n_class), dtype="float32")
        y = tensor.set_subtensor(y[tensor.arange(n), x], 1.)
        return y

    @staticmethod
    def label_smooth(y, factor=0.):
        assert y.ndim == 2
        n_class = y.shape[1]
        y *= (1. - factor)
        y += (factor / n_class)

        return y

    def cross_entropy_loss_smooth(self, logit, y, y_mask=None, ctx_mask=None, factor=0.1):
        logit_shape = logit.shape
        y_flat = y.flatten()
        if logit.ndim == 3:
            # logit: n_batch, n_step, n_class
            # y: n_batch, n_step
            log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shape[0] * logit_shape[1], logit_shape[2]]))
            if ctx_mask is not None:
                log_probs = log_probs.reshape(logit_shape)
                log_probs = log_probs * ctx_mask
                log_probs = log_probs.reshape([logit_shape[0] * logit_shape[1], logit_shape[2]])
            y_onehot_smooth = self.label_smooth(self.one_hot(y_flat, logit_shape[2]), factor)
            cost_y = tensor.sum(log_probs * y_onehot_smooth, axis=1)
            cost_y = cost_y.reshape([logit_shape[0], logit_shape[1]])
            cost_y = (cost_y * y_mask).sum(1)

        else:
            # logit: n_batch, n_class
            # y: n_batch
            log_probs = -tensor.nnet.logsoftmax(logit)
            y_onehot_smooth = self.label_smooth(self.one_hot(y_flat, logit_shape[1]), factor)
            cost_y = tensor.sum(log_probs * y_onehot_smooth, axis=1)  # n_samples

        return cost_y

    def build_model(self):
        input_sequence = tensor.matrix("input_sequence", dtype="int32")
        input_mask = tensor.matrix("input_mask", dtype="float32")

        input_type_0 = tensor.matrix("input_type_0", dtype="int32")
        input_type_1 = tensor.matrix("input_type_1", dtype="int32")
        input_type_2 = tensor.matrix("input_type_2", dtype="int32")

        # ENCODER
        join_table = tensor.tensor3('join_table', dtype="int32")
        join_mask = tensor.tensor3('join_table', dtype="float32")

        separator = tensor.matrix("separator", dtype="int32")
        s_e_h = tensor.tensor3('s_e_h', dtype="int32")
        s_e_t = tensor.tensor3('s_e_t', dtype="int32")
        s_e_t_h = tensor.tensor3('s_e_t_h', dtype="int32")
        h2t_idx = tensor.matrix('h2t_idx', dtype="int32")
        q_mask = tensor.matrix("q_mask", dtype="float32")
        t_mask = tensor.matrix("t_mask", dtype="float32")
        p_mask = tensor.matrix("p_mask", dtype="float32")
        t_w_mask = tensor.tensor3('t_w_mask', dtype="float32")
        t_h_mask = tensor.tensor3('t_h_mask', dtype="float32")
        h_mask = tensor.matrix("h_mask", dtype="float32")
        h_w_mask = tensor.tensor3('h_w_mask', dtype="float32")

        # DECODER
        y_column = tensor.matrix("y_column", dtype="int32")
        y_type2 = tensor.matrix("y_type2", dtype="int32")
        y_agg = tensor.matrix("y_agg", dtype="int32")
        y_op = tensor.matrix("y_op", dtype="int32")
        y_vls = tensor.matrix("y_vls", dtype="int32")
        y_vle = tensor.matrix("y_vle", dtype="int32")
        y_vrs = tensor.matrix("y_vrs", dtype="int32")
        y_vre = tensor.matrix("y_vre", dtype="int32")
        y_cd = tensor.matrix("y_cd", dtype="int32")
        y_vn = tensor.matrix("y_vn", dtype="int32")

        y_c_mask = tensor.matrix("y_c_mask", dtype="float32")
        y_cw_mask = tensor.matrix("y_cw_mask", dtype="float32")
        y_w_mask = tensor.matrix("y_w_mask", dtype="float32")
        y_cwo_mask = tensor.matrix("y_cwo_mask", dtype="float32")

        y_t = tensor.matrix("y_t", dtype="int32")
        y_t_mask = tensor.matrix("y_t", dtype="float32")

        y_co = tensor.vector('y_co', dtype="int32")
        y_d = tensor.vector('y_d', dtype="int32")
        y_type1 = tensor.vector('y_type1', dtype="int32")
        y_o = tensor.vector('y_o', dtype="int32")
        y_c = tensor.vector('y_c', dtype="int32")
        y_l = tensor.vector('y_l', dtype="int32")
        y_nf = tensor.vector('y_nf', dtype="int32")

        self.inputs['input_sequence'] = input_sequence
        self.inputs['input_mask'] = input_mask
        self.inputs['input_type_0'] = input_type_0
        self.inputs['input_type_1'] = input_type_1
        self.inputs['input_type_2'] = input_type_2
        self.inputs['join_table'] = join_table
        self.inputs['join_mask'] = join_mask
        self.inputs['separator'] = separator
        self.inputs['s_e_h'] = s_e_h
        self.inputs['s_e_t'] = s_e_t
        self.inputs['s_e_t_h'] = s_e_t_h
        self.inputs['h2t_idx'] = h2t_idx
        self.inputs['q_mask'] = q_mask
        self.inputs['t_mask'] = t_mask
        self.inputs['p_mask'] = p_mask
        self.inputs['t_w_mask'] = t_w_mask
        self.inputs['t_h_mask'] = t_h_mask
        self.inputs['h_mask'] = h_mask
        self.inputs['h_w_mask'] = h_w_mask
        self.inputs['y_column'] = y_column
        self.inputs['y_type2'] = y_type2
        self.inputs['y_agg'] = y_agg
        self.inputs['y_op'] = y_op
        self.inputs['y_vls'] = y_vls
        self.inputs['y_vle'] = y_vle
        self.inputs['y_vrs'] = y_vrs
        self.inputs['y_vre'] = y_vre
        self.inputs['y_cd'] = y_cd
        self.inputs['y_vn'] = y_vn
        self.inputs['y_c_mask'] = y_c_mask
        self.inputs['y_cw_mask'] = y_cw_mask
        self.inputs['y_w_mask'] = y_w_mask
        self.inputs['y_cwo_mask'] = y_cwo_mask
        self.inputs['y_t'] = y_t
        self.inputs['y_t_mask'] = y_t_mask
        self.inputs['y_co'] = y_co
        self.inputs['y_d'] = y_d
        self.inputs['y_type1'] = y_type1
        self.inputs['y_o'] = y_o
        self.inputs['y_c'] = y_c
        self.inputs['y_l'] = y_l
        self.inputs['y_nf'] = y_nf

        n_batch, n_seq = input_sequence.shape

        emb_type_0 = self.type_0_emb_layer.get_output(input_type_0).reshape([n_batch, n_seq, self.emb_dim - self.emb_dim / 3 - self.hid_dim / 2])
        emb_type_1 = self.type_1_emb_layer.get_output(input_type_1).reshape([n_batch, n_seq, self.emb_dim / 3])
        emb_type_2 = self.type_2_emb_layer.get_output(input_type_2).reshape([n_batch, n_seq, self.hid_dim / 2])

        emb_type = tensor.concatenate([emb_type_0, emb_type_1, emb_type_2], axis=2)

        emb_type = dropout(emb_type, self.trng, self.dropout_rate, self.use_dropout)

        enc_x = self.bert.get_output(input_sequence, input_mask, self.use_bert_dropout)
        enc_x = dropout(enc_x, self.trng, self.dropout_rate, self.use_dropout)
        global_ctx, join_embs, emb_q, emb_p, emb_d, ctxs_t, ctxs_h, ctxs_q = self.encoder.get_output(
            enc_x, emb_type, join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
            q_mask, t_mask, t_w_mask, t_h_mask, h_w_mask, p_mask, self.dropout_rate, self.use_dropout
        )

        emb_y_type1 = self.type_1_emb_layer.get_output(y_type1)
        ctxs_q = self.linear_q.get_output(global_ctx, ctxs_q)
        ctxs_t = self.linear_t.get_output(global_ctx, ctxs_t)
        ctxs_h = self.linear_h.get_output(global_ctx, ctxs_h)

        init_state_h = self.linear_init_h.get_output(tensor.concatenate([emb_q, emb_p, emb_d, emb_y_type1], axis=-1),
                                                     activ="tanh")
        init_state_t = self.linear_init_t.get_output(tensor.concatenate([emb_q, emb_p, emb_d, emb_y_type1], axis=-1),
                                                     activ="tanh")
        embs_h_shifted = self.get_shifted_emb(y_column, ctxs_h, init_state_h)
        embs_t_shifted = self.get_shifted_emb(y_t, ctxs_t, init_state_t)

        embs_op = self.op_emb_layer.get_output(y_op).reshape([y_op.shape[0], y_op.shape[1], self.hid_dim / 4])
        embs_agg = self.agg_emb_layer.get_output(y_agg).reshape([y_agg.shape[0], y_agg.shape[1], self.hid_dim / 4])
        embs_y_type2 = self.type_2_emb_layer.get_output(y_type2).reshape([y_type2.shape[0], y_type2.shape[1],
                                                                         self.hid_dim / 2])

        embs_op_shifted = tensor.zeros_like(embs_op)
        embs_op_shifted = tensor.set_subtensor(embs_op_shifted[:, 1:, :], embs_op[:, :-1, :])
        embs_agg_shifted = tensor.zeros_like(embs_agg)
        embs_agg_shifted = tensor.set_subtensor(embs_agg_shifted[:, 1:, :], embs_agg[:, :-1, :])

        embs_app = tensor.concatenate([embs_y_type2, embs_agg_shifted, embs_op_shifted], axis=2)

        states_col = self.decoder.get_output_col(embs_h_shifted, y_c_mask, embs_app, enc_x, input_mask,
                                                 self.use_dropout)

        states_tab = self.decoder.get_output_tab(embs_t_shifted, y_t_mask, enc_x, input_mask, self.use_dropout)

        logit_column = self.output_column.get_output(states_col, ctxs_h, h_mask)
        logit_agg = self.output_agg.get_output(states_col, activ="linear")
        logit_op = self.output_op.get_output(states_col, activ="linear")
        logit_vls = self.output_vls.get_output(states_col, ctxs_q, q_mask)
        logit_vle = self.output_vle.get_output(states_col, ctxs_q, q_mask)
        logit_vrs = self.output_vrs.get_output(states_col, ctxs_q, q_mask)
        logit_vre = self.output_vre.get_output(states_col, ctxs_q, q_mask)
        logit_cd = self.output_cd.get_output(states_col, activ="linear")
        logit_vn = self.output_vn.get_output(states_col, activ="linear")

        logit_t = self.output_t.get_output(states_tab, tensor.concatenate([ctxs_t, join_embs], axis=2), t_mask)

        logit_co = self.output_co.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_d = self.output_d.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_o = self.output_o.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_c = self.output_c.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_l = self.output_l.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_nf = self.output_nf.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")

        cost_column = self.cross_entropy_loss_smooth(logit_column, y_column, y_c_mask, h_mask[:, None, :])
        cost_op = self.cross_entropy_loss_smooth(logit_op, y_op, y_w_mask, None)
        cost_agg = self.cross_entropy_loss_smooth(logit_agg, y_agg, y_cwo_mask, None)
        cost_vls = self.cross_entropy_loss_smooth(logit_vls, y_vls, y_w_mask, q_mask[:, None, :])
        cost_vle = self.cross_entropy_loss_smooth(logit_vle, y_vle, y_w_mask, q_mask[:, None, :])
        cost_vrs = self.cross_entropy_loss_smooth(logit_vrs, y_vrs, y_w_mask, q_mask[:, None, :])
        cost_vre = self.cross_entropy_loss_smooth(logit_vre, y_vre, y_w_mask, q_mask[:, None, :])
        cost_cd = self.cross_entropy_loss_smooth(logit_cd, y_cd, y_cw_mask, None)
        cost_vn = self.cross_entropy_loss_smooth(logit_vn, y_vn, y_w_mask, None)
        cost_t = self.cross_entropy_loss_smooth(logit_t, y_t, y_t_mask, t_mask[:, None, :])
        cost_d = self.cross_entropy_loss_smooth(logit_d, y_d, None, None)
        cost_co = self.cross_entropy_loss_smooth(logit_co, y_co, None, None)
        cost_o = self.cross_entropy_loss_smooth(logit_o, y_o, None, None)
        cost_c = self.cross_entropy_loss_smooth(logit_c, y_c, None, None)
        cost_l = self.cross_entropy_loss_smooth(logit_l, y_l, None, None)
        cost_nf = self.cross_entropy_loss_smooth(logit_nf, y_nf, None, None)

        cost = cost_column + cost_op + cost_agg + cost_vls + cost_vle + cost_vrs + cost_vre + \
            cost_cd + cost_vn + cost_t + cost_d + cost_co + cost_o + cost_c + cost_l + cost_nf

        self.f_log_probs = theano.function(self.inputs.values(), {
            'cost_column': cost_column,
            'cost_op': cost_op,
            'cost_agg': cost_agg,
            'cost_vls': cost_vls,
            'cost_vle': cost_vle,
            'cost_vrs': cost_vrs,
            'cost_vre': cost_vre,
            'cost_cd': cost_cd,
            'cost_vn': cost_vn,
            'cost_t': cost_t,
            'cost_d': cost_d,
            'cost_co': cost_co,
            'cost_o': cost_o,
            'cost_c': cost_c,
            'cost_l': cost_l,
            'cost_nf': cost_nf
        })

        return cost

    def build_sampler(self):
        input_sequence = tensor.matrix("input_sequence", dtype="int32")
        input_mask = tensor.matrix("input_mask", dtype="float32")

        input_type_0 = tensor.matrix("input_type_0", dtype="int32")
        input_type_1 = tensor.matrix("input_type_1", dtype="int32")
        input_type_2 = tensor.matrix("input_type_2", dtype="int32")

        # ENCODER
        join_table = tensor.tensor3('join_table', dtype="int32")
        join_mask = tensor.tensor3('join_table', dtype="float32")

        separator = tensor.matrix("separator", dtype="int32")
        s_e_h = tensor.tensor3('s_e_h', dtype="int32")
        s_e_t = tensor.tensor3('s_e_t', dtype="int32")
        s_e_t_h = tensor.tensor3('s_e_t_h', dtype="int32")
        h2t_idx = tensor.matrix('h2t_idx', dtype="int32")
        q_mask = tensor.matrix("q_mask", dtype="float32")
        t_mask = tensor.matrix("t_mask", dtype="float32")
        p_mask = tensor.matrix("p_mask", dtype="float32")
        t_w_mask = tensor.tensor3('t_w_mask', dtype="float32")
        t_h_mask = tensor.tensor3('t_h_mask', dtype="float32")
        h_mask = tensor.matrix("h_mask", dtype="float32")
        h_w_mask = tensor.tensor3('h_w_mask', dtype="float32")
        n_batch, n_seq = input_sequence.shape

        y_type1 = tensor.vector('y_type1', dtype="int32")

        emb_type_0 = self.type_0_emb_layer.get_output(input_type_0).reshape([n_batch, n_seq, self.emb_dim - self.emb_dim / 3 - self.hid_dim / 2])
        emb_type_1 = self.type_1_emb_layer.get_output(input_type_1).reshape([n_batch, n_seq, self.emb_dim / 3])
        emb_type_2 = self.type_2_emb_layer.get_output(input_type_2).reshape([n_batch, n_seq, self.hid_dim / 2])

        emb_type = tensor.concatenate([emb_type_0, emb_type_1, emb_type_2], axis=2)

        enc_x = self.bert.get_output(input_sequence, input_mask, self.use_bert_dropout)

        global_ctx, join_embs, emb_q, emb_p, emb_d, ctxs_t, ctxs_h, ctxs_q = self.encoder.get_output(
            enc_x, emb_type, join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
            q_mask, t_mask, t_w_mask, t_h_mask, h_w_mask, p_mask, 0., self.use_dropout
        )

        emb_y_type1 = self.type_1_emb_layer.get_output(y_type1)
        ctxs_q = self.linear_q.get_output(global_ctx, ctxs_q)
        ctxs_t = self.linear_t.get_output(global_ctx, ctxs_t)
        ctxs_h = self.linear_h.get_output(global_ctx, ctxs_h)

        init_state_h = self.linear_init_h.get_output(tensor.concatenate([emb_q, emb_p, emb_d, emb_y_type1], axis=-1),
                                                     activ="tanh")
        init_state_t = self.linear_init_t.get_output(tensor.concatenate([emb_q, emb_p, emb_d, emb_y_type1], axis=-1),
                                                     activ="tanh")

        logit_co = self.output_co.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_d = self.output_d.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_o = self.output_o.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_c = self.output_c.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_l = self.output_l.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")
        logit_nf = self.output_nf.get_output(tensor.concatenate([emb_q, emb_p, emb_d], axis=1), activ="linear")

        prob_logit_co = tensor.nnet.logsoftmax(logit_co)
        prob_logit_d = tensor.nnet.logsoftmax(logit_d)
        prob_logit_o = tensor.nnet.logsoftmax(logit_o)
        prob_logit_c = tensor.nnet.logsoftmax(logit_c)
        prob_logit_l = tensor.nnet.logsoftmax(logit_l)
        prob_logit_nf = tensor.nnet.logsoftmax(logit_nf)

        inputs = [
            input_sequence, input_mask, input_type_0, input_type_1, input_type_2,
            join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
            q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_w_mask, y_type1
        ]

        outputs = [
            ctxs_q, ctxs_t, ctxs_h, enc_x, init_state_h, init_state_t, join_embs,
            prob_logit_co, prob_logit_d, prob_logit_o, prob_logit_c, prob_logit_l, prob_logit_nf
        ]

        self.f_init = theano.function(inputs, outputs)

        y_t = tensor.matrix("y_t", dtype="int32")
        init_state_t = tensor.matrix("init_state_t", dtype="float32")

        emb_y_t_shifted = self.get_shifted_emb_sample(y_t, ctxs_t, init_state_t)

        # one_step_table

        y_t_mask = tensor.ones((emb_y_t_shifted.shape[0], emb_y_t_shifted.shape[1]), dtype="float32")
        states_tab = self.decoder.get_output_tab(emb_y_t_shifted, y_t_mask, enc_x, input_mask, self.use_dropout)

        logit_t = self.output_t.get_output(states_tab,
                                           tensor.concatenate([ctxs_t, join_embs], axis=2),
                                           t_mask)[:, -1, :]

        next_log_probs = tensor.nnet.logsoftmax(logit_t)  # n_batch, n_table

        inputs = [y_t, init_state_t, enc_x, input_mask, ctxs_t, t_mask, join_embs]

        outputs = next_log_probs
        self.f_next_t = theano.function(inputs, outputs)

        # one_step_column
        y_column = tensor.matrix("y_column", dtype="int32")  # n_batch, n_timestep_y - 1
        y_op = tensor.matrix("y_op", dtype="int32")  # n_batch, n_timestep_y
        y_agg = tensor.matrix("y_agg", dtype="int32")  # n_batch, n_timestep_y
        y_type2 = tensor.matrix("y_type2", dtype="int32")  # n_batch, n_timestep_y

        init_state_h = tensor.matrix("init_state_h", dtype="float32")

        emb_y_column_shifted = self.get_shifted_emb_sample(y_column, ctxs_h, init_state_h)

        embs_op = self.get_shifted_app_emb(y_op, self.op_emb_layer)

        embs_agg = self.get_shifted_app_emb(y_agg, self.agg_emb_layer)

        embs_y_type2 = self.type_2_emb_layer.get_output(y_type2).reshape([y_type2.shape[0], y_type2.shape[1],
                                                                          self.hid_dim / 2])

        embs_app = tensor.concatenate([embs_y_type2, embs_agg, embs_op], axis=2)

        y_c_mask = tensor.ones((emb_y_column_shifted.shape[0], emb_y_column_shifted.shape[1]), dtype="float32")

        states_col = self.decoder.get_output_col(emb_y_column_shifted, y_c_mask, embs_app, enc_x, input_mask,
                                                 self.use_dropout)

        logit_column = self.output_column.get_output(states_col, ctxs_h, h_mask)[:, -1, :]
        logit_agg = self.output_agg.get_output(states_col, activ="linear")[:, -1, :]
        logit_op = self.output_op.get_output(states_col, activ="linear")[:, -1, :]
        logit_vls = self.output_vls.get_output(states_col, ctxs_q, q_mask)[:, -1, :]
        logit_vle = self.output_vle.get_output(states_col, ctxs_q, q_mask)[:, -1, :]
        logit_vrs = self.output_vrs.get_output(states_col, ctxs_q, q_mask)[:, -1, :]
        logit_vre = self.output_vre.get_output(states_col, ctxs_q, q_mask)[:, -1, :]
        logit_cd = self.output_cd.get_output(states_col, activ="linear")[:, -1, :]
        logit_vn = self.output_vn.get_output(states_col, activ="linear")[:, -1, :]

        log_prob_column = tensor.nnet.logsoftmax(logit_column)
        log_prob_agg = tensor.nnet.logsoftmax(logit_agg)
        log_prob_op = tensor.nnet.logsoftmax(logit_op)
        log_prob_vls = tensor.nnet.logsoftmax(logit_vls)
        log_prob_vle = tensor.nnet.logsoftmax(logit_vle)
        log_prob_vrs = tensor.nnet.logsoftmax(logit_vrs)
        log_prob_vre = tensor.nnet.logsoftmax(logit_vre)
        log_prob_cd = tensor.nnet.logsoftmax(logit_cd)
        log_prob_vn = tensor.nnet.logsoftmax(logit_vn)

        inputs = [
            y_column, y_op, y_agg, y_type2, init_state_h, enc_x, input_mask,
            ctxs_q, q_mask, ctxs_h, h_mask
        ]

        outputs = [
            log_prob_column, log_prob_agg, log_prob_op, log_prob_vls, log_prob_vle, log_prob_vrs, log_prob_vre,
            log_prob_cd, log_prob_vn,
        ]

        self.f_next_h = theano.function(inputs, outputs)

    def inf_encode(self, data):

        ctxs_q, ctxs_t, ctxs_h, ctxs_all, init_state_h, init_state_t, join_embs, \
            y_co, y_d, y_o, y_c, y_l, y_nf = self.f_init(*data)

        return ctxs_q, ctxs_t, ctxs_h, ctxs_all, init_state_h, init_state_t, join_embs, y_co, y_d, y_o, y_c, y_l, y_nf

    def beam_search_table(self, init_state_t, ctxs_all, ctx_mask, ctxs_t, t_mask, join_embs, beam_size=2, maxlen=20):
        num_t = t_mask.shape[1]

        beam_size = min(beam_size, num_t)

        final_sample = []
        final_score = []
        hyp_samples = [[]]
        hyp_scores = numpy.zeros(1, dtype='float32')

        next_w = -1 * numpy.ones((1, 1)).astype('int32')  # bos indicator  n_batch, n_timestep

        live_beam = beam_size

        tiled_ctxs_t = numpy.tile(ctxs_t, [1, 1, 1])
        tiled_t_mask = numpy.tile(t_mask, [1, 1])
        tiled_join_embs = numpy.tile(join_embs, [1, 1, 1])
        tiled_init_state = numpy.tile(init_state_t, [1, 1])
        tiled_ctxs_all = numpy.tile(ctxs_all, [1, 1, 1])
        tiled_ctx_mask = numpy.tile(ctx_mask, [1, 1])

        for t in xrange(maxlen):
            next_log_probs = self.f_next_t(
                next_w, tiled_init_state, tiled_ctxs_all, tiled_ctx_mask, tiled_ctxs_t, tiled_t_mask, tiled_join_embs
            )

            if t == 0:
                next_log_probs[:, 0] = -numpy.inf
            cand_scores = hyp_scores[:, None] - next_log_probs
            cand_scores.shape = cand_scores.size
            ranks_flat = cand_scores.argpartition(live_beam - 1)[:live_beam]
            costs = cand_scores[ranks_flat]
            live_beam = 0
            new_hyp_scores = []
            new_hyp_samples = []
            trans_idxs = ranks_flat // next_log_probs.shape[1]
            word_idxs = ranks_flat % next_log_probs.shape[1]
            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                new_hyp = hyp_samples[ti] + [wi]
                if wi == 0:
                    # <eos> found, separate out finished hypotheses
                    final_sample.append(new_hyp)
                    final_score.append(costs[idx])
                else:
                    new_hyp_samples.append(new_hyp)
                    new_hyp_scores.append(costs[idx])
                    live_beam += 1
            hyp_scores = numpy.array(new_hyp_scores, dtype='float32')
            hyp_samples = new_hyp_samples
            if live_beam == 0:
                break

            next_w = numpy.array([[-1] + w for w in hyp_samples], dtype="int32")
            tiled_init_state = numpy.tile(init_state_t, [live_beam, 1])
            tiled_ctxs_t = numpy.tile(ctxs_t, [live_beam, 1, 1])
            tiled_t_mask = numpy.tile(t_mask, [live_beam, 1])
            tiled_join_embs = numpy.tile(join_embs, [live_beam, 1, 1])
            tiled_ctxs_all = numpy.tile(ctxs_all, [live_beam, 1, 1])
            tiled_ctx_mask = numpy.tile(ctx_mask, [live_beam, 1])

        for idx in range(live_beam):
            final_sample.append(hyp_samples[idx])
            final_score.append(hyp_scores[idx])

        return final_sample, final_score

    def beam_search_column(self, init_state_h, ctxs_all, ctxs_mask, ctxs_q, q_mask, ctxs_h, h_mask,
                           beam_size=4, maxlen=20):
        final_list = []

        hyp_samples = [[]]
        hyp_type2s = [[17]]
        hyp_ops = [[]]
        hyp_aggs = [[]]
        hyp_vlss = [[]]
        hyp_vles = [[]]
        hyp_vrss = [[]]
        hyp_vres = [[]]
        hyp_cds = [[]]
        hyp_vns = [[]]

        hyp_scores = numpy.zeros(1, dtype='float32')

        live_beam = beam_size
        next_type2 = 17 * numpy.ones((1, 1), dtype="int32")
        next_col = -1 * numpy.ones((1, 1), dtype="int32")
        next_op = -1 * numpy.ones((1, 1), dtype="int32")
        next_agg = -1 * numpy.ones((1, 1), dtype="int32")

        tiled_ctxs_all = numpy.tile(ctxs_all, [1, 1, 1])
        tiled_ctxs_mask = numpy.tile(ctxs_mask, [1, 1])
        tiled_ctxs_q = numpy.tile(ctxs_q, [1, 1, 1])
        tiled_q_mask = numpy.tile(q_mask, [1, 1])
        tiled_ctxs_h = numpy.tile(ctxs_h, [1, 1, 1])
        tiled_h_mask = numpy.tile(h_mask, [1, 1])
        tiled_init_state = numpy.tile(init_state_h, [1, 1])

        for t in xrange(maxlen):

            rval = self.f_next_h(next_col, next_op, next_agg, next_type2, tiled_init_state,
                                 tiled_ctxs_all, tiled_ctxs_mask,
                                 tiled_ctxs_q, tiled_q_mask, tiled_ctxs_h, tiled_h_mask)

            next_log_probs, log_prob_agg, log_prob_op, log_prob_vls, log_prob_vle, log_prob_vrs, log_prob_vre, \
                log_prob_cd, log_prob_vn = rval
            if t == 0:
                next_log_probs[:, 0] = -numpy.inf

            cand_scores = hyp_scores[:, None] - next_log_probs

            cand_scores.shape = cand_scores.size

            ranks_flat = cand_scores.argpartition(live_beam - 1)[:live_beam]

            costs = cand_scores[ranks_flat]

            live_beam = 0
            new_hyp_scores = []
            new_hyp_samples = []
            new_hyp_type2s = []
            new_hyp_ops = []
            new_hyp_aggs = []
            new_hyp_vlss = []
            new_hyp_vles = []
            new_hyp_vrss = []
            new_hyp_vres = []
            new_hyp_cds = []
            new_hyp_vns = []

            trans_idxs = ranks_flat // next_log_probs.shape[1]
            word_idxs = ranks_flat % next_log_probs.shape[1]

            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                new_col = hyp_samples[ti] + [wi]
                if hyp_samples[ti].count(0) == 0:
                    t2 = 17
                    new_op = hyp_ops[ti] + [0]
                    new_agg = hyp_aggs[ti] + [log_prob_agg[ti].argmax()]
                    new_vls = hyp_vlss[ti] + [-1]
                    new_vle = hyp_vles[ti] + [-1]
                    new_vrs = hyp_vrss[ti] + [-1]
                    new_vre = hyp_vres[ti] + [-1]
                    new_cd = hyp_cds[ti] + [log_prob_cd[ti].argmax()]
                    new_vn = hyp_vns[ti] + [0]
                elif hyp_samples[ti].count(0) == 1:
                    t2 = 18
                    new_op = hyp_ops[ti] + [log_prob_op[ti].argmax()]
                    new_agg = hyp_aggs[ti] + [log_prob_agg[ti].argmax()]
                    new_vls = hyp_vlss[ti] + [log_prob_vls[ti].argmax()]
                    new_vle = hyp_vles[ti] + [log_prob_vle[ti].argmax()]
                    new_vrs = hyp_vrss[ti] + [log_prob_vrs[ti].argmax()]
                    new_vre = hyp_vres[ti] + [log_prob_vre[ti].argmax()]
                    new_cd = hyp_cds[ti] + [log_prob_cd[ti].argmax()]
                    new_vn = hyp_vns[ti] + [log_prob_vn[ti].argmax()]
                elif hyp_samples[ti].count(0) == 2:
                    t2 = 19
                    new_op = hyp_ops[ti] + [0]
                    new_agg = hyp_aggs[ti] + [log_prob_agg[ti].argmax()]
                    new_vls = hyp_vlss[ti] + [-1]
                    new_vle = hyp_vles[ti] + [-1]
                    new_vrs = hyp_vrss[ti] + [-1]
                    new_vre = hyp_vres[ti] + [-1]
                    new_cd = hyp_cds[ti] + [0]
                    new_vn = hyp_vns[ti] + [0]
                elif hyp_samples[ti].count(0) == 3:
                    t2 = 20
                    new_op = hyp_ops[ti] + [0]
                    new_agg = hyp_aggs[ti] + [0]
                    new_vls = hyp_vlss[ti] + [-1]
                    new_vle = hyp_vles[ti] + [-1]
                    new_vrs = hyp_vrss[ti] + [-1]
                    new_vre = hyp_vres[ti] + [-1]
                    new_cd = hyp_cds[ti] + [0]
                    new_vn = hyp_vns[ti] + [0]
                else:
                    raise

                if wi == 0:
                    if t2 == 20:
                        new_type2 = hyp_type2s[ti]
                        final_list.append({
                            "s_column": new_col,
                            's_op': new_op,
                            's_agg': new_agg,
                            's_distinct': new_cd,
                            's_nested_value': new_vn,
                            's_value_left_start': new_vls,
                            's_value_left_end': new_vle,
                            's_value_right_start': new_vrs,
                            's_value_right_end': new_vre,
                            's_type2': new_type2,
                            'score': costs[idx]
                        })
                        continue
                    else:
                        t2 += 1
                new_type2 = hyp_type2s[ti] + [t2]
                new_hyp_scores.append(costs[idx])
                new_hyp_samples.append(new_col)
                new_hyp_ops.append(new_op)
                new_hyp_aggs.append(new_agg)
                new_hyp_cds.append(new_cd)
                new_hyp_type2s.append(new_type2)
                new_hyp_vlss.append(new_vls)
                new_hyp_vles.append(new_vle)
                new_hyp_vrss.append(new_vrs)
                new_hyp_vres.append(new_vre)
                new_hyp_vns.append(new_vn)
                live_beam += 1

            hyp_scores = numpy.array(new_hyp_scores, dtype='float32')
            hyp_samples = new_hyp_samples
            hyp_ops = new_hyp_ops
            hyp_aggs = new_hyp_aggs
            hyp_cds = new_hyp_cds
            hyp_type2s = new_hyp_type2s
            hyp_vlss = new_hyp_vlss
            hyp_vles = new_hyp_vles
            hyp_vrss = new_hyp_vrss
            hyp_vres = new_hyp_vres
            hyp_vns = new_hyp_vns

            if live_beam == 0:
                break

            next_col = numpy.array([[-1] + w for w in hyp_samples], dtype="int32")
            next_op = numpy.array([[-1] + w for w in hyp_ops], dtype="int32")
            next_agg = numpy.array([[-1] + w for w in hyp_aggs], dtype="int32")
            next_type2 = numpy.array(hyp_type2s, dtype="int32")

            tiled_ctxs_all = numpy.tile(ctxs_all, [live_beam, 1, 1])
            tiled_ctxs_mask = numpy.tile(ctxs_mask, [live_beam, 1])
            tiled_ctxs_q = numpy.tile(ctxs_q, [live_beam, 1, 1])
            tiled_q_mask = numpy.tile(q_mask, [live_beam, 1])
            tiled_ctxs_h = numpy.tile(ctxs_h, [live_beam, 1, 1])
            tiled_h_mask = numpy.tile(h_mask, [live_beam, 1])
            tiled_init_state = numpy.tile(init_state_h, [live_beam, 1])

        for idx in range(live_beam):
            if len(hyp_samples[idx]) < len(hyp_type2s[idx]):
                assert len(hyp_samples[idx]) == len(hyp_type2s[idx]) - 1
                hyp_samples[idx].append(0)
                hyp_ops[idx].append(0)
                hyp_aggs[idx].append(0)
                hyp_cds[idx].append(0)
                hyp_vns[idx].append(0)
                hyp_vlss[idx].append(-1)
                hyp_vles[idx].append(-1)
                hyp_vrss[idx].append(-1)
                hyp_vres[idx].append(-1)
                last_type2 = hyp_type2s[idx][-1]
                for i in range(last_type2 + 1, 21):
                    hyp_type2s[idx].append(i)
                    hyp_samples[idx].append(0)
                    hyp_ops[idx].append(0)
                    hyp_aggs[idx].append(0)
                    hyp_cds[idx].append(0)
                    hyp_vns[idx].append(0)
                    hyp_vlss[idx].append(-1)
                    hyp_vles[idx].append(-1)
                    hyp_vrss[idx].append(-1)
                    hyp_vres[idx].append(-1)
            final_list.append({
                "s_column": hyp_samples[idx],
                's_op': hyp_ops[idx],
                's_agg': hyp_aggs[idx],
                's_distinct': hyp_cds[idx],
                's_nested_value': hyp_vns[idx],
                's_value_left_start': hyp_vlss[idx],
                's_value_left_end': hyp_vles[idx],
                's_value_right_start': hyp_vrss[idx],
                's_value_right_end': hyp_vres[idx],
                's_type2': hyp_type2s[idx],
                'score': hyp_scores[idx]
            })

        return final_list

    def construct_simple_sql(self, data_list):
        input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
        join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
        q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, y_type1 = data_list

        ctxs_q, ctxs_t, ctxs_h, ctxs_all, init_state_h, init_state_t, join_embs, prob_logit_co, prob_logit_d, prob_logit_o, prob_logit_c, prob_logit_l, prob_logit_nf = self.inf_encode(
            [input_sequence, input_mask, input_type_0, input_type_1, input_type_2,
             join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
             q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_w_mask, y_type1]
        )

        y_co = numpy.argmax(prob_logit_co, axis=1)
        y_d = numpy.argmax(prob_logit_d, axis=1)
        y_o = numpy.argmax(prob_logit_o, axis=1)
        y_c = numpy.argmax(prob_logit_c, axis=1)
        y_l = numpy.argmax(prob_logit_l, axis=1)
        y_nf = numpy.argmax(prob_logit_nf, axis=1)

        sql_dict = {
            'distinct': y_d[0],
            'connector': y_co[0],
            'order': y_o[0],
            'combine': y_c[0],
            'limit': y_l[0],
            'nested_from': y_nf[0],
        }

        table_seqs, table_scores = self.beam_search_table(
            init_state_t, ctxs_all, input_mask, ctxs_t, t_mask, join_embs, beam_size=2, maxlen=8
        )

        table_seq_idx = numpy.argmin(table_scores)
        # table_seq_idx = numpy.argmin(table_scores / numpy.array([len(ts) for ts in table_seqs]))

        sql_dict['from_table'] = table_seqs[table_seq_idx]

        col_list = self.beam_search_column(
            init_state_h, ctxs_all, input_mask, ctxs_q, q_mask, ctxs_h, h_mask,
            beam_size=2, maxlen=12
        )

        col_scores = numpy.array([col_dict['score'] for col_dict in col_list])
        # col_scores = col_scores / numpy.array([len([col_dict['s_column']]) for col_dict in col_list])

        sql_dict.update(col_list[numpy.argmin(col_scores)])

        return sql_dict

    def construct_comp_sql(self, data, database):
        nv_idx = 0
        nf_idx = 0
        nu_idx = 0
        ni_idx = 0
        ne_idx = 0

        input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
        join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
        q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, input_blocks = data

        y_type1 = numpy.array([7], dtype="int32")

        data_list = [
            input_sequence, input_mask, input_type_0, input_type_1, input_type_2,
            join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
            q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, y_type1
        ]

        if input_sequence.shape[1] > 510:
            print input_blocks[0]
            exit(0)

        sql_block = self.construct_simple_sql(data_list)

        stack = []
        input_block = input_blocks[0]
        q_toks = input_block["question_token"]
        db_id = input_block["db_id"]
        table_dict = database[db_id]
        sql_result = get_plat_sql(sql_block, q_toks, table_dict, nv_idx, nf_idx, nu_idx, ni_idx, ne_idx, stack)
        sql, stack, nv_idx, nf_idx, nu_idx, ni_idx, ne_idx = sql_result
        append_sql = [2]
        append_type1 = [6]
        append_type2 = [0]
        comp_sql = sql
        pre_sql_block = sql_block
        idx = "ORI"

        inner_count = 1
        while len(stack) > 0:
            inner_count += 1
            if inner_count > 8:
                break
            append_sql.pop()
            append_type1.pop()
            append_type2.pop()
            cnse = input_block['column_name_start_end']
            input_seq = input_block['input_sequence']
            for c, t2 in zip(pre_sql_block['s_column'], pre_sql_block['s_type2']):
                if c == 0:
                    continue
                st, ed = cnse[c]
                append_sql.extend(input_seq[st:ed])
                append_type2.extend([t2] * (ed - st))
                if idx == "ORI":
                    t1 = 7
                elif 'VALUE' in idx:
                    t1 = 8
                elif 'INTERSECT' in idx:
                    t1 = 9
                elif 'UNION' in idx:
                    t1 = 10
                elif 'EXCEPT' in idx:
                    t1 = 11
                elif 'FROM' in idx:
                    t1 = 12
                else:
                    raise
                append_type1.extend([t1] * (ed - st))
            append_sql.append(2)
            append_type1.append(6)
            append_type2.append(0)
            idx = stack.pop()
            assert idx in comp_sql
            assert len(append_sql) == len(append_type2) == len(append_type1)
            input_blocks[0]['input_sequence'] = input_blocks[0]['input_sequence'][:-1] + append_sql
            input_blocks[0]['input_type_0'] = input_blocks[0]['input_type_0'][:-1] + [3] * len(append_sql)

            input_blocks[0]['input_type_1'] = input_blocks[0]['input_type_1'][:-1] + append_type1
            input_blocks[0]['input_type_2'] = input_blocks[0]['input_type_2'][:-1] + append_type2
            input_blocks[0]['separator'][-1] = len(input_blocks[0]['input_sequence'])
            data = DataHolder.prepare_data_dev(input_blocks)

            input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
            join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
            q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, input_blocks = data
            if idx == "ORI":
                t1 = 7
            elif 'VALUE' in idx:
                t1 = 8
            elif 'INTERSECT' in idx:
                t1 = 9
            elif 'UNION' in idx:
                t1 = 10
            elif 'EXCEPT' in idx:
                t1 = 11
            elif 'FROM' in idx:
                t1 = 12
            else:
                raise
            y_type1 = numpy.array([t1], dtype="int32")
            data_list = [
                input_sequence, input_mask, input_type_0, input_type_1, input_type_2,
                join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
                q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, y_type1
            ]
            if input_sequence.shape[1] > 511:
                break
            sql_block = self.construct_simple_sql(data_list)
            sql_result = get_plat_sql(sql_block, q_toks, table_dict, nv_idx, nf_idx, nu_idx, ni_idx, ne_idx, stack)
            sql, stack, nv_idx, nf_idx, nu_idx, ni_idx, ne_idx = sql_result
            comp_sql = comp_sql.replace(idx, sql)

        return comp_sql


def test_bs_comp():
    from config_default import simple_model_config
    model = NL2SQL(simple_model_config)
    model.init_model()
    model.load("../data/save/simple_bert/val015-accuracy_0.006.npz")
    # model.bert.load_partial("../roberta.base/new_roberta_base.npz", "bert_embeddings_word_embeddings_Wemb")
    model.set_dropout_bert(False)
    model.set_dropout(False)

    from config_default import simple_val_config
    from data import DataHolder

    valid_holder = DataHolder(**simple_val_config)
    valid_holder.batch_size = 1
    valid_holder.read_data()
    valid_holder.reset()

    database = parse_table()

    print "Build Sampler..."
    model.build_sampler()
    print "Build Finishied"

    count = 0
    fw = open("../data/test_comp_dev_20200804_nobs.sql", "w")

    for data in valid_holder.get_batch_data():
        count += 1
        print count
        if count == 429:
            print "debug"
        comp_sql = model.construct_comp_sql(data, database)
        fw.write("%s\n" % comp_sql)



def test_bs():
    from config_default import simple_model_config
    model = NL2SQL(simple_model_config)
    model.init_model()
    model.load("../data/save/reason_bert/val002-loss_3.834.npz")
    model.bert.load_partial("../roberta.base/new_roberta_base.npz", "bert_embeddings_word_embeddings_Wemb")
    model.set_dropout_bert(False)
    model.set_dropout(False)

    from config_default import simple_val_config
    from data import DataHolder

    valid_holder = DataHolder(**simple_val_config)

    valid_holder.batch_size = 1
    valid_holder.read_data()
    valid_holder.reset()
    database = parse_table()
    print "Build Sampler..."
    model.build_sampler()
    print "Build Finishied"
    f_dbid = open("../data/dev.db_id")
    f_q = open("../data/clean_dev.question")
    fw = open("../data/test_split_dev.sql", 'w')
    for data in valid_holder.get_batch_data():
        db_id = f_dbid.readline().strip()
        q_toks = f_q.readline().strip().split()
        table_dict = database[db_id]
        input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
        join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
        q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, \
        _, _, \
        _, _, y_type1, _, _, _, _ = data

        ctxs_q, ctxs_t, ctxs_h, ctxs_all, init_state_h, init_state_t, join_embs, prob_logit_co, prob_logit_d, prob_logit_o, prob_logit_c, prob_logit_l, prob_logit_nf = model.inf_encode(
            [input_sequence, input_mask, input_type_0, input_type_1, input_type_2,
             join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
             q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_w_mask, y_type1]
        )

        y_co = numpy.argmax(prob_logit_co, axis=1)
        y_d = numpy.argmax(prob_logit_d, axis=1)
        y_o = numpy.argmax(prob_logit_o, axis=1)
        y_c = numpy.argmax(prob_logit_c, axis=1)
        y_l = numpy.argmax(prob_logit_l, axis=1)
        y_nf = numpy.argmax(prob_logit_nf, axis=1)

        sql_dict = {
            'distinct': y_d,
            'connector': y_co,
            'order': y_o,
            'combine': y_c,
            'limit': y_l,
            'nested_from': y_nf,
        }

        table_seqs, table_scores = model.beam_search_table(
            init_state_t, ctxs_q, q_mask, ctxs_t, t_mask, join_embs,
        )

        table_seq_idx = numpy.argmin(table_scores)
        # table_seq_idx = numpy.argmin(table_scores / numpy.array([len(ts) for ts in table_seqs]))

        sql_dict['from_table'] = table_seqs[table_seq_idx]

        col_list = model.beam_search_column(
            init_state_h, ctxs_all, input_mask, ctxs_q, q_mask, ctxs_h, h_mask
        )

        col_scores = numpy.array([col_dict['score'] for col_dict in col_list])
        col_scores = col_scores / numpy.array([len([col_dict['s_column']]) for col_dict in col_list])

        sql_dict.update(col_list[numpy.argmin(col_scores)])

        sql_result = get_plat_sql(sql_dict, q_toks, table_dict, 0, 0, 0, 0, 0, [])
        sql = sql_result[0]
        fw.write("%s\n" % sql)


if __name__ == '__main__':
    test_bs_comp()
    # test_bs()




















