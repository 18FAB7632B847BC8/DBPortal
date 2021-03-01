#!/usr/bin/env python

from layers_bert import AttentionPoolingLayer, CombineLayer, LayerNormLayer, FeedForwardLayer, dropout
from TransformerDecoder_Pre import TransformerDecoder
from Bert import Bert

from basemodel import BaseModel
import theano
import numpy
from theano import tensor
from time import time


class SimpleNL2SQL(BaseModel):
    def __init__(self, config):
        super(NL2SQL, self).__init__(config)
        self.token_num = config.get('token_num', 21128)
        self.n_in = config.get('n_in', 768)
        self.n_hid = config.get('n_hid', 256)
        self.dropout_rate = config.get("dropout_rate", 0.2)
        self.dropout_rate_bert = config.get("dropout_rate_bert", 0.1)
        self.decoder_num = config.get("decoder_num", 4)
        self.agg_num = config.get('agg_num', 6)
        self.op_num = config.get('op_num', 5)
        self.co_num = config.get('co_num', 3)
        self.head_num = config.get('head_num', 6)
        self.bert_last_num = config.get('bert_last_num', 2)
        self.use_dropout_bert = theano.shared(numpy.float32(config.get('use_dropout_bert', True)))
        self.eps = 1e-12

        self.bert = None
        self.schema_pool = None
        self.global_pool = None
        self.linear_emb_init = None
        self.layer_norm_emb_init = None
        self.decoder = None
        self.linear_enc_h = None
        self.linear_enc_q = None
        self.layer_norm_h = None
        self.layer_norm_q = None
        self.linear_co = None

        # self.inputs_bert = OrderedDict()
        # self.inputs_decoder = OrderedDict()

        self.layers = []

        self.f_init = None
        self.f_next = None

    def set_dropout_bert(self, val):
        self.use_dropout_bert.set_value(numpy.float32(val))

    def init_model(self):
        self.schema_pool = AttentionPoolingLayer(self.n_in * self.bert_last_num,
                                                 self.n_hid, name="schema_pool")
        self.global_pool = AttentionPoolingLayer(self.n_in * self.bert_last_num,
                                                 self.n_hid, name="global_pool")
        self.bert = Bert(self.token_num, type_num=5, bert_last_num=self.bert_last_num,
                         trng=self.trng, dropout_rate=self.dropout_rate_bert, name="bert")
        self.linear_emb_init = FeedForwardLayer(self.n_in * self.bert_last_num,
                                                self.n_hid, name="linear_emb_init")
        self.layer_norm_emb_init = LayerNormLayer(self.n_hid, name="layer_norm_emb_init", epsilon=self.eps)
        self.decoder = TransformerDecoder(self.agg_num, self.op_num, self.n_in * self.bert_last_num, self.decoder_num, self.head_num,
                                          self.n_hid, self.dropout_rate_bert, trng=self.trng,
                                          name="decoder")
        self.linear_enc_h = CombineLayer(self.n_in * self.bert_last_num, self.n_hid, name="combine_layer_enc_h")
        self.linear_enc_q = CombineLayer(self.n_in * self.bert_last_num, self.n_hid, name="combine_layer_enc_q")
        self.layer_norm_h = LayerNormLayer(self.n_hid, name="layer_norm_enc_h", epsilon=self.eps)
        self.layer_norm_q = LayerNormLayer(self.n_hid, name="layer_norm_enc_q", epsilon=self.eps)
        self.linear_co = FeedForwardLayer(self.n_in * self.bert_last_num, self.co_num, name="linear_co")

        self.layers = [
            self.schema_pool, self.global_pool, self.linear_emb_init, self.layer_norm_emb_init,
            self.linear_enc_h, self.linear_enc_q, self.layer_norm_h, self.layer_norm_q,
            self.linear_co
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

        for k, v in self.decoder.tparams.iteritems():
            assert k not in self.tparams, "Duplicated parameter: %s" % k
            self.tparams[k] = v

    def get_encoded_layers(self, bert_output, length_q, s_e_h, mask_h):
        # bert_output: [n_samples, n_timestep, n_dim]
        ctx = bert_output[:, 0, :]  # n_samples, n_dim
        n_emb = ctx.shape[-1]
        max_l_q = length_q.max()
        n_w_s, n_samples, n_s_s = mask_h.shape
        # enc_question = tensor.zeros((n_samples, max_l_q, n_emb), "float32")
        # enc_schema_w = tensor.zeros((n_w_s, n_samples, n_s_s, n_emb), "float32")

        def _step_batch(encoded_layer, length, s_e_h_b):
            enc_question_b = tensor.zeros((max_l_q, n_emb), "float32")
            enc_question_b = tensor.set_subtensor(enc_question_b[:length, :], encoded_layer[1:length + 1, :])
            enc_schema_w_b, _ = theano.scan(fn=_step_column,
                                            outputs_info=None,
                                            sequences=[s_e_h_b],
                                            non_sequences=[encoded_layer, length])  # n_s_s, n_w_s, n_emb
            return enc_question_b, enc_schema_w_b

        def _step_column(s_e_h_b_c, encoded_layer, length):
            start = s_e_h_b_c[0] + length + 1
            end = s_e_h_b_c[1] + 1 + length + 1
            enc_schema_w_b_c = tensor.zeros((n_w_s, n_emb), "float32")
            enc_schema_w_b_c = tensor.set_subtensor(enc_schema_w_b_c[:end - start, :], encoded_layer[start:end, :])

            return enc_schema_w_b_c

        res, _ = theano.scan(fn=_step_batch,
                             outputs_info=None,
                             sequences=[bert_output, length_q, s_e_h]
                             )

        enc_question = res[0]  # n_samples, max_l_q, n_emb
        enc_schema_w = res[1]  # n_samples, n_s_s, n_s_w, n_emb
        enc_schema_w = enc_schema_w.dimshuffle(2, 0, 1, 3)  # n_s_w, n_samples, n_s_s, n_emb
        enc_schema_w = enc_schema_w.reshape([n_w_s, n_samples * n_s_s, n_emb])
        tiled_ctx = tensor.tile(ctx[:, None, :], [1, n_s_s, 1]).reshape([n_samples * n_s_s, n_emb])
        mask_h = mask_h.reshape([n_w_s, n_samples * n_s_s])
        enc_schema = self.schema_pool.get_output(tiled_ctx, enc_schema_w, mask_h)  # n_samples * n_s_s, n_emb
        enc_schema = enc_schema.reshape([n_samples, n_s_s, n_emb])

        return ctx, enc_question, enc_schema

    def get_shifted_emb(self, y, enc_h, emb_init):
        y_flat = y.flatten()  # n_samples * n_timestep
        n_samples = enc_h.shape[0]
        n_h = enc_h.shape[1]
        n_timestep_y = y.shape[1]
        emb_dim = self.n_hid
        y_flat_idx = tensor.tile(tensor.arange(n_samples)[:, None],
                                 [1, n_timestep_y]).reshape(y_flat.shape) * n_h + y_flat
        emb_y = enc_h.reshape([n_samples * n_h, emb_dim])[y_flat_idx]
        emb_y = emb_y.reshape([n_samples, n_timestep_y, emb_dim])
        emb_y_shifted = tensor.zeros_like(emb_y)
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 1:, :], emb_y[:, :-1, :])
        emb_y_shifted = tensor.set_subtensor(emb_y_shifted[:, 0, :], emb_init)

        return emb_y_shifted

    def mask_log_softmax(self, logit, mask):
        # logit: n_samples, n_timestep_target, n_timestep_ctx
        # mask: m_samples, n_timestep_ctx
        alpha = tensor.exp(logit - logit.max(2, keepdims=True))
        alpha = alpha * mask[:, None, :]
        alpha = alpha / alpha.sum(2, keepdims=True)

        return tensor.log(alpha + self.eps)

    def build_model(self):
        x = tensor.matrix('x', dtype='int32')
        x_type = tensor.matrix('x_type', dtype='int32')
        x_mask = tensor.matrix('x_mask', dtype='float32')
        y = tensor.matrix('y', dtype='int32')
        y_mask = tensor.matrix('y_mask', dtype='float32')
        y_agg = tensor.matrix('y_agg', dtype='int32')
        y_agg_mask = tensor.matrix('y_agg_mask', dtype='float32')
        y_op = tensor.matrix('y_op', dtype='int32')
        y_op_mask = tensor.matrix('y_op_mask', dtype='float32')
        len_q = tensor.vector('len_q', dtype='int32')
        mask_q = tensor.matrix('mask_q', dtype='float32')
        s_e_h = tensor.tensor3('s_e_h', dtype='int32')
        mask_h_w = tensor.tensor3('mask_h_w', dtype='float32')
        mask_h_s = tensor.matrix('mask_h_s', dtype='float32')
        y_vs = tensor.matrix('y_wvs', dtype='int32')
        y_vs_mask = tensor.matrix('y_wvs_mask', dtype='float32')
        y_ve = tensor.matrix('y_wve', dtype='int32')
        y_ve_mask = tensor.matrix('y_wve_mask', dtype='float32')
        y_co = tensor.vector('y_co', dtype='int32')

        self.inputs['x'] = x
        self.inputs['x_type'] = x_type
        self.inputs['x_mask'] = x_mask
        self.inputs['y'] = y
        self.inputs['y_mask'] = y_mask
        self.inputs['y_agg'] = y_agg
        self.inputs['y_agg_mask'] = y_agg_mask
        self.inputs['y_op'] = y_op
        self.inputs['y_op_mask'] = y_op_mask
        self.inputs['len_q'] = len_q
        self.inputs['mask_q'] = mask_q
        self.inputs['s_e_h'] = s_e_h
        self.inputs['mask_h_w'] = mask_h_w
        self.inputs['mask_h_s'] = mask_h_s
        self.inputs['y_wvs'] = y_vs
        self.inputs['y_wvs_mask'] = y_vs_mask
        self.inputs['y_wve'] = y_ve
        self.inputs['y_wve_mask'] = y_ve_mask
        self.inputs['y_co'] = y_co

        enc_x = self.bert.get_output(x, x_type, mask=x_mask, in_train=self.use_dropout_bert)
        enc_x = dropout(enc_x, self.trng, self.dropout_rate, self.use_dropout)
        ctx, enc_q, enc_h = self.get_encoded_layers(enc_x, len_q, s_e_h, mask_h_w)
        enc_global = enc_x.dimshuffle(1, 0, 2)[1:, :, :]
        mask_global = x_mask.dimshuffle(1, 0)[1:, :]
        global_emb = self.global_pool.get_output(ctx, enc_global, mask_global)
        global_emb = dropout(global_emb, self.trng, self.dropout_rate, self.use_dropout)
        enc_q = self.linear_enc_q.get_output(ctx, enc_q, activ="linear")
        enc_h = self.linear_enc_h.get_output(ctx, enc_h, activ="linear")
        enc_q = self.layer_norm_q.get_output(enc_q)
        enc_h = self.layer_norm_h.get_output(enc_h)
        emb_init = self.linear_emb_init.get_output(global_emb, activ="linear")
        emb_init = self.layer_norm_emb_init.get_output(emb_init)

        emb_y_shifted = self.get_shifted_emb(y, enc_h, emb_init)

        ctx_y, logit, logit_agg, logit_op, logit_vs, logit_ve = self.decoder.get_output(emb_y_shifted, y_mask,
                                                                                        enc_x, x_mask,
                                                                                        enc_q, enc_h, mask_q, mask_h_s,
                                                                                        in_train=self.use_dropout_bert)
        # ctx_y: n_samples, n_timestep, n_dim
        # logit: n_samples, n_timestep, n_h
        logit_shape = logit.shape
        log_probs = -tensor.nnet.logsoftmax(logit.reshape([logit_shape[0] * logit_shape[1], logit_shape[2]]))
        y_flat = y.flatten()   # n_samples * n_timestep
        y_flat_idx = tensor.arange(y_flat.shape[0]) * logit_shape[2] + y_flat

        cost_y = log_probs.flatten()[y_flat_idx]
        cost_y = cost_y.reshape([logit_shape[0], logit_shape[1]])
        cost_y = (cost_y * y_mask).sum(1)

        logit_agg_shape = logit_agg.shape
        log_probs_agg = -tensor.nnet.logsoftmax(logit_agg.reshape([logit_agg_shape[0] * logit_agg_shape[1],
                                                                   logit_agg_shape[2]]))
        y_agg_flat = y_agg.flatten()
        y_agg_flat_idx = tensor.arange(y_agg_flat.shape[0]) * logit_agg_shape[2] + y_agg_flat
        cost_agg = log_probs_agg.flatten()[y_agg_flat_idx]
        cost_agg = cost_agg.reshape([logit_agg_shape[0], logit_agg_shape[1]])
        cost_agg = (cost_agg * y_agg_mask).sum(1)

        logit_op_shape = logit_op.shape
        log_probs_op = -tensor.nnet.logsoftmax(logit_op.reshape([logit_op_shape[0] * logit_op_shape[1],
                                                                 logit_op_shape[2]]))
        y_op_flat = y_op.flatten()
        y_op_flat_idx = tensor.arange(y_op_flat.shape[0]) * logit_op_shape[2] + y_op_flat
        cost_op = log_probs_op.flatten()[y_op_flat_idx]
        cost_op = cost_op.reshape([logit_op_shape[0], logit_op_shape[1]])
        cost_op = (cost_op * y_op_mask).sum(1)

        logit_vs_shape = logit_vs.shape
        log_probs_vs = -tensor.nnet.logsoftmax(logit_vs.reshape([logit_vs_shape[0] * logit_vs_shape[1],
                                                                 logit_vs_shape[2]]))
        y_vs_flat = y_vs.flatten()
        y_vs_flat_idx = tensor.arange(y_vs_flat.shape[0]) * logit_vs_shape[2] + y_vs_flat
        cost_vs = log_probs_vs.flatten()[y_vs_flat_idx]
        cost_vs = cost_vs.reshape([logit_vs_shape[0], logit_vs_shape[1]])
        cost_vs = (cost_vs * y_vs_mask).sum(1)

        logit_ve_shape = logit_ve.shape
        log_probs_ve = -tensor.nnet.logsoftmax(logit_ve.reshape([logit_ve_shape[0] * logit_ve_shape[1],
                                                                 logit_ve_shape[2]]))

        y_ve_flat = y_ve.flatten()
        y_ve_flat_idx = tensor.arange(y_ve_flat.shape[0]) * logit_ve_shape[2] + y_ve_flat
        cost_ve = log_probs_ve.flatten()[y_ve_flat_idx]
        cost_ve = cost_ve.reshape([logit_ve_shape[0], logit_ve_shape[1]])
        cost_ve = (cost_ve * y_ve_mask).sum(1)

        logit_co = self.linear_co.get_output(global_emb, activ="linear")   # n_samples, 3
        log_probs_co = -tensor.nnet.logsoftmax(logit_co)
        y_co_flat = y_co.flatten()
        y_co_flat_idx = tensor.arange(y_co_flat.shape[0]) * logit_co.shape[1] + y_co_flat
        cost_co = log_probs_co.flatten()[y_co_flat_idx]  # n_samples

        cost = cost_y + cost_agg + cost_op + cost_vs + cost_ve + cost_co
        self.f_log_probs = theano.function(list(self.inputs.values()), {'cost': cost,
                                                                        'cost_y': cost_y,
                                                                        'cost_agg': cost_agg,
                                                                        'cost_op': cost_op,
                                                                        'cost_vs': cost_vs,
                                                                        'cost_ve': cost_ve,
                                                                        'cost_co': cost_co})

        return cost
    """
    def build_sampler(self):
        x = tensor.matrix('x', dtype='int32')
        x_type = tensor.matrix('x_type', dtype='int32')
        x_mask = tensor.matrix('x_mask', dtype='float32')
        len_q = tensor.vector('len_q', dtype='int32')
        s_e_h = tensor.tensor3('s_e_h', dtype='int32')
        mask_h_w = tensor.tensor3('mask_h_w', dtype='float32')
        inputs_init = [x, x_type, x_mask, len_q, s_e_h, mask_h_w]
        enc_x = self.bert.get_output(x, x_type, mask=x_mask, in_train=self.use_dropout_bert)
        ctx, enc_q, enc_h = self.get_encoded_layers(enc_x, len_q, s_e_h, mask_h_w)
        enc_global = enc_x.dimshuffle(1, 0, 2)[1:, :, :]
        mask_global = x_mask.dimshuffle(1, 0)[1:, :]
        global_emb = self.global_pool.get_output(ctx, enc_global, mask_global)
        enc_q = self.linear_enc_q.get_output(ctx, enc_q, activ="linear")
        enc_h = self.linear_enc_h.get_output(ctx, enc_h, activ="linear")
        enc_q = self.layer_norm_q.get_output(enc_q)
        enc_h = self.layer_norm_h.get_output(enc_h)
        emb_init = self.linear_emb_init.get_output(global_emb, activ="linear")
        emb_init = self.layer_norm_emb_init.get_output(emb_init)

        logit_co = self.linear_co.get_output(global_emb, activ="linear")

        y_co = tensor.argmax(logit_co, axis=1, keepdims=False)

        outs = [emb_init, enc_q, enc_h, y_co, enc_x]

        self.f_init = theano.function(inputs_init, outs, name='f_init')

        y = tensor.vector('y_sampler', dtype='int64')
        init_state = tensor.matrix('init_state', dtype='float32')
        mask_q = tensor.matrix('mask_q', dtype='float32')
        mask_h_s = tensor.matrix('mask_h_s', dtype='float32')
        n_samples = y.shape[0]
        emb_y = tensor.switch(y[:, None] < 0,
                              tensor.alloc(0., 1, self.n_hid),
                              enc_h[tensor.arange(n_samples), y]
                              )

        next_state, logit, logit_agg, logit_op, logit_vs, logit_ve = self.decoder.get_one_step(
            emb_y, init_state, enc_x, x_mask, enc_q, enc_h, mask_q, mask_h_s, in_train=self.use_dropout_bert,
            final=True
        )

        next_log_probs = tensor.nnet.logsoftmax(logit[:, -1, :])
        next_log_probs_agg = tensor.nnet.logsoftmax(logit_agg[:, -1, :])
        next_log_probs_op = tensor.nnet.logsoftmax(logit_op[:, -1, :])
        next_log_probs_vs = tensor.nnet.logsoftmax(logit_vs[:, -1, :])
        next_log_probs_ve = tensor.nnet.logsoftmax(logit_ve[:, -1, :])

        inputs_next = [init_state, enc_q, enc_h, y, mask_q, mask_h_s, enc_x, x_mask]
        outs_next = [next_state,
                     next_log_probs, next_log_probs_agg, next_log_probs_op, next_log_probs_vs, next_log_probs_ve]

        self.f_next = theano.function(inputs_next, outs_next, name="f_next")

    def beam_search(self, x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s, maxlen=10, beam_size=1):
        final_score = []
        final_sc = []
        final_wc = []
        final_op = []
        final_agg = []
        final_vs = []
        final_ve = []

        hyp_samples = [[]]
        hyp_scores = numpy.zeros(1, dtype='float32')
        hyp_aggs = [[]]
        hyp_ops = [[]]
        hyp_vss = [[]]
        hyp_ves = [[]]

        ret = self.f_init(x, x_type, x_mask, len_q, s_e_h, mask_h_w)
        emb_init, enc_q, enc_h, y_co, enc_x = ret[0], ret[1], ret[2], ret[3], ret[4]
        next_w = -1 * numpy.ones((1,)).astype('int64')
        live_beam = beam_size
        next_state = emb_init  # n_samples, n_emb
        tiled_enc_q = numpy.tile(enc_q, [1, 1, 1])  # n_samples, n_timestep, n_emb
        tiled_enc_h = numpy.tile(enc_h, [1, 1, 1])  # n_samples, n_timestep, n_emb
        tiled_mask_q = numpy.tile(mask_q, [1, 1])  # n_samples, n_timestep
        tiled_mask_h_s = numpy.tile(mask_h_s, [1, 1])  # n_samples, n_timestep
        tiled_enc_x = numpy.tile(enc_x, [1, 1, 1])   # n_samples, n_timestep, n_emb
        tiled_mask_x = numpy.tile(x_mask, [1, 1])
        y_co = y_co[0]

        for t in xrange(maxlen):
            next_state, next_log_probs, next_log_probs_agg, next_log_probs_op, next_log_probs_vs, next_log_probs_ve = self.f_next(
                next_state, tiled_enc_q, tiled_enc_h, next_w, tiled_mask_q, tiled_mask_h_s, tiled_enc_x, tiled_mask_x)

            h_score = next_log_probs
            cand_scores = hyp_scores[:, None] - h_score
            cand_scores.shape = cand_scores.size
            ranks_flat = cand_scores.argpartition(live_beam - 1)[:live_beam]
            costs = cand_scores[ranks_flat]
            live_beam = 0
            new_hyp_scores = []
            new_hyp_samples = []
            new_hyp_aggs = []
            new_hyp_ops = []
            new_hyp_vss = []
            new_hyp_ves = []
            hyp_states = []

            trans_idxs = ranks_flat // next_log_probs.shape[1]
            word_idxs = ranks_flat % next_log_probs.shape[1]

            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_agg = hyp_aggs[ti] + [next_log_probs_agg[ti].argmax()]
                new_op = hyp_ops[ti] + [next_log_probs_op[ti].argmax()]
                new_vs = hyp_vss[ti] + [next_log_probs_vs[ti].argmax()]
                new_ve = hyp_ves[ti] + [next_log_probs_ve[ti].argmax()]

                if new_hyp.count(0) == 2:
                    where_s = new_hyp.index(0) + 1
                    final_sc.append(new_hyp[:where_s])
                    final_wc.append(new_hyp[where_s:])
                    final_op.append(new_op[where_s:])
                    final_agg.append(new_agg[:where_s])
                    final_vs.append(new_vs[where_s:])
                    final_ve.append(new_ve[where_s:])
                    final_score.append(costs[idx])
                else:
                    new_hyp_samples.append(new_hyp)
                    new_hyp_scores.append(costs[idx])
                    new_hyp_ops.append(new_op)
                    new_hyp_aggs.append(new_agg)
                    new_hyp_vss.append(new_vs)
                    new_hyp_ves.append(new_ve)
                    hyp_states.append(next_state[ti])
                    live_beam += 1

            hyp_scores = numpy.array(new_hyp_scores, dtype='float32')
            hyp_samples = new_hyp_samples
            hyp_ops = new_hyp_ops
            hyp_aggs = new_hyp_aggs
            hyp_vss = new_hyp_vss
            hyp_ves = new_hyp_ves

            if live_beam == 0:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states, dtype='float32')  # n_samples, n_emb
            tiled_enc_q = numpy.tile(enc_q, [live_beam, 1, 1])  # n_samples, n_timestep, n_emb
            tiled_enc_h = numpy.tile(enc_h, [live_beam, 1, 1])  # n_samples, n_timestep, n_emb
            tiled_mask_q = numpy.tile(mask_q, [live_beam, 1])  # n_samples, n_timestep
            tiled_mask_h_s = numpy.tile(mask_h_s, [live_beam, 1])  # n_samples, n_timestep
            tiled_enc_x = numpy.tile(enc_x, [live_beam, 1, 1])
            tiled_mask_x = numpy.tile(x_mask, [live_beam, 1])
        for idx in range(live_beam):
            new_hyp = hyp_samples[idx]
            where_s = new_hyp.index(0) + 1
            final_score.append(hyp_scores[idx])
            final_sc.append(new_hyp[:where_s])
            final_wc.append(new_hyp[where_s:])
            final_op.append(hyp_ops[idx])
            final_agg.append(hyp_aggs[idx])
            final_vs.append(hyp_vss[idx])
            final_ve.append(hyp_ves[idx])

        min_score = numpy.array(final_score).argmin()

        y_sc = numpy.array(final_sc[min_score])
        y_wc = numpy.array(final_wc[min_score])
        y_agg = numpy.array(final_agg[min_score])
        y_op = numpy.array(final_op[min_score])
        y_vs = numpy.array(final_vs[min_score])
        y_ve = numpy.array(final_ve[min_score])

        return y_sc, y_wc, y_agg, y_op, y_vs, y_ve, y_co

    """
    def build_sampler(self):
        x = tensor.matrix('x', dtype='int32')
        x_type = tensor.matrix('x_type', dtype='int32')
        x_mask = tensor.matrix('x_mask', dtype='float32')
        len_q = tensor.vector('len_q', dtype='int32')
        s_e_h = tensor.tensor3('s_e_h', dtype='int32')
        mask_h_w = tensor.tensor3('mask_h_w', dtype='float32')
        inputs_init = [x, x_type, x_mask, len_q, s_e_h, mask_h_w]
        enc_x = self.bert.get_output(x, x_type, mask=x_mask, in_train=self.use_dropout_bert)
        enc_x = dropout(enc_x, self.trng, self.dropout_rate, self.use_dropout)
        ctx, enc_q, enc_h = self.get_encoded_layers(enc_x, len_q, s_e_h, mask_h_w)
        enc_global = enc_x.dimshuffle(1, 0, 2)[1:, :, :]
        mask_global = x_mask.dimshuffle(1, 0)[1:, :]
        global_emb = self.global_pool.get_output(ctx, enc_global, mask_global)
        enc_q = self.linear_enc_q.get_output(ctx, enc_q, activ="linear")
        enc_h = self.linear_enc_h.get_output(ctx, enc_h, activ="linear")
        enc_q = self.layer_norm_q.get_output(enc_q)
        enc_h = self.layer_norm_h.get_output(enc_h)
        emb_init = self.linear_emb_init.get_output(global_emb, activ="linear")
        emb_init = self.layer_norm_emb_init.get_output(emb_init)

        logit_co = self.linear_co.get_output(global_emb, activ="linear")

        y_co = tensor.argmax(logit_co, axis=1, keepdims=False)
        outs_init = [enc_q, enc_h, emb_init, enc_x, y_co]

        self.f_init = theano.function(inputs_init, outs_init, name='f_init')

        y = tensor.matrix('y', dtype='int32')
        y_mask = tensor.matrix('y_mask', dtype='float32')
        mask_q = tensor.matrix('mask_q', dtype='float32')
        mask_h_s = tensor.matrix('mask_h_s', dtype='float32')

        inputs_next = [emb_init, enc_q, enc_h, y, y_mask, mask_q, mask_h_s, enc_x, x_mask]

        emb_y_shifted = tensor.switch(y.sum() < 0,
                                      emb_init,  # n_samples, n_timestep, n_emb
                                      self.get_shifted_emb(y, enc_h, emb_init)
                                      )

        ctx_y, logit, logit_agg, logit_op, logit_vs, logit_ve = self.decoder.get_output(emb_y_shifted, y_mask,
                                                                                        enc_x, x_mask,
                                                                                        enc_q, enc_h, mask_q, mask_h_s,
                                                                                        in_train=self.use_dropout_bert)

        prob_logit = tensor.nnet.logsoftmax(logit[:, -1, :])
        prob_logit_agg = tensor.nnet.logsoftmax(logit_agg[:, -1, :])
        prob_logit_op = tensor.nnet.logsoftmax(logit_op[:, -1, :])
        prob_logit_vs = tensor.nnet.logsoftmax(logit_vs[:, -1, :])
        prob_logit_ve = tensor.nnet.logsoftmax(logit_ve[:, -1, :])

        outs_next = [ctx_y, prob_logit, prob_logit_agg, prob_logit_op, prob_logit_vs, prob_logit_ve]
        self.f_next = theano.function(inputs_next, outs_next, name="f_next")

    def beam_search(self, x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s, maxlen=10, beam_size=2):
        final_score = []
        final_sc = []
        final_wc = []
        final_op = []
        final_agg = []
        final_vs = []
        final_ve = []

        hyp_samples = [[]]
        hyp_scores = numpy.zeros(1, dtype='float32')
        hyp_aggs = [[]]
        hyp_ops = [[]]
        hyp_vss = [[]]
        hyp_ves = [[]]

        ret = self.f_init(x, x_type, x_mask, len_q, s_e_h, mask_h_w)
        enc_q, enc_h, emb_init, enc_x, y_co = ret[0], ret[1], ret[2], ret[3], ret[4]
        y_co = y_co[0]
        y = -1 * numpy.ones((1, 1), dtype='int32')
        live_beam = beam_size

        tiled_emb_init = numpy.tile(emb_init, [1, 1])  # n_samples, n_emb
        tiled_enc_q = numpy.tile(enc_q, [1, 1, 1])  # n_samples, n_timestep, n_emb
        tiled_enc_h = numpy.tile(enc_h, [1, 1, 1])  # n_samples, n_timestep, n_emb
        tiled_mask_q = numpy.tile(mask_q, [1, 1])  # n_samples, n_timestep
        tiled_mask_h_s = numpy.tile(mask_h_s, [1, 1])  # n_samples, n_timestep
        tiled_enc_x = numpy.tile(enc_x, [1, 1, 1])
        tiled_mask_x = numpy.tile(x_mask, [1, 1])

        for t in xrange(maxlen):
            y_mask = numpy.ones_like(y, dtype='float32')

            ctx_y, prob_logit, prob_logit_agg, prob_logit_op, prob_logit_vs, prob_logit_ve = self.f_next(
                tiled_emb_init, tiled_enc_q, tiled_enc_h, y, y_mask, tiled_mask_q, tiled_mask_h_s,
                tiled_enc_x, tiled_mask_x)

            next_log_probs = prob_logit
            next_log_probs_agg = prob_logit_agg
            next_log_probs_op = prob_logit_op
            next_log_probs_vs = prob_logit_vs
            next_log_probs_ve = prob_logit_ve
            h_score = next_log_probs
            cand_scores = hyp_scores[:, None] - h_score
            cand_scores.shape = cand_scores.size
            ranks_flat = cand_scores.argpartition(live_beam - 1)[:live_beam]
            costs = cand_scores[ranks_flat]
            live_beam = 0
            new_hyp_scores = []
            new_hyp_samples = []
            new_hyp_aggs = []
            new_hyp_ops = []
            new_hyp_vss = []
            new_hyp_ves = []

            trans_idxs = ranks_flat // next_log_probs.shape[1]
            word_idxs = ranks_flat % next_log_probs.shape[1]

            for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                # Form the new hypothesis by appending new word to the left hyp
                new_hyp = hyp_samples[ti] + [wi]
                new_agg = hyp_aggs[ti] + [next_log_probs_agg[ti].argmax()]
                new_op = hyp_ops[ti] + [next_log_probs_op[ti].argmax()]
                new_vs = hyp_vss[ti] + [next_log_probs_vs[ti].argmax()]
                new_ve = hyp_ves[ti] + [next_log_probs_ve[ti].argmax()]
                if new_hyp.count(0) == 2:
                    where_s = new_hyp.index(0) + 1
                    final_sc.append(new_hyp[:where_s])
                    final_wc.append(new_hyp[where_s:])
                    final_op.append(new_op[where_s:])
                    final_agg.append(new_agg[:where_s])
                    final_vs.append(new_vs[where_s:])
                    final_ve.append(new_ve[where_s:])
                    final_score.append(costs[idx])
                else:
                    new_hyp_samples.append(new_hyp)
                    new_hyp_scores.append(costs[idx])
                    new_hyp_ops.append(new_op)
                    new_hyp_aggs.append(new_agg)
                    new_hyp_vss.append(new_vs)
                    new_hyp_ves.append(new_ve)
                    live_beam += 1

            hyp_scores = numpy.array(new_hyp_scores, dtype='float32')
            hyp_samples = new_hyp_samples
            hyp_ops = new_hyp_ops
            hyp_aggs = new_hyp_aggs
            hyp_vss = new_hyp_vss
            hyp_ves = new_hyp_ves

            if live_beam == 0:
                break

            y = []
            for sample in hyp_samples:
                y.append(sample + [0])
            y = numpy.array(y, dtype='int32')
            tiled_emb_init = numpy.tile(emb_init, [live_beam, 1])  # n_samples, n_emb
            tiled_enc_q = numpy.tile(enc_q, [live_beam, 1, 1])  # n_samples, n_timestep, n_emb
            tiled_enc_h = numpy.tile(enc_h, [live_beam, 1, 1])  # n_samples, n_timestep, n_emb
            tiled_mask_q = numpy.tile(mask_q, [live_beam, 1])  # n_samples, n_timestep
            tiled_mask_h_s = numpy.tile(mask_h_s, [live_beam, 1])  # n_samples, n_timestep
            tiled_enc_x = numpy.tile(enc_x, [live_beam, 1, 1])
            tiled_mask_x = numpy.tile(x_mask, [live_beam, 1])

        for idx in range(live_beam):
            new_hyp = hyp_samples[idx]
            try:
                where_s = new_hyp.index(0) + 1
            except:
                where_s = 2
            final_score.append(hyp_scores[idx])
            final_sc.append(new_hyp[:where_s])
            final_wc.append(new_hyp[where_s:])
            final_op.append(hyp_ops[idx])
            final_agg.append(hyp_aggs[idx])
            final_vs.append(hyp_vss[idx])
            final_ve.append(hyp_ves[idx])

        min_score = numpy.array(final_score).argmin()

        y_sc = numpy.array(final_sc[min_score])
        y_wc = numpy.array(final_wc[min_score])
        y_agg = numpy.array(final_agg[min_score])
        y_op = numpy.array(final_op[min_score])
        y_vs = numpy.array(final_vs[min_score])
        y_ve = numpy.array(final_ve[min_score])
        return y_sc, y_wc, y_agg, y_op, y_vs, y_ve, y_co

    def multi_beam_search(self, x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s, maxlen=10, beam_size=2):
        ret = self.f_init(x, x_type, x_mask, len_q, s_e_h, mask_h_w)
        enc_q_, enc_h_, emb_init_, enc_x_, y_co_ = ret[0], ret[1], ret[2], ret[3], ret[4]

        n_samples = x.shape[0]
        return_list = []

        for sample_id in range(n_samples):
            final_score = []
            final_sc = []
            final_wc = []
            final_op = []
            final_agg = []
            final_vs = []
            final_ve = []

            hyp_samples = [[]]
            hyp_scores = numpy.zeros(1, dtype='float32')
            hyp_aggs = [[]]
            hyp_ops = [[]]
            hyp_vss = [[]]
            hyp_ves = [[]]

            enc_q = enc_q_[sample_id][None, :, :]
            enc_h = enc_h_[sample_id][None, :, :]
            enc_x = enc_x_[sample_id][None, :, :]
            emb_init = emb_init_[sample_id][None, :]
            y_co = y_co_[sample_id]

            y = -1 * numpy.ones((1, 1), dtype='int32')
            live_beam = beam_size

            tiled_emb_init = numpy.tile(emb_init, [1, 1])  # n_samples, n_emb
            tiled_enc_q = numpy.tile(enc_q, [1, 1, 1])  # n_samples, n_timestep, n_emb
            tiled_enc_h = numpy.tile(enc_h, [1, 1, 1])  # n_samples, n_timestep, n_emb
            tiled_mask_q = numpy.tile(mask_q, [1, 1])  # n_samples, n_timestep
            tiled_mask_h_s = numpy.tile(mask_h_s, [1, 1])  # n_samples, n_timestep
            tiled_enc_x = numpy.tile(enc_x, [1, 1, 1])
            tiled_mask_x = numpy.tile(x_mask, [1, 1])

            for t in xrange(maxlen):
                y_mask = numpy.ones_like(y, dtype='float32')

                ctx_y, prob_logit, prob_logit_agg, prob_logit_op, prob_logit_vs, prob_logit_ve = self.f_next(
                    tiled_emb_init, tiled_enc_q, tiled_enc_h, y, y_mask, tiled_mask_q, tiled_mask_h_s,
                    tiled_enc_x, tiled_mask_x)

                next_log_probs = prob_logit
                next_log_probs_agg = prob_logit_agg
                next_log_probs_op = prob_logit_op
                next_log_probs_vs = prob_logit_vs
                next_log_probs_ve = prob_logit_ve
                h_score = next_log_probs
                cand_scores = hyp_scores[:, None] - h_score
                cand_scores.shape = cand_scores.size
                ranks_flat = cand_scores.argpartition(live_beam - 1)[:live_beam]
                costs = cand_scores[ranks_flat]
                live_beam = 0
                new_hyp_scores = []
                new_hyp_samples = []
                new_hyp_aggs = []
                new_hyp_ops = []
                new_hyp_vss = []
                new_hyp_ves = []

                trans_idxs = ranks_flat // next_log_probs.shape[1]
                word_idxs = ranks_flat % next_log_probs.shape[1]

                for idx, [ti, wi] in enumerate(zip(trans_idxs, word_idxs)):
                    # Form the new hypothesis by appending new word to the left hyp
                    new_hyp = hyp_samples[ti] + [wi]
                    new_agg = hyp_aggs[ti] + [next_log_probs_agg[ti].argmax()]
                    new_op = hyp_ops[ti] + [next_log_probs_op[ti].argmax()]
                    new_vs = hyp_vss[ti] + [next_log_probs_vs[ti].argmax()]
                    new_ve = hyp_ves[ti] + [next_log_probs_ve[ti].argmax()]
                    if new_hyp.count(0) == 2:
                        where_s = new_hyp.index(0) + 1
                        final_sc.append(new_hyp[:where_s])
                        final_wc.append(new_hyp[where_s:])
                        final_op.append(new_op[where_s:])
                        final_agg.append(new_agg[:where_s])
                        final_vs.append(new_vs[where_s:])
                        final_ve.append(new_ve[where_s:])
                        final_score.append(costs[idx])
                    else:
                        new_hyp_samples.append(new_hyp)
                        new_hyp_scores.append(costs[idx])
                        new_hyp_ops.append(new_op)
                        new_hyp_aggs.append(new_agg)
                        new_hyp_vss.append(new_vs)
                        new_hyp_ves.append(new_ve)
                        live_beam += 1

                hyp_scores = numpy.array(new_hyp_scores, dtype='float32')
                hyp_samples = new_hyp_samples
                hyp_ops = new_hyp_ops
                hyp_aggs = new_hyp_aggs
                hyp_vss = new_hyp_vss
                hyp_ves = new_hyp_ves

                if live_beam == 0:
                    break

                y = []
                for sample in hyp_samples:
                    y.append(sample + [0])
                y = numpy.array(y, dtype='int32')
                tiled_emb_init = numpy.tile(emb_init, [live_beam, 1])  # n_samples, n_emb
                tiled_enc_q = numpy.tile(enc_q, [live_beam, 1, 1])  # n_samples, n_timestep, n_emb
                tiled_enc_h = numpy.tile(enc_h, [live_beam, 1, 1])  # n_samples, n_timestep, n_emb
                tiled_mask_q = numpy.tile(mask_q, [live_beam, 1])  # n_samples, n_timestep
                tiled_mask_h_s = numpy.tile(mask_h_s, [live_beam, 1])  # n_samples, n_timestep
                tiled_enc_x = numpy.tile(enc_x, [live_beam, 1, 1])
                tiled_mask_x = numpy.tile(x_mask, [live_beam, 1])

            for idx in range(live_beam):
                new_hyp = hyp_samples[idx]
                try:
                    where_s = new_hyp.index(0) + 1
                except:
                    where_s = 2
                final_score.append(hyp_scores[idx])
                final_sc.append(new_hyp[:where_s])
                final_wc.append(new_hyp[where_s:])
                final_op.append(hyp_ops[idx])
                final_agg.append(hyp_aggs[idx])
                final_vs.append(hyp_vss[idx])
                final_ve.append(hyp_ves[idx])

            min_score = numpy.array(final_score).argmin()

            y_sc = numpy.array(final_sc[min_score])
            y_wc = numpy.array(final_wc[min_score])
            y_agg = numpy.array(final_agg[min_score])
            y_op = numpy.array(final_op[min_score])
            y_vs = numpy.array(final_vs[min_score])
            y_ve = numpy.array(final_ve[min_score])

            return_list.append([y_sc, y_wc, y_agg, y_op, y_vs, y_ve, y_co])

        return return_list







