import theano
from theano import tensor
from layers_bert import AttentionPoolingLayer, LayerNormLayer, EmbeddingLayer, SimpleCombineLayer, dropout
from collections import OrderedDict


class NL2SQLEncoder(object):
    def __init__(self, emb_dim, hid_dim, num_foreign=32, forigen_dim=128, scale="xavier", trng=None,
                 name="encoder"):
        self.emb_dim = emb_dim
        self.forigen_dim = forigen_dim
        self.name = name
        self.num_foreign = num_foreign
        self.tparams = OrderedDict()
        self.trng = trng

        self.join_emb_layer = EmbeddingLayer(num_foreign, forigen_dim, scale=scale, name=self.name + "_join_emb_layer")
        self.q_att_layer = AttentionPoolingLayer(emb_dim, hid_dim, scale=scale, name=self.name + "_q_att_layer")
        self.p_att_layer = AttentionPoolingLayer(emb_dim, hid_dim, scale=scale, name=self.name + "_p_att_layer")
        self.d_att_layer = AttentionPoolingLayer(emb_dim, hid_dim, scale=scale, name=self.name + "_d_att_layer")
        self.t_w_att_layer = AttentionPoolingLayer(emb_dim, hid_dim, scale=scale, name=self.name + "_t_w_att_layer")
        self.h_w_att_layer = AttentionPoolingLayer(emb_dim, hid_dim, scale=scale, name=self.name + "_h_w_att_layer")
        self.t_h_att_layer = AttentionPoolingLayer(emb_dim, hid_dim, scale=scale, name=self.name + "_t_h_att_layer")

        self.combine_t = SimpleCombineLayer(emb_dim, emb_dim, scale=scale, name=self.name + "_combine_table")
        self.combine_c = SimpleCombineLayer(emb_dim, emb_dim, scale=scale, name=self.name + "_combine_column")
        self.combine_query = SimpleCombineLayer(emb_dim, emb_dim, scale=scale, name=self.name + "_combine_query")

        for layer in [self.join_emb_layer, self.q_att_layer, self.p_att_layer, self.d_att_layer, self.t_h_att_layer,
                      self.t_w_att_layer, self.h_w_att_layer, self.combine_c, self.combine_t, self.combine_query]:
            if layer is None:
                raise KeyError("layer is None")
            for k, v in layer.tparams.iteritems():
                p_name = layer.name + '_' + k
                assert p_name not in self.tparams, "Duplicated parameter: %s" % p_name
                self.tparams[p_name] = v

    def get_output(self, encoded_layers, emb_type, join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx,
                   q_mask, t_mask, t_w_mask, t_h_mask, h_w_mask, p_mask, dropout_rate, use_dropout):

        # encoded_layers: n_batch, n_seq, emb_dim
        # emb_type: n_batch, n_seq, emb_dim
        # join_table: n_batch, n_table, n_join
        # join_mask: n_batch, n_table, n_join
        # separator: n_batch, 8
        # s_e_h: n_batch, n_column, 2
        # s_e_t: n_batch, n_table, 2
        # s_e_t_h: n_batch, n_table - 1, 2
        # h2t_idx: n_batch, n_column - 2
        # q_mask: n_batch, q_len
        # t_mask: n_batch, n_table
        # t_h_mask: n_batch, n_table - 1, n_table_h
        # t_w_mask: n_batch, n_table, n_table_w
        # h_w_mask: n_batch, n_column, n_column_w
        # p_mask: n_batch, p_len

        n_batch = encoded_layers.shape[0]
        q_len = q_mask.shape[1]
        p_len = p_mask.shape[1]
        t_len = t_mask.shape[1]
        h_len = h_w_mask.shape[1]
        t_len_w = t_w_mask.shape[2]
        h_len_w = h_w_mask.shape[2]
        t_len_h = t_h_mask.shape[2]
        join_len = join_table.shape[2]

        dim = self.emb_dim

        global_ctx = encoded_layers[:, 0, :]

        join_embs = self.join_emb_layer.get_output(join_table).reshape([n_batch, t_len, join_len, self.forigen_dim])
        join_embs = (join_embs * join_mask[:, :, :, None]).sum(2) / self.num_foreign

        def _step_batch(encoded_layer, s_e_t_b, s_e_h_b, sep_b):
            # encoded_layer: n_seq, emb_dim
            # s_e_h_b: n_column, 2
            # s_e_t_b: n_table, 2
            # sep_b: 8
            q_s, q_e, p_s, p_e = sep_b[0], sep_b[1], sep_b[6], sep_b[7]
            ctx_q_b = tensor.zeros((q_len, dim), 'float32')
            ctx_q_b = tensor.set_subtensor(ctx_q_b[:q_e - q_s, :], encoded_layer[q_s:q_e, :])
            ctx_p_b = tensor.zeros((p_len, dim), 'float32')
            ctx_p_b = tensor.set_subtensor(ctx_p_b[:p_e - p_s, :], encoded_layer[p_s:p_e, :])

            ctx_t_b, _ = theano.scan(fn=_step_inner,
                                     outputs_info=None,
                                     sequences=[s_e_t_b],
                                     non_sequences=[encoded_layer, t_len_w]
                                     )  # n_table, n_table_w, dim

            ctx_h_b, _ = theano.scan(fn=_step_inner,
                                     outputs_info=None,
                                     sequences=[s_e_h_b],
                                     non_sequences=[encoded_layer, h_len_w]
                                     )  # n_column, n_column_w, dim

            return ctx_q_b, ctx_t_b, ctx_h_b, ctx_p_b

        def _step_batch_single(encoded_layer, s_e_t_h_b):
            ctx_th_b, _ = theano.scan(fn=_step_inner,
                                      outputs_info=None,
                                      sequences=[s_e_t_h_b],
                                      non_sequences=[encoded_layer, t_len_h]
                                      )
            return ctx_th_b

        def _step_inner(s_e, encoded_layer, l):
            s = s_e[0]
            e = s_e[1]
            ctx_ = tensor.zeros((l, dim), "float32")
            ctx_ = tensor.set_subtensor(ctx_[:e - s, :], encoded_layer[s:e, :])

            return ctx_

        encoded_layers = encoded_layers + emb_type

        res, _ = theano.scan(fn=_step_batch,
                             outputs_info=None,
                             sequences=[encoded_layers, s_e_t, s_e_h, separator]
                             )
        enc_q = res[0]  # n_batch, q_len, emb_dim
        enc_t_w = res[1]  # n_batch, t_len, n_t_w, emb_dim
        enc_h_w = res[2]  # n_batch, h_len, n_h_w, emb_dim
        enc_p = res[3]  # n_batch, p_len, emb_dim

        emb_q = self.q_att_layer.get_output(global_ctx, dropout(enc_q, self.trng, dropout_rate, use_dropout), q_mask)  # n_batch, emb_dim
        emb_p = self.p_att_layer.get_output(global_ctx, dropout(enc_p, self.trng, dropout_rate, use_dropout), p_mask)  # n_batch, emb_dim

        enc_t_w = enc_t_w.reshape([n_batch * t_len, t_len_w, dim])
        tiled_global_ctx = tensor.tile(global_ctx[:, None, :], [1, t_len, 1]).reshape([n_batch * t_len, dim])
        embs_t = self.t_w_att_layer.get_output(tiled_global_ctx, enc_t_w, t_w_mask.reshape([n_batch * t_len, t_len_w]))
        embs_t = embs_t.reshape([n_batch, t_len, dim])

        emb_empty_t = embs_t[:, 0, :]

        enc_h_w = enc_h_w.reshape([n_batch * h_len, h_len_w, dim])
        tiled_global_ctx = tensor.tile(global_ctx[:, None, :], [1, h_len, 1]).reshape([n_batch * h_len, dim])
        embs_h = self.h_w_att_layer.get_output(tiled_global_ctx, enc_h_w, h_w_mask.reshape([n_batch * h_len, h_len_w]))
        embs_h = embs_h.reshape([n_batch, h_len, dim])

        enc_t_h, _ = theano.scan(fn=_step_batch_single,
                                 outputs_info=None,
                                 sequences=[embs_h, s_e_t_h]
                                 )  # n_batch, n_table - 1, n_table_h, dim
        enc_t_h = enc_t_h.reshape([n_batch * (t_len - 1), t_len_h, dim])
        tiled_global_ctx = tensor.tile(global_ctx[:, None, :], [1, (t_len - 1), 1]).reshape([n_batch * (t_len - 1),
                                                                                             dim])
        embs_t_hname = self.t_h_att_layer.get_output(tiled_global_ctx, enc_t_h, t_h_mask.reshape([n_batch * (t_len - 1),
                                                                                                  t_len_h]))
        embs_t_hname = embs_t_hname.reshape([n_batch, (t_len - 1), dim])

        emb_empty_c = embs_h[:, 0, :]
        emb_all_c = embs_h[:, 1, :]

        h2t_idx_flat = h2t_idx.flatten()  # n_batch * (n_column - 2)
        h2t_idx_flat_idx = tensor.tile(tensor.arange(n_batch)[:, None],
                                       [1, h_len - 2]).reshape(h2t_idx_flat.shape) * t_len + h2t_idx_flat
        embs_h_tname = embs_t.reshape([n_batch * t_len, dim])[h2t_idx_flat_idx]
        embs_h_tname = embs_h_tname.reshape([n_batch, h_len - 2, dim])

        embs_fusion_t = self.combine_t.get_output(embs_t[:, 1:, :], embs_t_hname)
        ctx_t = tensor.concatenate([emb_empty_t[:, None, :], embs_fusion_t], axis=1)  # n_batch, n_table, dim

        embs_fusion_c = self.combine_c.get_output(embs_h[:, 2:, :], embs_h_tname)
        ctx_h = tensor.concatenate([emb_empty_c[:, None, :], emb_all_c[:, None, :],
                                    embs_fusion_c], axis=1)  # n_batch, n_column, dim

        emb_query = self.combine_query.get_output(emb_q, emb_p)  # n_batch, dim

        emb_d = self.d_att_layer.get_output(emb_query, dropout(ctx_t, self.trng, dropout_rate, use_dropout), t_mask)

        return global_ctx, join_embs, emb_q, emb_p, emb_d, ctx_t, ctx_h, enc_q










