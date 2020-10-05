#!/usr/bin/env python

from layers_bert import MultiHeadAttentionLayer, FeedForwardLayer, PositionEmbeddingLayer, \
    PointerNetwork
from layers_bert import LayerNormLayer, dropout

from collections import OrderedDict

from theano import tensor


class TransformerDecoder(object):
    def __init__(self, agg_num=6, op_num=4, embed_dim=512,
                 decoder_num=6, head_num=6, hidden_num=512, dropout_rate=0.1, trng=None,
                 name="Transformer"):
        self.agg_num = agg_num
        self.op_num = op_num
        self.max_length = 512
        self.embed_dim = embed_dim
        self.decoder_num = decoder_num
        self.head_num = head_num
        self.hidden_num = hidden_num
        self.mid_num = self.hidden_num * 4
        self.dropout_rate = dropout_rate
        self.trng = trng
        self.tparams = OrderedDict()
        self.name = name
        self.INF = 1e+10
        self.decoder_layers = []
        self.pos_emb = PositionEmbeddingLayer(self.max_length, self.hidden_num, initializer='pos',
                                              name='embeddings_position_embeddings')
        self.emb_layer_norm = LayerNormLayer(self.hidden_num, name="embeddings_LayerNorm")
        self.pointer_network = PointerNetwork(self.hidden_num, self.hidden_num,
                                              scale="xavier", name="pointer_network")
        self.pointer_network_vs = PointerNetwork(self.hidden_num, self.hidden_num,
                                                 scale="xavier", name="pointer_network_vs")
        self.pointer_network_ve = PointerNetwork(self.hidden_num, self.hidden_num,
                                                 scale="xavier", name="pointer_network_ve")
        self.linear_agg = FeedForwardLayer(self.hidden_num, self.agg_num, name="linear_agg")
        self.linear_op = FeedForwardLayer(self.hidden_num, self.op_num, name="linear_op")
        self.final_layer_norm = LayerNormLayer(self.hidden_num, name="final_LayerNorm")
        for i in range(self.decoder_num):
            self.decoder_layers.append(self.init_decoder_block(i))

        for layer in [self.pos_emb, self.emb_layer_norm,
                      self.pointer_network, self.pointer_network_vs, self.pointer_network_ve,
                      self.linear_agg, self.linear_op, self.final_layer_norm]:
            for k, v in layer.tparams.iteritems():
                self.tparams[self.name + '_' + layer.name + '_' + k] = v

        for decoder_layer in self.decoder_layers:
            for layer_name, layer in decoder_layer.iteritems():
                for k, v in layer.tparams.iteritems():
                    self.tparams[self.name + '_' + layer.name + '_' + k] = v

    def init_decoder_block(self, layer_idx):
        block = OrderedDict()
        prefix = "decoder_layer_%d" % layer_idx

        self_att_layer_norm = LayerNormLayer(self.hidden_num, name="%s_self_attention_output_LayerNorm" % prefix)

        attention_self = MultiHeadAttentionLayer(self.hidden_num, self.hidden_num, self.hidden_num, self.hidden_num,
                                                 self.head_num, history_only=True, scale="xavier", ortho=True,
                                                 name="%s_self_attention" % prefix)

        query_att_layer_norm = LayerNormLayer(self.hidden_num, name="%s_query_attention_output_LayerNorm" % prefix)

        attention_query = MultiHeadAttentionLayer(self.hidden_num, self.embed_dim, self.embed_dim, self.hidden_num,
                                                  self.head_num, history_only=False, scale="xavier", ortho=True,
                                                  name="%s_query_attention" % prefix)

        out_layer_norm = LayerNormLayer(self.hidden_num, name="%s_feed_forward_LayerNorm" % prefix)

        inter_layer = FeedForwardLayer(self.hidden_num, self.mid_num, scale="xavier", ortho=True,
                                       name="%s_inter_feed_forward" % prefix)

        out_layer = FeedForwardLayer(self.mid_num, self.hidden_num, scale="xavier", ortho=True,
                                     name="%s_out_feed_forward" % prefix)

        block['self_attention'] = attention_self
        block['self_att_layer_norm'] = self_att_layer_norm
        block['query_attention'] = attention_query
        block['query_att_layer_norm'] = query_att_layer_norm
        block['intermediate'] = inter_layer
        block['out'] = out_layer
        block['out_layer_norm'] = out_layer_norm

        return block

    def build_decoder_block(self, block, input_ctx, input_y, ctx_mask=None, mask_y=None, in_train=True):

        input_norm = block['self_att_layer_norm'].get_output(input_y)
        query = block['self_attention'].get_output(q=input_norm, k=input_norm, v=input_norm, k_mask=mask_y,
                                                   activ='linear')
        query = dropout(query, self.trng, self.dropout_rate, in_train)
        query = input_y + query

        query_norm = block['query_att_layer_norm'].get_output(query)
        ctx = block['query_attention'].get_output(q=query_norm, k=input_ctx, v=input_ctx, k_mask=ctx_mask,
                                                  activ='linear')
        ctx = dropout(ctx, self.trng, self.dropout_rate, in_train)
        ctx = ctx + query

        ctx_norm = block['out_layer_norm'].get_output(ctx)
        inter = block['intermediate'].get_output(ctx_norm, activ='gelu')
        out = block['out'].get_output(inter, activ='linear')
        out = dropout(out, self.trng, self.dropout_rate, in_train)
        out = ctx + out

        return out

    def get_output(self, emb_y_shifted, y_mask, src_ctx, ctx_mask, enc_q, enc_h, mask_q, mask_h, in_train=True,
                   final=True):
        # input_x: n_samples, n_timestep
        # input_mask: n_samples, n_timestep
        # enc_q: n_samples, n_q, n_emb
        # enc_h: n_samples, n_h, n_emb
        # mask_q: n_samples, n_q
        # mask_h: n_samples, n_h
        n_timestep_y = emb_y_shifted.shape[1]

        y_pos = tensor.arange(0, n_timestep_y, dtype='int32')
        emb_pos_y = self.pos_emb.get_output(y_pos)  # n_timestep, n_emb
        emb_y = emb_y_shifted + emb_pos_y[None, :, :]
        ctx_y = self.emb_layer_norm.get_output(emb_y)

        for decoder_layer in self.decoder_layers:
            ctx_y = self.build_decoder_block(decoder_layer, src_ctx, ctx_y, ctx_mask, y_mask, in_train=in_train)

        ctx_y = self.final_layer_norm.get_output(ctx_y)
        # ctx_y: n_samples, n_timestep, n_dim
        # enc_h: n_samples, n_h, n_dim
        if final:
            logit = self.pointer_network.get_output(ctx_y, enc_h)

            logit = logit * mask_h[:, None, :] + (1. - mask_h)[:, None, :] * -self.INF
            # n_samples, n_timestep, n_h
            logit_agg = self.linear_agg.get_output(ctx_y, activ="linear")
            logit_op = self.linear_op.get_output(ctx_y, activ="linear")
            logit_vs = self.pointer_network_vs.get_output(ctx_y, enc_q)
            logit_vs = logit_vs * mask_q[:, None, :] + (1. - mask_q)[:, None, :] * -self.INF
            logit_ve = self.pointer_network_ve.get_output(ctx_y, enc_q)
            logit_ve = logit_ve * mask_q[:, None, :] + (1. - mask_q)[:, None, :] * -self.INF
            return ctx_y, logit, logit_agg, logit_op, logit_vs, logit_ve
        else:
            return ctx_y
