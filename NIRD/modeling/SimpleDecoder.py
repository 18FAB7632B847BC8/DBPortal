#!/usr/bin/env python

from collections import OrderedDict
from TransformerDecoder import TransformerDecoder
from layers_bert import FeedForwardLayer, LayerNormLayer


class SimpleDecoder(object):
    def __init__(self, hid_dim, ctx_dim, decoder_num, head_num, foreign_dim, scale, dropout_rate, name, trng=None):
        self.hid_dim = hid_dim
        self.decoder_num = decoder_num
        self.ctx_dim = ctx_dim
        self.head_num = head_num
        self.foreign_dim = foreign_dim
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.trng = trng
        self.name = name
        self.max_length = 20

        self.tparams = OrderedDict()

        self.reason_decoder_layer = TransformerDecoder(self.ctx_dim, self.decoder_num, self.head_num, self.hid_dim,
                                                       self.dropout_rate, self.trng,
                                                       name=self.name + "_simple_decoder")

        self.from_decoder_layer = TransformerDecoder(self.ctx_dim,
                                                     self.decoder_num, self.head_num, self.hid_dim,
                                                     self.dropout_rate, self.trng,
                                                     name=self.name + "_from_decoder")

        self.linear_table = FeedForwardLayer(hid_dim * 3 + self.foreign_dim, hid_dim, scale,
                                             name=self.name + "_linear_table_layer")

        for layer in [self.reason_decoder_layer, self.from_decoder_layer,
                      ]:
            if layer is None:
                raise KeyError("layer is None")
            for k, v in layer.tparams.iteritems():
                p_name = layer.name + '_' + k
                assert p_name not in self.tparams, "Duplicated parameter: %s" % p_name
                self.tparams[p_name] = v

    def get_output_col(self, y_embs, y_mask, y_embs_app, ctx_all, ctx_mask, use_dropout):

        co_embs = y_embs + y_embs_app

        states_col = self.reason_decoder_layer.get_output(co_embs, y_mask, ctx_all, ctx_mask, use_dropout)

        return states_col

    def get_output_tab(self, y_embs_t, y_t_mask, ctx_all, ctx_mask, use_dropout):

        states_tab = self.from_decoder_layer.get_output(y_embs_t, y_t_mask, ctx_all, ctx_mask, use_dropout)

        return states_tab
