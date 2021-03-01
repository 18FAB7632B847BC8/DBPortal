#!/usr/bin/env python

from layers_bert import MultiHeadAttentionLayer, FeedForwardLayer, EmbeddingLayer, PositionEmbeddingLayer
from layers_bert import LayerNormLayer, dropout
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from collections import OrderedDict

from theano import tensor
import numpy
import logging


class Bert(object):
    def __init__(self, token_num, embed_dim=768, bert_last_num=1,
                 encoder_num=12, head_num=12, hidden_num=768, dropout_rate=0.,
                 name="bert"):
        self.layers = []
        self.token_num = token_num
        self.bert_last_num = bert_last_num
        self.max_length = 514
        self.embed_dim = embed_dim
        self.encoder_num = encoder_num
        self.head_num = head_num
        self.hidden_num = hidden_num
        self.mid_num = self.hidden_num * 4
        self.type_num = 1
        self.dropout_rate = dropout_rate
        self.trng = RandomStreams(1235)
        self.tparams = OrderedDict()
        self.name = name

        self.encoder_layers = []
        self.word_emb = EmbeddingLayer(self.token_num, self.embed_dim, scale=0.02,
                                       name="embeddings_word_embeddings")
        self.pos_emb = PositionEmbeddingLayer(self.max_length, self.embed_dim, trainable=True,
                                              name='embeddings_position_embeddings')
        self.type_emb = EmbeddingLayer(self.type_num, self.embed_dim, scale=0.02,
                                       name="embeddings_token_type_embeddings")
        self.outer_type_emb = EmbeddingLayer(2, self.embed_dim, scale=0.02,
                                             name="embeddings_outer_token_type_embeddings")

        self.emb_layer_norm = LayerNormLayer(self.embed_dim, name="embeddings_LayerNorm")

        for i in range(self.encoder_num):
            encoder_layer = self.init_block(i)
            self.encoder_layers.append(encoder_layer)

        for layer in [self.word_emb, self.type_emb, self.pos_emb, self.outer_type_emb, self.emb_layer_norm]:
            for k, v in layer.tparams.iteritems():
                self.tparams[self.name + '_' + layer.name + '_' + k] = v

        for encoder_layer in self.encoder_layers:
            for layer_name, layer in encoder_layer.iteritems():
                for k, v in layer.tparams.iteritems():
                    self.tparams[self.name + '_' + layer.name + '_' + k] = v

    def init_block(self, layer_idx):
        block = OrderedDict()
        prefix = "encoder_layer_%d" % layer_idx
        attention = MultiHeadAttentionLayer(self.hidden_num, self.hidden_num, self.hidden_num, self.hidden_num,
                                            self.head_num, history_only=False, scale=0.02, ortho=True,
                                            name="%s_attention" % prefix)
        att_layer_norm = LayerNormLayer(self.hidden_num, name="%s_attention_output_LayerNorm" % prefix)

        intermediate = FeedForwardLayer(self.hidden_num, self.mid_num, scale=0.02, ortho=True,
                                        name="%s_intermediate" % prefix)

        out = FeedForwardLayer(self.mid_num, self.hidden_num, scale=0.02, ortho=True,
                               name="%s_output" % prefix)
        out_layer_norm = LayerNormLayer(self.hidden_num, name="%s_output_LayerNorm" % prefix)

        block['attention'] = attention
        block['att_layer_norm'] = att_layer_norm
        block['intermediate'] = intermediate
        block['out'] = out
        block['out_layer_norm'] = out_layer_norm

        return block

    def from_pretrained_bert(self, path, ignore=None):
        params = numpy.load(path, allow_pickle=True)['tparams'].tolist()
        for k, v in params.iteritems():
            if ignore and ignore in k:
                logging.info('ignore key from pretrained model: %s' % k)
                continue
            k = k.replace('bert', self.name)
            if k in self.tparams:
                if self.tparams[k].get_value().shape != v.shape:
                    logging.warn("dismatch shape in BERT: %s %s %s" % (k, str(self.tparams[k].get_value().shape), str(v.shape)))
                    """
                    v_ori = self.tparams[k].get_value()
                    assert v.ndim == 2 and v_ori.ndim == 2
                    assert v.shape[0] >= v_ori.shape[0] and v.shape[1] >= v_ori.shape[1]
                    self.tparams[k].set_value(v[:v_ori.shape[0], :v_ori.shape[1]])
                    """
                    raise
                else:
                    self.tparams[k].set_value(v)
            else:
                logging.warning('unknown key: %s in model' % k)

        for k, v in self.tparams.iteritems():
            k = k.replace(self.name, 'bert')
            if k not in params:
                logging.info('added key: %s in model, shape : %s' % (k, str(v.get_value().shape)))

    def load_partial(self, path, key):
        params = numpy.load(path, allow_pickle=True)['tparams'].tolist()
        assert key in params and key in self.tparams, "Key Error: %s" % key
        new_v = params[key]
        ori_v = self.tparams[key].get_value()

        assert new_v.ndim == 2 and ori_v.ndim == 2
        assert new_v.shape[1] == ori_v.shape[1]
        assert new_v.shape[0] > ori_v.shape[0]

        v = new_v.copy()
        v[:ori_v.shape[0], :] = ori_v

        self.tparams[key].set_value(v)

    def build_block(self, block, input_x, encoder_mask=None, in_train=True):
        ctx = block['attention'].get_output(q=input_x, k=input_x, v=input_x, k_mask=encoder_mask,
                                            activ='linear')
        ctx = dropout(ctx, self.trng, self.dropout_rate, in_train)
        ctx = input_x + ctx
        ctx = block['att_layer_norm'].get_output(ctx)

        inter = block['intermediate'].get_output(ctx, activ='gelu')
        out = block['out'].get_output(inter, activ='linear')
        out = dropout(out, self.trng, self.dropout_rate, in_train)
        out = ctx + out
        out = block['out_layer_norm'].get_output(out)

        return out

    def get_output(self, x, x_type_outer, mask=None, in_train=True):
        # x: n_batch, n_timestep
        # x_type: n_batch, n_timestep
        n_samples = x.shape[0]
        n_timestep = x.shape[1]
        emb_x = self.word_emb.get_output(x)  # n_batch, n_timestep, n_emb
        x_type = tensor.zeros_like(x, dtype="int32")
        emb_x = emb_x.reshape([n_samples, n_timestep, self.embed_dim])
        emb_type = self.type_emb.get_output(x_type)
        emb_outer_type = self.outer_type_emb.get_output(x_type_outer)
        emb_type = emb_type.reshape([n_samples, n_timestep, self.embed_dim])
        emb_outer_type = emb_outer_type.reshape([n_samples, n_timestep, self.embed_dim])

        x_pos = tensor.arange(2, n_timestep + 2, dtype='int32')
        emb_pos = self.pos_emb.get_output(x_pos)  # n_timestep, n_emb

        emb = emb_x + emb_pos[None, :, :] + emb_type + emb_outer_type

        emb = self.emb_layer_norm.get_output(emb)

        last_layer_output = emb

        encoded_layers = []
        for encoder_layer in self.encoder_layers:
            last_layer_output = self.build_block(encoder_layer, last_layer_output,
                                                 encoder_mask=mask, in_train=in_train)
            encoded_layers.append(last_layer_output)

        output = tensor.concatenate(encoded_layers[-self.bert_last_num:], axis=2)

        return output

