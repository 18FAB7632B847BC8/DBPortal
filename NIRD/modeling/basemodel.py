#!/usr/bin/env python

import theano.tensor as tensor
from optimizer import adam_warm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.debugmode import DebugMode
from collections import OrderedDict
from utils import unzip

import theano
import numpy
import logging


class BaseModel(object):
    def __init__(self, config):
        self.options = config.copy()
        self.config = config
        self.trng = None
        self.set_trng(config.get('seed', 1235))
        self.inputs = OrderedDict()
        self.lrate = config.get('lrate', 1e-7)
        self.lrate_bert = config.get('lrate_bert', 1e-7)
        self.learning_rate = None
        self.learning_rate_bert = None
        self.save_path = config['save_path']
        self.clip_c = config.get('clip_c', 0.)
        self.use_dropout = theano.shared(numpy.float32(config.get('use_dropout', True)))
        self.use_bert_dropout = theano.shared(numpy.float32(config.get('use_bert_dropout', True)))
        self.warmup_steps = config.get('warmup_steps', 10000)
        self.decay_steps = config.get('decay_steps', 1e+7)
        self.layers = []
        self.f_log_probs = None
        self.train_batch = None
        self.tparams = OrderedDict()
        self.dont_update = None
        self.dont_decay = None

    def update_shared_variables(self, _from):
        """Reset some variables from _from dict."""
        for k, v in _from.keys():
            self.tparams[k].set_value(v)

    def save(self, file_name):
        numpy.savez('%s/%s' % (self.save_path, file_name), tparams=unzip(self.tparams), options=self.options)

    def load(self, path, ingnore_prefix=None, key_prefix=None):
        params = numpy.load(path, allow_pickle=True)['tparams'].tolist()
        for k, v in params.iteritems():
            if ingnore_prefix and ingnore_prefix in k:
                logging.info("ignore parameter: %s" % k)
                continue
            if key_prefix and key_prefix not in k:
                logging.info("ignore parameter: %s" % k)
                continue
            if k in self.tparams:
                self.tparams[k].set_value(v)
            else:
                logging.warning('unknown key: %s in model' % k)
        for k, v in self.tparams.iteritems():
            if k not in params:
                logging.info('added key: %s in model, shape : %s' % (k, str(v.get_value().shape)))

    def load_partial(self, path, key):
        params = numpy.load(path, allow_pickle=True)['tparams'].tolist()
        assert key in params and key in self.tparams
        value = params[key]
        self.tparams[key].set_value(value)

    def set_trng(self, seed):
        """Set the seed for Theano RNG."""
        self.trng = RandomStreams(seed)

    def set_dropout(self, val):
        self.use_dropout.set_value(numpy.float32(val))

    def set_dropout_bert(self, val):
        self.use_bert_dropout.set_value(numpy.float32(val))

    def get_l2_weight_decay(self, decay_c, skip_bias=True):
        """Return l2 weight decay regularization term."""
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        dont_decay = self.dont_decay
        if dont_decay is None:
            dont_decay = set()
        dont_update = self.dont_update
        if dont_update is None:
            dont_update = set()
        for kk, vv in self.tparams.items():
            if kk in dont_decay or kk in dont_update:
                logging.info("Don't use decay: %s" % kk)
                continue
            # Skip biases for L2 regularization
            if not skip_bias or (skip_bias and vv.get_value().ndim > 1):
                weight_decay += (vv ** 2).sum()
            elif skip_bias and vv.get_value().ndim <= 1:
                logging.info("Don't use decay: %s" % kk)
        weight_decay *= decay_c
        return weight_decay

    @staticmethod
    def get_clipped_grads(grads, clip_c):
        """Clip gradients a la Pascanu et al."""
        g2 = 0.
        new_grads = []
        for g in grads:
            g2 += (g**2).sum()
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        return new_grads

    def build_optimizer(self, cost, regcost, debug=False):
        """Build optimizer by optionally disabling learning for some weights."""
        tparams = OrderedDict()

        if self.dont_update:
            for key, value in self.tparams.iteritems():
                if key not in self.dont_update:
                    tparams[key] = value
                else:
                    logging.info("Don't update key: %s" % key)
        else:
            tparams.update(self.tparams)

        # Our final cost
        final_cost = cost.mean()

        # If we have a regularization cost, add it
        if regcost is not None:
            final_cost += regcost

        # Normalize cost w.r.t sentence lengths to correctly compute perplexity
        # Only active when y_mask is available

        norm_cost = final_cost

        # Get gradients of cost with respect to variables
        # This uses final_cost which is not normalized w.r.t sentence lengths
        for k in tparams:
            logging.info("Update parameter: %s" % k)
        grads = tensor.grad(final_cost, wrt=list(tparams.values()))

        # Clip gradients if requested
        if self.clip_c > 0:
            grads = self.get_clipped_grads(grads=grads, clip_c=self.clip_c)

        # Create theano shared variable for learning rate
        # self.lrate comes from **kwargs / nmt-train params
        self.learning_rate = theano.shared(numpy.float32(self.lrate), name='lrate')

        # Get updates
        updates = adam_warm(tparams, grads, self.warmup_steps, self.decay_steps, lr0=self.learning_rate)

        # Compile forward/backward function
        if debug:
            self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates,
                                               mode=DebugMode(optimizer='fast_compile', check_c_code=True,
                                                              check_py_code=False, check_isfinite=True,))
        else:
            self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates)

    def build_optimizer_bert(self, cost, regcost):
        """Build optimizer by optionally disabling learning for some weights."""
        tparams = OrderedDict()
        tparams_bert = OrderedDict()

        if self.dont_update:
            for key, value in self.tparams.iteritems():
                if key not in self.dont_update:
                    if 'bert' in key.lower():
                        logging.info("Update BERT parameter: %s" % key)
                        tparams_bert[key] = value
                    else:
                        logging.info("Update Other parameter: %s" % key)
                        tparams[key] = value
                else:
                    logging.info("Don't update key: %s" % key)
        else:
            for key, value in self.tparams.iteritems():
                if 'bert' in key.lower():
                    logging.info("Update BERT parameter: %s" % key)
                    tparams_bert[key] = value
                else:
                    logging.info("Update Other parameter: %s" % key)
                    tparams[key] = value

        # Our final cost
        final_cost = cost.mean()

        # If we have a regularization cost, add it
        if regcost is not None:
            final_cost += regcost

        # Normalize cost w.r.t sentence lengths to correctly compute perplexity
        # Only active when y_mask is available

        norm_cost = final_cost

        # Get gradients of cost with respect to variables
        # This uses final_cost which is not normalized w.r.t sentence lengths
        grads = tensor.grad(final_cost, wrt=list(tparams.values()))
        grads_bert = tensor.grad(final_cost, wrt=list(tparams_bert.values()))

        # Clip gradients if requested
        if self.clip_c > 0:
            grads = self.get_clipped_grads(grads=grads, clip_c=self.clip_c)
            grads_bert = self.get_clipped_grads(grads=grads_bert, clip_c=self.clip_c)

        # Create theano shared variable for learning rate
        # self.lrate comes from **kwargs / nmt-train params
        self.learning_rate = theano.shared(numpy.float32(self.lrate), name='lrate')
        self.learning_rate_bert = theano.shared(numpy.float32(self.lrate_bert), name='lrate_bert')

        # Get updates
        updates = adam_warm(tparams, grads, self.warmup_steps, self.decay_steps,
                            lr0=self.learning_rate)
        updates_bert = adam_warm(tparams_bert, grads_bert, self.warmup_steps, self.decay_steps,
                                 lr0=self.learning_rate_bert)

        self.train_batch = theano.function(list(self.inputs.values()), norm_cost, updates=updates + updates_bert)

    def val_loss(self, valid_holder, simple=True):
        if simple:
            loss = []
            loss_column = []
            loss_op = []
            loss_agg = []
            loss_vls = []
            loss_vle = []
            loss_vrs = []
            loss_vre = []
            loss_cd = []
            loss_vn = []
            loss_t = []
            loss_d = []
            loss_co = []
            loss_o = []
            loss_c = []
            loss_l = []
            loss_nf = []
            for data in valid_holder.get_batch_data():
                input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
                join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
                q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, \
                y_column, y_type2, y_agg, y_op, y_vls, y_vle, y_vrs, y_vre, y_cd, y_vn, \
                y_c_mask, y_cw_mask, y_w_mask, y_cwo_mask, \
                y_t, y_t_mask, \
                y_co, y_d, y_type1, y_o, y_c, y_l, y_nf = data

                norm_c = y_c_mask.sum(1)
                norm_cw = y_cw_mask.sum(1)
                norm_w = y_w_mask.sum(1)
                norm_cwo = y_cwo_mask.sum(1)
                norm_t = y_t_mask.sum(1)

                valid_dict = self.f_log_probs(*data)
                cost_column = valid_dict["cost_column"] / norm_c
                cost_op = valid_dict["cost_op"] / norm_w
                cost_agg = valid_dict["cost_agg"] / norm_cwo
                cost_vls = valid_dict["cost_vls"] / norm_w
                cost_vle = valid_dict["cost_vle"] / norm_w
                cost_vrs = valid_dict["cost_vrs"] / norm_w
                cost_vre = valid_dict["cost_vre"] / norm_w
                cost_cd = valid_dict["cost_cd"] / norm_cw
                cost_vn = valid_dict["cost_vn"] / norm_w
                cost_t = valid_dict["cost_t"] / norm_t
                cost_d = valid_dict["cost_d"]
                cost_co = valid_dict["cost_co"]
                cost_o = valid_dict["cost_o"]
                cost_c = valid_dict["cost_c"]
                cost_l = valid_dict["cost_l"]
                cost_nf = valid_dict["cost_nf"]
                cost = cost_column + cost_op + cost_agg + cost_vls + cost_vle + cost_vrs + cost_vre + \
                    cost_cd + cost_vn + cost_t + cost_d + cost_co + cost_o + cost_c + cost_l + cost_nf
                loss.extend(cost)
                loss_column.extend(cost_column)
                loss_op.extend(cost_op)
                loss_agg.extend(cost_agg)
                loss_vls.extend(cost_vls)
                loss_vle.extend(cost_vle)
                loss_vrs.extend(cost_vrs)
                loss_vre.extend(cost_vre)
                loss_cd.extend(cost_cd)
                loss_vn.extend(cost_vn)
                loss_t.extend(cost_t)
                loss_d.extend(cost_d)
                loss_co.extend(cost_co)
                loss_o.extend(cost_o)
                loss_c.extend(cost_c)
                loss_l.extend(cost_l)
                loss_nf.extend(cost_nf)
            return {
                "loss": numpy.array(loss).mean(),
                "loss_column": numpy.array(loss_column).mean(),
                "loss_op": numpy.array(loss_op).mean(),
                "loss_agg": numpy.array(loss_agg).mean(),
                "loss_vls": numpy.array(loss_vls).mean(),
                "loss_vle": numpy.array(loss_vle).mean(),
                "loss_vrs": numpy.array(loss_vrs).mean(),
                "loss_vre": numpy.array(loss_vre).mean(),
                "loss_cd": numpy.array(loss_cd).mean(),
                "loss_vn": numpy.array(loss_vn).mean(),
                "loss_t": numpy.array(loss_t).mean(),
                "loss_d": numpy.array(loss_d).mean(),
                "loss_co": numpy.array(loss_co).mean(),
                "loss_o": numpy.array(loss_o).mean(),
                "loss_c": numpy.array(loss_c).mean(),
                "loss_l": numpy.array(loss_l).mean(),
                "loss_nf": numpy.array(loss_nf).mean()
            }



