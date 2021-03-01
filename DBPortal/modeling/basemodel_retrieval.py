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
        self.train_mode = config.get('train_mode', "inner")
        self.margin = config.get('margin', 0.3)
        self.f_log_probs = None
        self.train_batch = None
        self.forward = None
        self.backword = None
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
                        logging.info("Update BERT parameter: %s, %s" % (key, value.get_value().shape))
                        tparams_bert[key] = value
                    else:
                        logging.info("Update Other parameter: %s, %s" % (key, value.get_value().shape))
                        tparams[key] = value
                else:
                    logging.info("Don't update key: %s" % key)
        else:
            for key, value in self.tparams.iteritems():
                if 'bert' in key.lower():
                    logging.info("Update BERT parameter: %s, %s" % (key, value.get_value().shape))
                    tparams_bert[key] = value
                else:
                    logging.info("Update Other parameter: %s, %s" % (key, value.get_value().shape))
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

    def build_optimizer_asc(self, cost, regcost):
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

        # Get gradients of cost with respect to variables
        # This uses final_cost which is not normalized w.r.t sentence lengths
        for k in tparams:
            logging.info("Update Other parameter: %s, %s" % (k, tparams[k].get_value().shape))
        grads = tensor.grad(final_cost, wrt=list(tparams.values()))

        # Clip gradients if requested
        if self.clip_c > 0:
            grads = self.get_clipped_grads(grads=grads, clip_c=self.clip_c)

        # Create theano shared variable for learning rate
        # self.lrate comes from **kwargs / nmt-train params
        self.learning_rate = theano.shared(numpy.float32(self.lrate), name='lrate')

        # Get updates
        updates = adam_warm(tparams, grads, self.warmup_steps, self.decay_steps, lr0=self.learning_rate)

        self.forward = theano.function(list(self.inputs.values()), [final_cost] + grads)
        self.backword = theano.function(grads, None, updates=updates)

    def build_optimizer_bert_asc(self, cost, regcost):
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

        self.forward = theano.function(list(self.inputs.values()), [final_cost] + grads + grads_bert)
        self.backword = theano.function(grads + grads_bert, None, updates=updates + updates_bert)

    def val_loss(self, valid_holder):
        total_count = 0
        if self.train_mode == "inner":
            logging.info("Training mode: Inner")
            right_count = 0
            for data in valid_holder.get_batch_data():
                x_q, x_q_mask, x_c, x_c_mask, t_mask, join_paths = data

                t_mask[:, 0] = 0.
                cosines = self.f_log_probs(x_q, x_q_mask, x_c, x_c_mask)  # n_sample, n_table
                cosines *= t_mask
                for cosine, join_path in zip(cosines, join_paths):
                    total_count += 1
                    right_count += int(all(cosine[join_path] >= self.margin))
        elif self.train_mode == "all":
            logging.info("Training mode: All")
            right_count = 0
            embs_q = []
            embs_c = []

            for data in valid_holder.get_batch_data():
                x_q, x_q_mask, x_c, x_c_mask, t_mask, _ = data
                f_output = self.f_log_probs(x_q, x_q_mask, x_c, x_c_mask)
                emb_q = f_output['emb_q'].tolist()  # n_batch, dim
                emb_c = f_output['emb_c'].tolist()  # n_batch, n_table, dim
                embs_q.extend([numpy.array(emb) for emb in emb_q])
                for emb, mask in zip(emb_c, t_mask):
                    # n_table, dim
                    embs_c.append(numpy.array(emb)[1:int(sum(mask))])

            sim = numpy.zeros((len(embs_q), len(embs_c)), dtype="float32")
            for i, emb_q in enumerate(embs_q):
                emb_q_norm = emb_q / numpy.sqrt((emb_q ** 2).sum(-1, keepdims=True))  # 1024
                for j, emb_c in enumerate(embs_c):
                    emb_c_norm = emb_c / numpy.sqrt((emb_c ** 2).sum(-1, keepdims=True))  # n_table, 1024
                    sim[i, j] = ((emb_q_norm[None, :] * emb_c_norm).sum(-1)).max()
            eps = numpy.diag(1e-6 * numpy.ones((len(embs_q),)))
            sim += eps
            ranks = sim.argsort(axis=1)  # 1000 * 1000

            for j, rank in enumerate(ranks):
                total_count += 1
                r = len(rank) - int(numpy.argwhere(rank == j)) - 1
                if r == 0:
                    right_count += 1
        else:
            raise
        return float(right_count) / float(total_count)

    def val_loss_simple(self, valid_holder):
        if self.train_mode == "all":
            logging.info("Training mode: All")
            right_count = 0
            embs_q = []
            embs_c = []

            for data in valid_holder.get_batch_data():
                x_q, x_q_mask, x_c, x_c_mask, t_mask, _ = data
                f_output = self.f_log_probs(x_q, x_q_mask, x_c, x_c_mask)
                emb_q = f_output['emb_q'].tolist()  # n_batch, dim
                emb_c = f_output['emb_c'].tolist()  # n_batch, n_table, dim
                embs_q.extend([numpy.array(emb) / numpy.sqrt((numpy.array(emb) ** 2).sum(-1, keepdims=True)) for emb in emb_q])
                for emb, mask in zip(emb_c, t_mask):
                    # n_table, dim
                    emb = emb[0]
                    embs_c.append(numpy.array(emb) / numpy.sqrt((numpy.array(emb) ** 2).sum(-1, keepdims=True)))
            embs_q = numpy.array(embs_q)
            embs_c = numpy.array(embs_c)
            eps = numpy.diag(1e-6 * numpy.ones(1000,))
            for i in range(5):
                emb_q = embs_q[i * 1000:i * 1000 + 1000]  # 1000, 768
                emb_c = embs_c[i * 1000:i * 1000 + 1000]  # 1000, 768
                sim = numpy.dot(emb_q, emb_c.T)  # 1000, 1000
                sim += eps
                ranks = sim.argsort(axis=1)  # 1000 * 1000
                for j, rank in enumerate(ranks):
                    r = len(rank) - int(numpy.argwhere(rank == j)) - 1
                    if r == 0:
                        right_count += 1

            return float(right_count) / 5000.
        else:
            raise









