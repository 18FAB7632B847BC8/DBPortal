from SimpleNL2SQL import SimpleNL2SQL
from NL2SQL import NL2SQL
from data import DataHolder
from config_default import simple_val_loss_config
import numpy
import logging
import time
from TableParsing import parse_table
from evaluation import evaluate, build_foreign_key_map_from_json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )


class Trainer(object):
    def __init__(self, model_type, config, model_config, train_data_config, valid_data_config):
        if model_type == "simple":
            self.model = SimpleNL2SQL(model_config)
        else:
            self.model = NL2SQL(model_config)
        self.mode_type = model_type
        self.data_holder = DataHolder(**train_data_config)
        self.valid_holder = DataHolder(**valid_data_config)
        self.valid_loss_holder = DataHolder(**simple_val_loss_config)

        # Counters
        self.uctr = 0  # update ctr
        self.ectr = 0  # epoch ctr
        self.vctr = 0  # validation ctr
        self.early_bad = 0  # early-stop counter

        self.save_best_n = config.get('save_best_n', 10)  # Keep N best validation models on disk
        self.max_updates = config.get('max_updates', 0)  # Training stops if uctr hits 'max_updates'
        self.max_epochs = config.get('max_epochs', 500)  # Training stops if ectr hits 'max_epochs'

        # Validation related parameters
        self.patience = config.get('patience', 20)  # Stop training if no improvement after this validations
        self.valid_start = config.get('valid_start', 20)  # Start validation at epoch 'valid_start'
        self.dump_frequency = config.get('dump_frequency', 5000)
        self.log_path = '../logs/%s.log' % self.mode_type
        self.metrics = config.get('metrics', ['loss', 'accuracy'])
        self.early_metric = config.get('early_metric', 'accuracy')
        self.f_valid = config.get('f_valid', 1000)  # Validation frequency in terms of updates
        self.epoch_valid = (self.f_valid == 0)  # 0: end of epochs
        # self.valid_save_hyp = train_args.valid_save_hyp  # save validation hypotheses under 'valid_hyps' folder
        self.f_verbose = config.get('f_verbose', 1)  # Print frequency
        self.decay_c = config.get('decay_c', 0.)
        self.epoch_losses = []
        self.valid_mode = "simple"

        # Multiple comma separated metrics are supported
        # Each key is a metric name, values are metrics so far.
        self.valid_metrics = dict()
        if self.f_valid >= 0:
            # NOTE: This is relevant only for fusion models + WMTIterator

            # first one is for early-stopping

            for metric in self.metrics:
                self.valid_metrics[metric] = []

            # Ensure that loss exists
            self.valid_metrics['loss'] = []

            # Best N checkpoint saver
            self.best_models = []

            if self.early_metric in ['loss', 'px', 'ter']:
                self.early_metric_larger_better = False
            elif self.early_metric in ['bleu', 'meteor', 'cider', 'rouge', 'accuracy']:
                self.early_metric_larger_better = True
            else:
                raise KeyError('metric %s invalid' % self.early_metric)

    def prepare(self):

        self.model.init_model()
        # logging.info("Loading pretrained Model to %s" % self.mode_type)
        # self.model.load("../data/save/val091-loss_1.794_trans.npz")
        if self.model.bert.encoder_num >= 12:
            logging.info("Loading pretrained Model to %s" % self.model.bert.name)
            self.model.bert.from_pretrained_bert("../bert.base/new_bert_base.npz")
        else:
            logging.info("No pre-trained encoder")
        # logging.info("Loading Checkpoint Model to %s" % "../data/save/reason_bert/val001-loss_3.841.npz")
        # self.model.load("../data/save/reason_bert/val001-loss_3.841.npz")
        if self.model.dont_update is None:
            self.model.dont_update = set()
        #  for k in self.model.tparams:
        #     if 'bert' in k and 'type' not in k:
        #         self.model.dont_update.add(k)

        logging.info('loading traing data...')
        self.data_holder.read_data()

        logging.info('loading valid data...')
        self.valid_holder.read_data()
        self.valid_holder.reset()
        self.valid_loss_holder.read_data()
        self.valid_loss_holder.reset()
        logging.info('build model...')
        data_loss = self.model.build_model()
        logging.info('build sampler...')
        self.model.build_sampler()
        reg_loss = []
        if self.decay_c > 0:
            reg_loss.append(self.model.get_l2_weight_decay(self.decay_c))

        reg_loss = sum(reg_loss) if len(reg_loss) > 0 else None
        # logging.info('build optimizer...')
        # self.model.build_optimizer(data_loss, reg_loss, debug=False)
        if self.model.bert.encoder_num >= 12:
            logging.info('build optimizer bert')
            self.model.build_optimizer_bert(data_loss, reg_loss)
        else:
            logging.info('build optimizer...')
            self.model.build_optimizer(data_loss, reg_loss, debug=False)

    def __is_last_best(self):
        value_list = self.valid_metrics[self.early_metric]
        if len(value_list) <= 1:
            return True
        if not self.early_metric_larger_better:
            return self.valid_metrics[self.early_metric][-1] < min(self.valid_metrics[self.early_metric][:-1])
        else:
            return self.valid_metrics[self.early_metric][-1] > max(self.valid_metrics[self.early_metric][:-1])

    def __save_best_model(self):
        if self.save_best_n > 0:
            # Get the score of the system that will be saved
            cur_score = self.valid_metrics[self.early_metric][-1]

            # Custom filename with metric score
            cur_fname = "val%3.3d-%s_%.3f.npz" % (self.vctr, self.early_metric, cur_score)

            # Stack is empty, save the model whatsoever
            if len(self.best_models) < self.save_best_n:
                self.best_models.append((cur_score, cur_fname))

            # Stack is full, replace the worst model
            else:
                worst_index = -1
                if self.early_metric_larger_better:
                    worst_value = min(self.best_models, key=lambda m: m[0])
                else:
                    worst_value = max(self.best_models, key=lambda m: m[0])

                for i, value in enumerate(self.best_models):
                    if worst_value == value[0]:
                        worst_index = i
                        break

                self.best_models[worst_index] = (cur_score, cur_fname)

            logging.info('Saving model with best validation %s' % self.early_metric)
            self.model.save(cur_fname)

    def __train_epoch(self):

        self.ectr += 1

        start = time.time()
        start_uctr = self.uctr
        title_str = '*****   Starting Epoch %d   *****' % self.ectr
        logging.info(title_str)
        logging.info('*' * len(title_str))

        batch_losses = []
        self.data_holder.reset()
        for data in self.data_holder.get_batch_data():
            self.uctr += 1

            loss = self.model.train_batch(*data)

            batch_losses.append(loss)
            # Verbose
            if self.uctr % self.f_verbose == 0:
                logging.info("Epoch: %6d, update: %7d, cost: %10.6f" % (self.ectr, self.uctr, loss))

            # Should we stop
            if self.uctr == self.max_updates:
                logging.info("Max iteration %d reached." % self.uctr)
                return False

            # Do validation
            if not self.epoch_valid and self.f_valid > 0 and self.uctr % self.f_valid == 0:
                self.__do_validation()

            # Check stopping conditions
            if self.early_bad == self.patience:
                logging.info("Early stopped.")
                return False

                # An epoch is finished
        epoch_time = time.time() - start

        # Print epoch summary
        up_ctr = self.uctr - start_uctr
        self.__dump_epoch_summary(batch_losses, epoch_time, up_ctr)

        logging.info('save epoch model...')
        self.model.save('epoch_%d_model.npz' % self.ectr)

        # Do validation
        if self.epoch_valid:
            self.__do_validation()
            if self.early_bad == self.patience:
                logging.info("Early stopped.")
                return False

        # Check whether maximum epoch is reached
        if self.ectr == self.max_epochs:
            logging.info("Max epochs %d reached." % self.max_epochs)
            return False

        return True

    def __do_validation(self):
        if self.ectr >= self.valid_start:
            logging.info('Do validation')
            self.model.save('checkpoint_model.npz')
            self.vctr += 1

            # Compute validation loss
            # self.model.bert.load_partial("../bert.base/new_bert_base.npz", "bert_embeddings_word_embeddings_Wemb")
            self.model.set_dropout(False)
            self.model.set_dropout_bert(False)
            valid_results = self.model.val_loss(self.valid_loss_holder, True)

            for k, v in valid_results.iteritems():
                logging.info("Validation %3d - %s = %.3f" % (self.vctr, k.upper(), v))

            post_fix = "simple"
            if self.model.bert.encoder_num >= 12:
                post_fix = "bert"
            pred = "../data/test_comp_dev_%s.sql" % post_fix
            logging.info("Reread validation data...")
            self.valid_holder.re_read()
            fw = open(pred, "w")
            database = parse_table()
            logging.info("Making Beam Search...")
            for data in self.valid_holder.get_batch_data():
                comp_sql = self.model.construct_comp_sql(data, database)
                fw.write("%s\n" % comp_sql)
            logging.info("Beam Search Finished...")

            gold = "../data/gold_dev.sql"

            db_dir = "../data/spider/database"
            etype = "match"
            table = "../data/spider/tables.json"

            kmaps = build_foreign_key_map_from_json(table)

            f_int_w = open("../data/intermediate_val_%s_%d" % (post_fix, self.vctr), 'w')
            score = evaluate(gold, pred, db_dir, etype, kmaps, f_int_w)

            logging.info("Validation %3d - Accuracy = %.3f" % (self.vctr, score))

            # self.model.load_partial("%s/checkpoint_model.npz" % self.model.save_path,
            #                         "bert_embeddings_word_embeddings_Wemb")
            self.model.set_dropout(True)
            self.model.set_dropout_bert(True)

            # Add val_loss
            self.valid_metrics['loss'].append(valid_results['loss'])
            self.valid_metrics['accuracy'].append(score)

            # Print validation loss
            # logging.info("Validation %3d - Match Accuracy = %.3f" % (self.vctr, valid_results))

            # Is this the best evaluation based on early-stop metric?
            if self.__is_last_best():
                self.__save_best_model()
                self.early_bad = 0
            else:
                self.early_bad += 1
                logging.info("Early stopping patience: %d validation left" % (self.patience - self.early_bad))

            self.__dump_val_summary()

    def __dump_val_summary(self):
        for k, v_list in self.valid_metrics.iteritems():
            last_v = self.valid_metrics[k][-1]
            if k in ['loss', 'px', 'ter']:
                best_v = min(v_list)
            elif k in ['bleu', 'meteor', 'cider', 'rouge', 'accuracy']:
                best_v = max(v_list)
            else:
                best_v = last_v
            logging.info('Current BEST %s = %.3f, Last %s = %.3f' % (k, best_v, k, last_v))

        logging.info('Model Saved Path: %s' % self.model.save_path)

    def __dump_epoch_summary(self, losses, epoch_time, up_ctr):
        """Print epoch summary."""
        update_time = epoch_time / float(up_ctr)
        mean_loss = numpy.array(losses).mean()
        self.epoch_losses.append(mean_loss)

        logging.info("--> Epoch %d finished with mean loss %.5f" % (self.ectr, mean_loss))
        logging.info("--> Epoch took %.3f minutes, %.3f sec/update" % ((epoch_time / 60.0), update_time))

    def run(self):
        while self.__train_epoch():
            pass

        self.__dump_val_summary()
        logging.info('Saving final model')
        self.model.save('final_model.npz')
