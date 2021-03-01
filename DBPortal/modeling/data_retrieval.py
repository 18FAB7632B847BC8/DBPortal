#!/usr/bin/env python

from multiprocessing.pool import ThreadPool as Pool
import numpy
import logging
from json import dump


class RetrievalClassifiedHolder(object):
    def __init__(self, question_path, header_path, cluster_path,
                 batch_size=128, shuffle_mode='simple', cluster_num=15,
                 num_workers=5):
        self.question_path = question_path
        self.column_path = header_path
        self.cluster_path = cluster_path
        self.cluster_num = cluster_num
        self.shuffle_mode = shuffle_mode
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers < self.batch_size else self.batch_size
        self.questions = [list() for i in xrange(cluster_num)]
        self.columns = [list() for i in xrange(cluster_num)]
        self.header_mask = [list() for i in xrange(cluster_num)]
        self.pool = None
        self._idx = None
        self._idx_idx = None

    def read_data(self):
        logging.info('Data Holder Read Data')
        f1 = open(self.question_path)
        f2 = open(self.column_path)
        f3 = open(self.cluster_path)

        while True:
            line1 = f1.readline()
            line2 = f2.readline()
            line3 = f3.readline()

            if not line1:
                break

            tokens_q = map(int, line1.strip().split())
            tokens_c = map(int, line2.strip().split())
            cluster_id = int(line3.strip())
            self.questions[cluster_id].append(tokens_q)
            self.columns[cluster_id].append(tokens_c)
            start = 0
            header_mask = []
            while start < len(tokens_c) - 1:
                end = tokens_c[1:].index(102, start)
                header_mask.append([start, end])
                start = end + 1
            self.header_mask[cluster_id].append(header_mask)

    def __len__(self):
        if not self.questions:
            self.read_data()
        length = 0
        for cluster in self.questions:
            length += len(cluster)

        return length

    def reset(self):
        self._idx = [None for _ in xrange(self.cluster_num)]
        self._idx_idx = []
        if self.shuffle_mode is None:
            for index, cluster in enumerate(self.questions):
                length = len(cluster)
                idx = numpy.arange(0, length)
                self._idx[index] = [idx[i:i + self.batch_size] for i in xrange(0, length, self.batch_size)]
                self._idx_idx.extend([[index, i] for i in xrange(len(self._idx[index]))])
        elif self.shuffle_mode == 'simple':
            for index, cluster in enumerate(self.questions):
                length = len(cluster)
                idx = numpy.random.permutation(length)
                self._idx[index] = [idx[i:i + self.batch_size] for i in xrange(0, length, self.batch_size)]
                self._idx_idx.extend([[index, i] for i in xrange(len(self._idx[index]))])
            _idx_idx = [self._idx_idx[i] for i in numpy.random.permutation(len(self._idx_idx))]
            self._idx_idx = _idx_idx
        else:
            raise

    def get_batch_data(self):

        for _idx_idx in self._idx_idx:
            cluster_id, idx = _idx_idx
            idxes = self._idx[cluster_id][idx]
            yield self.prepare_data([self.questions[cluster_id][idx] for idx in idxes],
                                    [self.columns[cluster_id][idx] for idx in idxes],
                                    [self.header_mask[cluster_id][idx] for idx in idxes])

    def prepare_data(self, questions, headers, column_masks):
        sample_size = len(questions)
        max_len_q = max([len(question) for question in questions])
        max_len_c = max([len(columns) for columns in headers])
        max_len_m = max([len(column_mask) for column_mask in column_masks])
        x_q = numpy.zeros((sample_size, max_len_q), dtype='int64')
        x_q_mask = numpy.zeros_like(x_q, dtype='float32')
        x_c = numpy.zeros((sample_size, max_len_c), dtype='int64')
        x_c_mask = numpy.zeros_like(x_c, dtype='float32')
        x_cm_mask = numpy.zeros((max_len_c - 1, max_len_m, sample_size), dtype='float32')
        x_cs_mask = numpy.zeros((max_len_m, sample_size), dtype='float32')
        for index, data in enumerate(zip(questions, headers, column_masks)):
            question, columns, column_mask = data
            print question
            print columns
            print column_mask
            len_q = len(question)
            len_c = len(columns)
            len_cm = len(column_mask)
            x_q[index, :len_q] = question
            x_q_mask[index, :len_q] = 1.
            x_c[index, :len_c] = columns
            x_c_mask[index, :len_c] = 1.
            for i, c_i in enumerate(column_mask):
                start, end = c_i
                x_cm_mask[start:end + 1, i, index] = 1.

            x_cm_mask[0, len_cm:, index] = 1.
            x_cs_mask[:len_cm, index] = 1.

        return x_q, x_q_mask, x_c, x_c_mask, x_cm_mask, x_cs_mask


class RetrievalHolder(object):
    def __init__(self, question_path, header_path,
                 batch_size=128, shuffle_mode='simple',
                 num_workers=5):
        self.question_path = question_path
        self.column_path = header_path
        self.shuffle_mode = shuffle_mode
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers < self.batch_size else self.batch_size
        self.questions = []
        self.columns = []
        self.pool = None
        self._idx = None

    def read_data(self):
        logging.info('Data Holder Read Data')
        f1 = open(self.question_path)
        f2 = open(self.column_path)

        while True:
            line1 = f1.readline()
            line2 = f2.readline()

            if not line1:
                break

            tokens_q = map(int, line1.strip().split())
            tokens_c = map(int, line2.strip().split())
            self.questions.append(tokens_q)
            self.columns.append(tokens_c)

        assert len(self.questions) == len(self.columns)

    def __len__(self):
        if not self.questions:
            self.read_data()
        return len(self.questions)

    def reset(self):
        if self.shuffle_mode is None:
            idx = numpy.arange(0, len(self))
            self._idx = [idx[i:i + self.batch_size] for i in xrange(0, len(self), self.batch_size)]
        elif self.shuffle_mode == 'simple':
            idx = numpy.random.permutation(len(self))
            self._idx = [idx[i:i + self.batch_size] for i in xrange(0, len(self), self.batch_size)]
        else:
            raise

    def get_batch_data(self):

        for idxes in self._idx:
            yield self.prepare_data([self.questions[idx] for idx in idxes],
                                    [self.columns[idx] for idx in idxes])

    def prepare_data(self, questions, headers):
        sample_size = len(questions)
        max_len_q = max([len(question) for question in questions])
        max_len_c = max([len(columns) for columns in headers])
        x_q = numpy.zeros((sample_size, max_len_q), dtype='int64')
        x_q_mask = numpy.zeros_like(x_q, dtype='float32')
        x_c = numpy.zeros((sample_size, max_len_c), dtype='int64')
        x_c_mask = numpy.zeros_like(x_c, dtype='float32')
        # x_cm_mask = numpy.zeros((max_len_c - 1, max_len_m, sample_size), dtype='float32')
        # x_cs_mask = numpy.zeros((max_len_m, sample_size), dtype='float32')
        for index, data in enumerate(zip(questions, headers)):
            question, columns = data
            # print question
            # print columns
            # print column_mask
            len_q = len(question)
            len_c = len(columns)
            x_q[index, :len_q] = question
            x_q_mask[index, :len_q] = 1.
            x_c[index, :len_c] = columns
            x_c_mask[index, :len_c] = 1.
        """
            for i, c_i in enumerate(column_mask):
                start, end = c_i
                x_cm_mask[start:end + 1, i, index] = 1.

            x_cm_mask[0, len_cm:, index] = 1.
            x_cs_mask[:len_cm, index] = 1.
            # print x_q[index]
            # print x_q_mask[index]
            # print x_c[index]
            # print x_c_mask[index]
            # print x_cm_mask[:, :, index]
            # print x_cs_mask[:, index]
            # print "**********************"
        """
        return x_q, x_q_mask, x_c, x_c_mask

