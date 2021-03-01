#!/usr/bin/env python

import numpy
from json import loads


class DataHolder(object):
    def __init__(self, input_path=None, output_path=None, partition="train",
                 batch_size=16, shuffle_mode=None, name="Data Holder"):
        self.input_path = input_path
        self.output_path = output_path
        self.partition = partition
        self.batch_size = batch_size
        self.shuffle_mode = shuffle_mode
        self.name = name
        self._idx = None
        self.unique_lengths = None
        self.length_index = dict()

        self.input_data = []
        self.output_data = []

    def read_data(self):
        if self.partition == "train":
            with open(self.input_path) as f:
                for line in f:
                    in_data = loads(line.strip())
                    self.input_data.append(in_data)

            with open(self.output_path) as f:
                for line in f:
                    ot_data = loads(line.strip())
                    self.output_data.append(ot_data)

            assert len(self.input_data) == len(self.output_data)

            if self.shuffle_mode == "hmgns":
                tgt_lengths = [len(d['s_column']) for d in self.output_data]
                self.unique_lengths = numpy.unique(tgt_lengths)
                for l in self.unique_lengths:
                    self.length_index[l] = numpy.where(tgt_lengths == l)[0]
        else:
            assert self.partition == "dev"
            with open(self.input_path) as f:
                for line in f:
                    in_data = loads(line.strip())
                    self.input_data.append(in_data)

    def re_read(self):
        assert self.partition == "dev"
        self._idx = None
        self.unique_lengths = None
        self.length_index = dict()

        self.input_data = []
        self.output_data = []

        self.read_data()
        self.reset()

    def __len__(self):
        return len(self.input_data)

    def reset(self):
        if self.shuffle_mode is None:
            idx = numpy.arange(0, len(self))
            self._idx = [idx[i:i + self.batch_size] for i in xrange(0, len(self), self.batch_size)]
        elif self.shuffle_mode == 'simple':
            idx = numpy.random.permutation(len(self))
            self._idx = [idx[i:i + self.batch_size] for i in xrange(0, len(self), self.batch_size)]
        elif self.shuffle_mode == 'hmgns':
            self._idx = self.get_homogeneous_idx()
        else:
            raise

    def get_homogeneous_idx(self):
        idx = []
        self.unique_lengths = numpy.random.permutation(self.unique_lengths)
        for l in self.unique_lengths:
            self.length_index[l] = numpy.random.permutation(self.length_index[l])

        for length in self.unique_lengths:
            indexes = self.length_index[length]
            if len(indexes) <= self.batch_size:
                idx.append(indexes)
            else:
                idx += [indexes[i:i + self.batch_size] for i in xrange(0, len(indexes), self.batch_size)]

        return idx

    def get_batch_data(self):
        if self.partition == "train":
            for idxs in self._idx:
                # idxs is a list
                yield self.prepare_data([(self.input_data[idx], self.output_data[idx]) for idx in idxs])
        elif self.partition == "dev":
            for idxs in self._idx:
                # idxs is a list
                yield self.prepare_data_dev([(self.input_data[idx]) for idx in idxs])

    @staticmethod
    def prepare_data(data_list):
        sample_num = len(data_list)
        max_num_input_tok = 1
        max_num_question_tok = 1
        max_num_column_tok = 1
        max_num_table_tok = 1
        max_num_psql_tok = 1
        max_num_column = 1
        max_num_table = 1
        max_column_per_table = 1
        max_num_join = 1
        max_num_output_tok = 1
        max_num_from_table = 1

        for data in data_list:
            in_d, ot_d = data
            max_num_input_tok = max(max_num_input_tok, len(in_d['input_sequence']))
            q_s, q_e, h_s, h_e, t_s, t_e, p_s, p_e = in_d['separator']
            max_num_question_tok = max(max_num_question_tok, q_e - q_s)
            max_num_psql_tok = max(max_num_psql_tok, p_e - p_s)
            for c_t_s_e in in_d['column_name_start_end']:
                c_t_s, c_t_e = c_t_s_e
                max_num_column_tok = max(max_num_column_tok, c_t_e - c_t_s)
            max_num_column = max(max_num_column, len(in_d['column_name_start_end']))

            for t_t_s_e in in_d['table_name_start_end']:
                t_t_s, t_t_e = t_t_s_e
                max_num_table_tok = max(max_num_table_tok, t_t_e - t_t_s)
            max_num_table = max(max_num_table, len(in_d['table_name_start_end']))

            for join_list in in_d['join_tables']:
                max_num_join = max(max_num_join, len(join_list))

            for t_c_s_e in in_d['table_column_start_end']:
                t_c_s, t_c_e = t_c_s_e
                max_column_per_table = max(max_column_per_table, t_c_e - t_c_s)

            max_num_from_table = max(max_num_from_table, len(ot_d['from_table']))

            max_num_output_tok = max(max_num_output_tok, len(ot_d['s_column']))

        # BERT
        input_sequence = numpy.zeros([sample_num, max_num_input_tok], dtype='int32')
        input_mask = numpy.zeros_like(input_sequence, dtype='float32')
        input_type_0 = numpy.zeros_like(input_sequence, dtype='int32')
        input_type_1 = numpy.zeros_like(input_sequence, dtype='int32')
        input_type_2 = numpy.zeros_like(input_sequence, dtype='int32')

        # ENCODER
        join_table = numpy.zeros([sample_num, max_num_table, max_num_join], dtype="int32")
        join_mask = numpy.zeros_like(join_table, dtype="float32")
        join_mask[:, :, 0] = 1.
        separator = numpy.zeros([sample_num, 8], dtype="int32")
        s_e_h = numpy.zeros([sample_num, max_num_column, 2], dtype="int32")
        s_e_t = numpy.zeros([sample_num, max_num_table, 2], dtype="int32")
        s_e_t_h = numpy.zeros([sample_num, max_num_table - 1, 2], dtype="int32")
        h2t_idx = numpy.zeros([sample_num, max_num_column - 2], dtype="int32")
        q_mask = numpy.zeros([sample_num, max_num_question_tok], dtype="float32")
        t_mask = numpy.zeros([sample_num, max_num_table], dtype="float32")
        p_mask = numpy.zeros([sample_num, max_num_psql_tok], dtype="float32")
        t_w_mask = numpy.zeros([sample_num, max_num_table, max_num_table_tok], dtype="float32")
        t_w_mask[:, :, 0] = 1.
        t_h_mask = numpy.zeros([sample_num, max_num_table - 1, max_column_per_table], dtype="float32")
        t_h_mask[:, :, 0] = 1.
        h_mask = numpy.zeros([sample_num, max_num_column], dtype="float32")
        h_w_mask = numpy.zeros([sample_num, max_num_column, max_num_column_tok], dtype="float32")
        h_w_mask[:, :, 0] = 1.

        # DECODER
        y_column = numpy.zeros([sample_num, max_num_output_tok], dtype="int32")
        y_type2 = numpy.zeros_like(y_column, dtype="int32")
        y_agg = numpy.zeros_like(y_column, dtype="int32")
        y_op = numpy.zeros_like(y_column, dtype="int32")
        y_vls = numpy.zeros_like(y_column, dtype="int32")
        y_vle = numpy.zeros_like(y_column, dtype="int32")
        y_vrs = numpy.zeros_like(y_column, dtype="int32")
        y_vre = numpy.zeros_like(y_column, dtype="int32")
        y_cd = numpy.zeros_like(y_column, dtype="int32")
        y_vn = numpy.zeros_like(y_column, dtype="int32")

        y_c_mask = numpy.zeros_like(y_column, dtype="float32")
        y_cw_mask = numpy.zeros_like(y_column, dtype="float32")
        y_w_mask = numpy.zeros_like(y_column, dtype="float32")
        y_cwo_mask = numpy.zeros_like(y_column, dtype="float32")

        y_t = numpy.zeros([sample_num, max_num_from_table], dtype="int32")
        y_t_mask = numpy.zeros_like(y_t, dtype="float32")

        y_co = numpy.zeros([sample_num], dtype="int32")
        y_d = numpy.zeros_like(y_co, dtype="int32")
        y_type1 = numpy.zeros_like(y_co, dtype="int32")
        y_o = numpy.zeros_like(y_co, dtype="int32")
        y_c = numpy.zeros_like(y_co, dtype="int32")
        y_l = numpy.zeros_like(y_co, dtype="int32")
        y_nf = numpy.zeros_like(y_co, dtype="int32")

        for idx, data in enumerate(data_list):
            # ENCODER
            in_d, ot_d = data
            input_sequence[idx, :len(in_d['input_sequence'])] = in_d['input_sequence']
            input_mask[idx, :len(in_d['input_sequence'])] = 1.
            input_type_0[idx, :len(in_d['input_sequence'])] = in_d['input_type_0']
            input_type_1[idx, :len(in_d['input_sequence'])] = in_d['input_type_1']
            input_type_2[idx, :len(in_d['input_sequence'])] = in_d['input_type_2']

            for i, jts in enumerate(in_d['join_tables']):
                if len(jts) == 0:
                    join_mask[idx, i, 0] = 1.
                else:
                    join_table[idx, i, :len(jts)] = jts
                    join_mask[idx, i, :len(jts)] = 1.
            separator[idx] = in_d['separator']
            q_s, q_e, _, _, _, _, p_s, p_e = in_d['separator']
            q_mask[idx, :q_e - q_s] = 1.
            p_mask[idx, :p_e - p_s] = 1.
            s_e_h[idx, :len(in_d['column_name_start_end'])] = in_d['column_name_start_end']
            h_mask[idx, :len(in_d['column_name_start_end'])] = 1.
            for ci, s_e in enumerate(in_d['column_name_start_end']):
                c_w_len = s_e[1] - s_e[0]
                h_w_mask[idx, ci, :c_w_len] = 1.

            s_e_t[idx, :len(in_d['table_name_start_end'])] = in_d['table_name_start_end']
            t_mask[idx, :len(in_d['table_name_start_end'])] = 1.
            for ti, s_e in enumerate(in_d['table_name_start_end']):
                t_w_len = s_e[1] - s_e[0]
                t_w_mask[idx, ti, :t_w_len] = 1.
            s_e_t_h[idx, :len(in_d['table_column_start_end'])] = in_d['table_column_start_end']
            for ti, s_e in enumerate(in_d['table_column_start_end']):
                t_h_len = s_e[1] - s_e[0]
                t_h_mask[idx, ti, :t_h_len] = 1.
            h2t_idx[idx, :len(in_d['column_to_table']) - 2] = in_d['column_to_table'][2:]

            # DECODER
            y_column[idx, :len(ot_d['s_column'])] = ot_d['s_column']
            y_type2[idx, :len(ot_d['s_column'])] = ot_d['s_type2']
            w_s = ot_d['s_type2'].index(18)
            o_s = ot_d['s_type2'].index(19)
            g_s = ot_d['s_type2'].index(20)
            y_c_mask[idx, :len(ot_d['s_column'])] = 1.
            y_cw_mask[idx, :o_s] = 1.
            y_w_mask[idx, w_s:o_s] = 1.
            y_cwo_mask[idx, :g_s] = 1.
            y_agg[idx, :len(ot_d['s_column'])] = ot_d['s_agg']
            y_op[idx, :len(ot_d['s_column'])] = ot_d['s_op']
            y_vls[idx, :len(ot_d['s_column'])] = ot_d['s_value_left_start']
            y_vls[idx, len(ot_d['s_column']):] = y_vls[idx, 0]
            y_vle[idx, :len(ot_d['s_column'])] = ot_d['s_value_left_end']
            y_vle[idx, len(ot_d['s_column']):] = y_vls[idx, 0]
            y_vrs[idx, :len(ot_d['s_column'])] = ot_d['s_value_right_start']
            y_vrs[idx, len(ot_d['s_column']):] = y_vls[idx, 0]
            y_vre[idx, :len(ot_d['s_column'])] = ot_d['s_value_right_end']
            y_vre[idx, len(ot_d['s_column']):] = y_vls[idx, 0]
            y_cd[idx, :len(ot_d['s_column'])] = ot_d['s_distinct']
            y_vn[idx, :len(ot_d['s_column'])] = ot_d['s_nested_value']
            y_t[idx, :len(ot_d['from_table'])] = ot_d['from_table']
            if max(ot_d['from_table']) >= t_mask.shape[1]:
                print ot_d
                print in_d

            y_t_mask[idx, :len(ot_d['from_table'])] = 1.
            y_co[idx] = ot_d['connector']
            y_d[idx] = ot_d['distinct']
            y_type1[idx] = ot_d['type1']
            y_o[idx] = ot_d['order']
            y_c[idx] = ot_d['combine']
            y_l[idx] = ot_d['limit']
            y_nf[idx] = ot_d['nested_from']

        return input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
            join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
            q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, \
            y_column, y_type2, y_agg, y_op, y_vls, y_vle, y_vrs, y_vre, y_cd, y_vn, \
            y_c_mask, y_cw_mask, y_w_mask, y_cwo_mask, \
            y_t, y_t_mask, \
            y_co, y_d, y_type1, y_o, y_c, y_l, y_nf

    @staticmethod
    def prepare_data_dev(data_list):
        sample_num = len(data_list)
        max_num_input_tok = 1
        max_num_question_tok = 1
        max_num_column_tok = 1
        max_num_table_tok = 1
        max_num_psql_tok = 1
        max_num_column = 1
        max_num_table = 1
        max_column_per_table = 1
        max_num_join = 1

        for data in data_list:
            in_d = data
            max_num_input_tok = max(max_num_input_tok, len(in_d['input_sequence']))
            q_s, q_e, h_s, h_e, t_s, t_e, p_s, p_e = in_d['separator']
            max_num_question_tok = max(max_num_question_tok, q_e - q_s)
            max_num_psql_tok = max(max_num_psql_tok, p_e - p_s)
            for c_t_s_e in in_d['column_name_start_end']:
                c_t_s, c_t_e = c_t_s_e
                max_num_column_tok = max(max_num_column_tok, c_t_e - c_t_s)
            max_num_column = max(max_num_column, len(in_d['column_name_start_end']))

            for t_t_s_e in in_d['table_name_start_end']:
                t_t_s, t_t_e = t_t_s_e
                max_num_table_tok = max(max_num_table_tok, t_t_e - t_t_s)
            max_num_table = max(max_num_table, len(in_d['table_name_start_end']))

            for join_list in in_d['join_tables']:
                max_num_join = max(max_num_join, len(join_list))

            for t_c_s_e in in_d['table_column_start_end']:
                t_c_s, t_c_e = t_c_s_e
                max_column_per_table = max(max_column_per_table, t_c_e - t_c_s)

        # BERT
        input_sequence = numpy.zeros([sample_num, max_num_input_tok], dtype='int32')
        input_mask = numpy.zeros_like(input_sequence, dtype='float32')
        input_type_0 = numpy.zeros_like(input_sequence, dtype='int32')
        input_type_1 = numpy.zeros_like(input_sequence, dtype='int32')
        input_type_2 = numpy.zeros_like(input_sequence, dtype='int32')

        # ENCODER
        join_table = numpy.zeros([sample_num, max_num_table, max_num_join], dtype="int32")
        join_mask = numpy.zeros_like(join_table, dtype="float32")
        join_mask[:, :, 0] = 1.
        separator = numpy.zeros([sample_num, 8], dtype="int32")
        s_e_h = numpy.zeros([sample_num, max_num_column, 2], dtype="int32")
        s_e_t = numpy.zeros([sample_num, max_num_table, 2], dtype="int32")
        s_e_t_h = numpy.zeros([sample_num, max_num_table - 1, 2], dtype="int32")
        h2t_idx = numpy.zeros([sample_num, max_num_column - 2], dtype="int32")
        q_mask = numpy.zeros([sample_num, max_num_question_tok], dtype="float32")
        t_mask = numpy.zeros([sample_num, max_num_table], dtype="float32")
        p_mask = numpy.zeros([sample_num, max_num_psql_tok], dtype="float32")
        t_w_mask = numpy.zeros([sample_num, max_num_table, max_num_table_tok], dtype="float32")
        t_w_mask[:, :, 0] = 1.
        t_h_mask = numpy.zeros([sample_num, max_num_table - 1, max_column_per_table], dtype="float32")
        t_h_mask[:, :, 0] = 1.
        h_mask = numpy.zeros([sample_num, max_num_column], dtype="float32")
        h_w_mask = numpy.zeros([sample_num, max_num_column, max_num_column_tok], dtype="float32")
        h_w_mask[:, :, 0] = 1.
        input_blocks = []

        for idx, data in enumerate(data_list):
            # ENCODER
            in_d = data
            input_blocks.append(in_d)
            input_sequence[idx, :len(in_d['input_sequence'])] = in_d['input_sequence']
            input_mask[idx, :len(in_d['input_sequence'])] = 1.
            input_type_0[idx, :len(in_d['input_sequence'])] = in_d['input_type_0']
            input_type_1[idx, :len(in_d['input_sequence'])] = in_d['input_type_1']
            input_type_2[idx, :len(in_d['input_sequence'])] = in_d['input_type_2']

            for i, jts in enumerate(in_d['join_tables']):
                if len(jts) == 0:
                    join_mask[idx, i, 0] = 1.
                else:
                    join_table[idx, i, :len(jts)] = jts
                    join_mask[idx, i, :len(jts)] = 1.
            separator[idx] = in_d['separator']
            q_s, q_e, _, _, _, _, p_s, p_e = in_d['separator']
            q_mask[idx, :q_e - q_s] = 1.
            p_mask[idx, :p_e - p_s] = 1.
            s_e_h[idx, :len(in_d['column_name_start_end'])] = in_d['column_name_start_end']
            h_mask[idx, :len(in_d['column_name_start_end'])] = 1.
            for ci, s_e in enumerate(in_d['column_name_start_end']):
                c_w_len = s_e[1] - s_e[0]
                h_w_mask[idx, ci, :c_w_len] = 1.

            s_e_t[idx, :len(in_d['table_name_start_end'])] = in_d['table_name_start_end']
            t_mask[idx, :len(in_d['table_name_start_end'])] = 1.
            for ti, s_e in enumerate(in_d['table_name_start_end']):
                t_w_len = s_e[1] - s_e[0]
                t_w_mask[idx, ti, :t_w_len] = 1.
            s_e_t_h[idx, :len(in_d['table_column_start_end'])] = in_d['table_column_start_end']
            for ti, s_e in enumerate(in_d['table_column_start_end']):
                t_h_len = s_e[1] - s_e[0]
                t_h_mask[idx, ti, :t_h_len] = 1.
            h2t_idx[idx, :len(in_d['column_to_table']) - 2] = in_d['column_to_table'][2:]

        return input_sequence, input_mask, input_type_0, input_type_1, input_type_2, \
            join_table, join_mask, separator, s_e_h, s_e_t, s_e_t_h, h2t_idx, \
            q_mask, t_mask, p_mask, t_w_mask, t_h_mask, h_mask, h_w_mask, input_blocks


if __name__ == '__main__':
    data_holder = DataHolder(
        input_path="../data/build/simple_dev/build_dev.inputs",
        output_path=None,
        partition="dev",
        batch_size=4,
        shuffle_mode=None,
    )

    data_holder.read_data()
    data_holder.reset()
    for data in data_holder.get_batch_data():
        for d in data:
            print d
        print "*****************"







