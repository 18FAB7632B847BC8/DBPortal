# coding: utf-8
from gevent import monkey
monkey.patch_all()
from json import dumps
import logging
from ChineseTokenizer import FullTokenizer
from config_default_retrieval import retrieval_model_config
from RetrievalModel import RetrievalModel
from basemodel import all_cosine
import numpy
from NL2SQL import NL2SQL
from redis_interface import RedisInterface
from interface_kafka import GeventConsumer
from json import loads
from config_default import simple_model_config
from TableParsing import parse_table


CLS_INDEX = 1
SEP_INDEX = 2


def prepare_data_nl2sql(data_list):
    sample_num = len(data_list)
    max_len_x = 0
    max_len_q = 0
    max_len_h = 0
    max_len_c = 0
    max_len_cw = 0
    max_len_y = 0
    len_q_list = []
    y_co_list = []
    for data in data_list:
        len_q = len(data['q'])
        len_q_list.append(len_q)
        len_h = len(data['h'])
        len_sc = len(data['sc'])
        len_wc = len(data['wc'])
        len_c = len(data['s_e_h'])
        y_co_list.append(data['co'])
        if max_len_q < len_q:
            max_len_q = len_q
        if max_len_h < len_h:
            max_len_h = len_h
        if max_len_x < len_q + len_h:
            max_len_x = len_q + len_h
        if max_len_c < len_c:
            max_len_c = len_c
        if max_len_y < len_sc + len_wc:
            max_len_y = len_sc + len_wc
        for s_e in data['s_e_h']:
            len_cw = s_e[1] - s_e[0] + 1
            if max_len_cw < len_cw:
                max_len_cw = len_cw

    x = numpy.zeros((sample_num, max_len_x + 1), dtype='int32')
    x[:, 0] = CLS_INDEX
    x_type = numpy.zeros_like(x, dtype='int32')
    x_mask = numpy.zeros_like(x, dtype='float32')
    y = numpy.zeros((sample_num, max_len_y), dtype='int32')
    y_mask = numpy.zeros_like(y, dtype='float32')
    y_agg = numpy.zeros_like(y, dtype='int32')
    y_agg_mask = numpy.zeros_like(y, dtype='float32')
    y_op = numpy.zeros_like(y, dtype='int32')
    y_op_mask = numpy.zeros_like(y, dtype='float32')
    len_q = numpy.array(len_q_list, dtype='int32')
    mask_q = numpy.zeros((sample_num, max_len_q), dtype='float32')
    s_e_h = numpy.zeros((sample_num, max_len_c, 2), dtype='int32')
    mask_h_w = numpy.zeros((max_len_cw, sample_num, max_len_c), dtype='float32')
    mask_h_s = numpy.zeros((sample_num, max_len_c), dtype='float32')
    y_vs = numpy.zeros_like(y, dtype='int32')
    y_vs_mask = numpy.zeros_like(y, dtype='float32')
    y_ve = numpy.zeros_like(y, dtype='int32')
    y_ve_mask = numpy.zeros_like(y, dtype='float32')
    y_co = numpy.array(y_co_list, dtype='int32')
    for index, data in enumerate(data_list):
        q_list = data['q']
        mask_q[index][:len(q_list)] = 1.
        h_list = data['h']
        x[index][1:1 + len(q_list)] = q_list
        x[index][1 + len(q_list):1 + len(q_list) + len(h_list)] = h_list
        h_type_list = data['h_type']
        x_type[index][1 + len(q_list):1 + len(q_list) + len(h_list)] = h_type_list
        x_mask[index][:1 + len(q_list) + len(h_list)] = 1.
        sc_list = data['sc']
        wc_list = data['wc']
        y[index][:len(sc_list)] = sc_list
        y[index][len(sc_list):len(sc_list) + len(wc_list)] = wc_list
        y_mask[index][:len(sc_list) + len(wc_list)] = 1.
        agg_list = data['agg']
        y_agg[index][:len(agg_list)] = agg_list
        y_agg_mask[index][:len(agg_list)] = 1.
        op_list = data['op']
        y_op[index][len(agg_list):len(agg_list) + len(op_list)] = op_list
        y_op_mask[index][len(agg_list):len(agg_list) + len(op_list)] = 1.
        s_e_list = data['s_e_h']
        s_e_h[index, :len(s_e_list)] = s_e_list
        mask_h_s[index, :len(s_e_list)] = 1.
        mask_h_w[0, :, :] = 1.
        for j, s_e in enumerate(s_e_list):
            s_e_len = s_e[1] - s_e[0] + 1
            mask_h_w[:s_e_len, index, j] = 1.
        vs_list = data['vs']
        y_vs[index][len(agg_list):len(agg_list) + len(vs_list)] = vs_list
        y_vs[index][:len(agg_list)] = vs_list[-1]
        y_vs[index][len(agg_list) + len(vs_list):] = vs_list[-1]
        y_vs_mask[index][len(agg_list):len(agg_list) + len(vs_list)] = 1.
        ve_list = data['ve']
        y_ve[index][len(agg_list):len(agg_list) + len(ve_list)] = ve_list
        y_ve[index][:len(agg_list)] = ve_list[-1]
        y_ve[index][len(agg_list) + len(ve_list):] = ve_list[-1]
        y_ve_mask[index][len(agg_list):len(agg_list) + len(ve_list)] = 1.

    return x, x_type, x_mask, y, y_mask, y_agg, y_agg_mask, y_op, y_op_mask, len_q, mask_q, s_e_h, \
           mask_h_w, mask_h_s, y_vs, y_vs_mask, y_ve, y_ve_mask, y_co


class MRedis(object):
    def __init__(self, config, *args, **kwargs):
        self.rds = RedisInterface(config, *args, **kwargs)
        self.conn = self.rds.conn

    def conn(self):
        return self.conn

    def hset(self, hname, key, value):
        def _hset(pipe, hname, key, value):
            pipe.multi()
            pipe.hset(hname, key, value)

        exe_r = self.rds.transaction(self.conn, _hset, [hname, key, value])

        return exe_r

    def hget(self, hname, key):
        def _hget(pipe, hname, key):
            pipe.multi()
            pipe.hget(hname, key)

        exe_r = self.rds.transaction(self.conn, _hget, [hname, key])

        return exe_r


def init_retrieval():
    ret_model = RetrievalModel(config=retrieval_model_config)
    ret_model.init_model()
    ret_model.load("../data/save/retrieval/ss_model.npz")
    logging.info("build retrieval model")
    ret_model.build_valid_single()
    logging.info("build retrieval model ss finished")

    return ret_model


def init_nl2sql():
    nl2sql_model = NL2SQL(simple_model_config)
    nl2sql_model.init_model()
    logging.info("Loading model")
    nl2sql_model.load("../data/save/nl2sql_model.npz")
    logging.info("Build NL2SQL Model")
    nl2sql_model.build_sampler()
    logging.info("Build NL2SQL Model Finished")

    return nl2sql_model


def preprocess_ss(ql, BT):
    q_list = []
    for q in ql:
        q_list.append([CLS_INDEX] + BT.convert_tokens_to_ids(BT.tokenize(q)) + [SEP_INDEX])
    n_sample = len(q_list)
    n_tok = max([len(q) for q in q_list])
    x_q = numpy.zeros((n_sample, n_tok), dtype='int64')
    x_q_mask = numpy.zeros_like(x_q, dtype='float32')
    for i, q in enumerate(q_list):
        x_q_mask[i, :len(q)] = 1.
    x_q_type = numpy.zeros_like(x_q, dtype='int64')

    return x_q, x_q_mask, x_q_type


def schema_selection(ret_model, q_list, BT):
    data = preprocess_ss(q_list, BT)
    emb_qs = ret_model.f_val_s(*data)
    emb_c = numpy.load("../data/schema_embs.npy")
    db_idxes = []

    for emb_q in emb_qs:
        tiled_emb_q = numpy.tile(emb_q, [emb_c.shape[0], 1])

        sim = all_cosine(tiled_emb_q, emb_c)[0]  # n_sample

        db_idx = int(sim.argmax())
        db_idxes.append(db_idx)

    return db_idxes


def preprocess_nl2sql(q_list, data_list, BT):

    for idx, q in enumerate(q_list):
        q_toks = BT.tokenize(q)
        q_ids = BT.convert_tokens_to_ids(q_toks)
        data_list[idx]['input_sequence'] = [CLS_INDEX] + q_ids + [SEP_INDEX, SEP_INDEX] + data_list[idx]['input_sequence']
        data_list[idx]['separator'] = [1, len(q_toks) + 2] + [i + len(q_toks) + 3 for i in data_list[idx]['separator']]
        data_list[idx]['input_type_0'] = [0] * (len(q_toks) + 3) + data_list[idx]['input_type_0']
        data_list[idx]['input_type_1'] = [0] * (len(q_toks) + 3) + data_list[idx]['input_type_1']
        data_list[idx]['input_type_2'] = [0] * (len(q_toks) + 3) + data_list[idx]['input_type_2']
        data_list[idx]['question_token'] = q_toks

    return prepare_data_nl2sql(data_list)


def nl2sql(model, q_list, db_idxes, database, BT, rds, timestep_list):
    data_list = []
    for idx in db_idxes:
        data_list.append(loads(rds.hget('DB_data', str(idx))))

    data = preprocess_nl2sql(q_list, data_list, BT)

    for idx, d in enumerate(data):
        sql = model.construct_comp_sql(d, database)
        r = {
            "q": q_list[idx],
            "sql": sql,
            "db_id": data_list[idx]['db_id'],
        }

        rds.hset('Inter_data', timestep_list[idx], dumps(r))


def polling(kafka_configs, redis_config):
    model_r = init_retrieval()
    model_nl2sql = init_nl2sql()
    BT = FullTokenizer(open("../data/vocab"))
    rds = MRedis(redis_config)
    database = parse_table()

    def func(m_list, mr, mn, tokenizer):
        q_list = [ele['question'] for ele in m_list]
        t_list = [ele['timestep'] for ele in m_list]
        db_idxes = schema_selection(mr, q_list, tokenizer)

        nl2sql(mn, q_list, db_idxes, database, tokenizer, rds, t_list)

    kafka_consumer = GeventConsumer(kafka_configs, func)

    kafka_consumer.consume(model_r, model_nl2sql, BT)













