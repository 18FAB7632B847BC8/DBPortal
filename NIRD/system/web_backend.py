# coding: utf-8
from gevent import monkey
monkey.patch_all()
import gevent
from flask import Flask
from flask import request, render_template
from functools import wraps
from flask import make_response
from gevent import pywsgi
from json import dumps
import logging
from ChineseTokenizer import FullTokenizer
from config_default_retrieval import retrieval_model_config
from RetrievalModel import RetrievalModel
from basemodel import all_cosine
import numpy
from NL2SQL import NL2SQL
from json import loads

from config_default import simple_model_config, simple_val_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

app = Flask(__name__)


def allow_cross_domain(fun):
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        rst = make_response(fun(*args, **kwargs))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
        allow_headers = "Referer,Accept,Origin,User-Agent"
        rst.headers['Access-Control-Allow-Headers'] = allow_headers
        return rst
    return wrapper_fun


def init_retrieval():
    ret_model = RetrievalModel(config=retrieval_model_config)
    ret_model.init_model()
    ret_model.load("../data/save/retrieval/xxxxx.npz")
    logging.info("build retrieval model")
    ret_model.build_valid_single()
    logging.info("build retrieval model finished")

    return ret_model


def init_nl2sql():
    nl2sql_model = NL2SQL(simple_model_config)
    nl2sql_model.init_model()
    logging.info("Loading model")
    nl2sql_model.load("../data/save/xxxxx.npz")
    logging.info("Build NL2SQL Model")
    nl2sql_model.build_sampler()
    logging.info("Build NL2SQL Model Finished")

    return nl2sql_model

ret_model = init_retrieval()
nl2sql_model = init_nl2sql()


def tokenize(text):
    vocab_path = "../data/vocab.txt"
    BT = FullTokenizer(vocab_path)
    tokens = BT.tokenize(text)
    ids = [101] + BT.convert_tokens_to_ids(tokens) + [102]
    return ids, tokens


def preprocess(type, q, h=None, h_type=None):
    cls_index = 101
    sep_index = 102
    q_tok, q_text = tokenize(q)
    if type == 1:
        x_q = numpy.zeros((1, len(q_tok)), dtype='int64')
        x_q[0] = q_tok
        x_q_mask = numpy.ones_like(x_q, dtype='float32')
        x_q_type = numpy.zeros_like(x_q, dtype='int64')

        return x_q, x_q_mask, x_q_type
    else:
        assert type == 2
        q_tok = q_tok[1:]
        q_len = len(q_tok)
        h_len = len(h)
        len_q_list = [q_len]
        s_e_h_list = []
        last_id = -1
        while True:
            start = last_id + 1
            try:
                end = h.index(sep_index, start)
                last_id = end
                s_e_h_list.append([start, end])
            except ValueError:
                break
        len_c = len(s_e_h_list)
        max_len_cw = 0
        for s_e in s_e_h_list:
            len_cw = s_e[1] - s_e[0] + 1
            if max_len_cw < len_cw:
                max_len_cw = len_cw
        x = numpy.zeros((1, q_len + h_len + 1), dtype='int32')
        x[:, 0] = cls_index
        x_type = numpy.zeros_like(x, dtype='int32')
        x_mask = numpy.zeros_like(x, dtype='float32')
        len_q = numpy.array(len_q_list, dtype='int32')
        mask_q = numpy.zeros((1, q_len), dtype='float32')
        s_e_h = numpy.zeros((1, len_c, 2), dtype='int32')
        mask_h_w = numpy.zeros((max_len_cw, 1, len_c), dtype='float32')
        mask_h_s = numpy.zeros((1, len_c), dtype='float32')

        mask_q[0][:len(q_tok)] = 1.
        x[0][1:1 + len(q_tok)] = q_tok
        x[0][1 + len(q_tok):1 + len(q_tok) + len(h)] = h
        x_type[0][1 + len(q_tok):1 + len(q_tok) + len(h)] = h_type
        x_mask[0][:1 + len(q_tok) + len(h)] = 1.
        s_e_h[0, :len(s_e_h_list)] = s_e_h_list
        mask_h_s[0, :len(s_e_h_list)] = 1.
        mask_h_w[0, :, :] = 1.
        for j, s_e in enumerate(s_e_h_list):
            s_e_len = s_e[1] - s_e[0] + 1
            mask_h_w[:s_e_len, 0, j] = 1.

        return x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s, q_text


@app.route("/schema_selection", methods=['POST'])
@allow_cross_domain
def schema_selection():
    q = request.form['q']
    logging.info("Request form_q: %s" % q)
    data = preprocess(1, q)
    emb_q = ret_model.f_val_s(*data)
    emb_c = numpy.load("../data/schema_embs.npy")

    tiled_emb_q = numpy.tile(emb_q, [emb_c.shape[0], 1])

    sim = all_cosine(tiled_emb_q, emb_c)[0]  # n_sample

    table_idx = int(sim.argmax())

    return str(table_idx)


@app.route("/nl2sql", methods=['POST'])
@allow_cross_domain
def nl2sql():
    q = request.form['q']
    h = loads(request.form['h'])
    h_type = loads(request.form['h_type'])

    logging.info("Request form_q: %s" % q)
    logging.info("Request form_h: %s" % h)
    logging.info("Request form_q: %s" % h_type)

    data = preprocess(2, q, h, h_type)

    x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s, q_text = data

    results = nl2sql_model.beam_search(x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s,
                                       maxlen=10, beam_size=2)

    y_sc_, y_wc_, y_agg_, y_op_, y_vs_, y_ve_, y_co_ = results

    values = []

    for vs, ve in zip(y_vs_[:-1], y_ve_[:-1]):
        value = "".join(q_text[vs:ve + 1]).encode('utf-8')
        if not value:
            value = "NULL"
        values.append(value)

    sql_dict = {
        "select_column": y_sc_[:-1].tolist(),
        "where_column": y_wc_[:-1].tolist(),
        'agg': y_agg_[:-1].tolist(),
        'op': y_op_[:-1].tolist(),
        'connector': y_co_,
        "values": values
    }

    return sql_dict


def test_nl2sql(q, h, h_type):
    data = preprocess(2, q, h, h_type)

    x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s, q_text = data

    results = nl2sql_model.beam_search(x, x_type, x_mask, len_q, s_e_h, mask_h_w, mask_q, mask_h_s,
                                       maxlen=10, beam_size=2)

    y_sc_, y_wc_, y_agg_, y_op_, y_vs_, y_ve_, y_co_ = results

    values = []

    for vs, ve in zip(y_vs_, y_ve_):
        value = "".join(q_text[vs:ve + 1]).encode('utf-8')
        if not value:
            value = "NULL"
        values.append(value)

    sql_dict = {
        "select_column": y_sc_.tolist(),
        "where_column": y_wc_.tolist(),
        'agg': y_agg_.tolist(),
        'op': y_op_.tolist(),
        'connector': y_co_,
        "values": values
    }

    return sql_dict
