from gevent import monkey
from flask import Flask
from flask import request, render_template
from flask_bootstrap import Bootstrap
from functools import wraps
from flask import make_response
from gevent import pywsgi
import requests
from json import dumps, loads
from _interface_mysql import db
from config.config_default import config_mysql, config_redis
from redis_interface import RedisInterface
import logging

monkey.patch_all()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'
Bootstrap(app)

agg_list = [
    "", "AVG", "MAX", "MIN", "COUNT", "SUM"
]

op_list = [
    "", ">", "<", "=", "<>"
]

conn = {
    0: "",
    1: "AND",
    2: "OR"
}


def sql2str(sql_dict, table_id, types, header):
    print sql_dict
    sql = "SELECT "
    sql_real = "SELECT "
    columns = []
    columns_real = []
    for agg, sc in zip(sql_dict['agg'], sql_dict['select_column']):
        column = "col_%d" % sc
        column_real = "%s" % header[sc - 1]
        if agg:
            column = "%s(%s)" % (agg_list[agg], column)
            column_real = "%s(%s)" % (agg_list[agg], column_real)
        columns.append(column)
        columns_real.append(column_real)
    sql += " %s" % " , ".join(columns)
    sql_real += " %s" % " , ".join(columns_real)
    sql += " FROM Table_%s" % table_id
    sql_real += " FROM Table_%s" % table_id
    if len(sql_dict['where_column']) > 0:
        sql += " WHERE"
        sql_real += " WHERE"
        w_cols = []
        w_real_cols = []
        for op, wc, v in zip(sql_dict['op'], sql_dict['where_column'], sql_dict['values']):
            v = v.replace("##", "")
            if types[wc - 1] == "text" or not v.isdigit():
                value = '"%s"' % v
            else:
                value = v
            w_cols.append('col_%d %s %s' % (wc, op_list[op], value))
            w_real_cols.append('%s %s %s' % (header[wc - 1], op_list[op], value))
        co = conn[sql_dict['connector']]
        sql += " %s" % ((" %s " % co).join(w_cols))
        sql_real += " %s" % ((" %s " % co).join(w_real_cols))
    return sql, sql_real


class MRedis(object):
    def __init__(self):
        self.rds_itf = RedisInterface(config_redis)
        self.conn = self.rds_itf.conn()

    def hget(self, pipe, name, key):
        pipe.multi()
        pipe.hget(name, key)

    def get_table(self, tidx):
        result = self.rds_itf.transaction(
            self.conn, self.hget, ["tables", "table_%d" % tidx]
        )

        return result


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


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
@allow_cross_domain
def index():
    if request.method == "POST":
        mredis = MRedis()
        url_ss = "http://xxx.xx.x.x:xxxx"
        url_sp = "http://xxx.xx.x.x:xxx"
        q = request.form.get("nl-question")
        logging.info("Question:%s" % q)

        data_r = {
            "q": q
        }

        res_r = requests.post("%s/schema_selection" % url_ss, data=data_r)

        tidx = int(res_r.content)

        table = loads(mredis.get_table(tidx)[0])
        h = map(int, table['build_header'])
        h_type = map(int, table['build_type'])
        table_id = table['table_id']
        logging.info("Target Table: \nTable_%s" % table_id)

        data_n = {
            "q": q,
            "h": dumps(h),
            "h_type": dumps(h_type)
        }

        res_n = requests.post("%s/nl2sql" % url_sp, data=data_n)
        sql, sql_real = sql2str(loads(res_n.content), table_id, table['types'], table['header'])

        logging.info("SQL: %s" % sql_real)

        sql_r = db.select(sql)

        data_num = len(sql_r)
        header = []
        hey_list = []
        results = [[]]
        if data_num == 0:
            pass
        elif len(sql_r[0]) == 0:
            pass
        else:
            results = []
            for k in sql_r[0]:
                col_id = k[k.index("_") + 1:]
                col_id = col_id.replace(")", "")
                col_id = int(col_id) - 1
                header.append(table['header'][col_id])
                hey_list.append(k)
            for data in sql_r:
                r = []
                for h in hey_list:
                    try:
                        v = str(data[h])
                    except:
                        v = data[h]
                    r.append(v)
                if len(r) > 0:
                    results.append(r)

        return render_template('index.html', sql=sql_real, question=q, db_id=table_id, header=header, result=results)
    return render_template("index.html")
