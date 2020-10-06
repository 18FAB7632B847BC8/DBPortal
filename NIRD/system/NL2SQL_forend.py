from gevent import monkey
from flask import Flask
from flask import request, render_template
from kafka import KafkaProducer
from NL2SQL_backend import MRedis
from functools import wraps
from flask import make_response
from gevent import pywsgi
import gevent
from time import time
import requests
from json import dumps, loads
import interface_mysql as db
from config.config_default import config_mysql, config_kafka, config_redis
from redis_interface import RedisInterface
import logging

monkey.patch_all()
MAX_TRY = 100


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    )

app = Flask(__name__)
app.config['SECRET_KEY'] = '*********************'
db.create_engine(**config_mysql['connection'])
kafka_producer = KafkaProducer(**config_kafka)
mredis = MRedis(config_redis)


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

        q = request.form.get("nl-question")
        logging.info("Question:%s" % q)
        timestep = str(time())
        message = {
            "question": q,
            "timestep": timestep
        }
        kafka_producer.send(config_kafka['topic'], dumps(message).encode('utf-8'))
        count = 0
        while True:
            count += 1
            if count > MAX_TRY:
                raise StandardError("Unsolvable query: %s at %s" % (q, timestep))
            try:
                data = mredis.hget("Inter_data", timestep)
                if not data:
                    gevent.sleep(0)
                else:
                    break
            except:
                gevent.sleep(0)
        data = loads(data, 'utf-8')
        sql = data['sql']
        db_id = data['db_id']
        db.change_database(db_id)

        logging.info("SQL: %s" % sql)

        sql_r = db.select(sql)

        data_num = len(sql_r)
        key_list = []
        results = [[]]
        if data_num == 0:
            pass
        elif len(sql_r[0]) == 0:
            pass
        else:
            results = []
            for k in sql_r[0]:
                key_list.append(k)
            for data in sql_r:
                r = []
                for h in key_list:
                    try:
                        v = str(data[h])
                    except:
                        v = data[h]
                    r.append(v)
                if len(r) > 0:
                    results.append(r)

        return render_template('index.html', sql=sql, question=q, db_id=db_id, header=key_list, result=results)
    return render_template("index.html")
