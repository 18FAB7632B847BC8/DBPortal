#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    redis_kafkastream.py
    ~~~~~~~~~~

    this module implements the function to insert ETLed article into redis
    :copyright (c) 2015 by Han Fu
    :license: BSD, see LICENSE for more details.
"""

from gevent import monkey; monkey.patch_all()
from redis import ConnectionError, StrictRedis, ConnectionPool
from redis.sentinel import Sentinel
from redis import WatchError

import logging


class RedisInterface(object):
    """
    A coroutine is assigned for one partition and do the job independently.
    """

    def __init__(self, config_redis, multi_server=False, num_partition=1):
        self.multi_server = multi_server
        self.connections = []
        if multi_server:
            self.sentinel = Sentinel(config_redis['sentinel'])
            # assign a connection for one coroutine
            for i in range(config_redis.get("num_partition", num_partition)):
                self.connections.append(self.sentinel.master_for(config_redis['sentinel_master_name']))
        else:
            self.pool = ConnectionPool(**config_redis['connection_pool'])
            self.connections.append(StrictRedis(connection_pool=self.pool, decode_responses=True))

    def conn(self, partition=0):
        rds = self.connections[partition]
        return rds

    @staticmethod
    def transaction(rds, func, params, *watches, **kwargs):
        """ a transaction to ensure the function execution successful. First, it will
        put a monitor to every key in redis related to write operation. If there is an
        error during execution, then keep on trying until the execution is successful.

        :param rds: redis object
        :param func: the function to be executed as a transaction
        :param params: a list or tuple of the parameters of the function
        :param watches: keys should be watched during a transaction
        :param kwargs: other parameters for pipeline
        :return: None
        """
        shard_hint = kwargs.pop('shard_hint', None)
        with rds.pipeline(True, shard_hint) as pipe:
            while True:
                try:
                    if watches:
                        pipe.watch(*watches)
                    func(pipe, *params)
                    return pipe.execute()
                except WatchError, we:
                    logging.warning('function executed failed')
                    logging.warning('WatchError: %s' % we)
                    continue
                except ConnectionError, ce:
                    logging.warning('redis connection error')
                    logging.warning('ConnectionError: %s' % ce)


if __name__ == '__main__':
    from json import loads, dumps
    rds_itf = RedisInterface()
    conn = rds_itf.conn()

    def hset(pipe, name, key, value):
        pipe.multi()
        pipe.hset(name, key, value)

    with open("../data/Chinese/sampled_data") as f:
        for idx, line in enumerate(f):
            table = loads(line.strip())
            exe_rt = rds_itf.transaction(conn, hset, ["tables", "table_%d" % idx, dumps(table)])

