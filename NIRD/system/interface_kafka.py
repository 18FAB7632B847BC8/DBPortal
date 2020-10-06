#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    _interface_kafka.interface.py
    ~~~~~~~~~~

    this module implements the function consume messages from kafka with gevent and
    offer relevant api for operation on if messages

    :copyright (c) 2015 by Han Fu
    :license: BSD, see LICENSE for more details.
"""


from gevent import monkey; monkey.patch_all()

from kafka import KafkaConsumer
from kafka.conn import ConnectionError
from kafka.common import ConsumerTimeout
import gevent
import logging
from time import time


NO_MESSAGE_WAIT_TIME = 2
FUNC_FAIL_WAIT_TIME = 1
KAFKA_UNAVAILABLE_WAIT_TIME = 300
MAX_NUM_MESSAGE = 32
MAX_TIME_INTERVAL = 0.1


def gevent_consume(partition, func, topic, q, configs, *args):
    """ this is an inner function of gevent job. It will keep on consuming messages
    forever and do certain jobs on the messages

    :param partition: the number of partition
    :param func: function to execute with the message
    :param topic: topic name
    :param q: message queue
    :param configs: other KafkaConsumer configs
    :return: None
    """
    last_time = 0

    while True:
        try:
            consumer = KafkaConsumer(topic, **configs)
            consumer.set_topic_partitions((topic, partition))
            logging.info('consumer %d initialized' % partition)
            break
        except ConnectionError as ce:
            logging.error('Kafka initialization Error %s at partition %d' % (ce, partition))
            gevent.sleep(KAFKA_UNAVAILABLE_WAIT_TIME)
            continue

    while True:
        new_time = time()
        if float(new_time - last_time) > MAX_TIME_INTERVAL:
            func(q, *args)
            for i in range(len(q)):
                q.pop()
        try:
            for message in consumer:
                logging.info('get message from partition %d, offeset: %d' % (partition, message.offset))
                q.insert(0, message)
                if len(q) >= MAX_NUM_MESSAGE:
                    func(q, *args)
                    for i in range(len(q)):
                        q.pop()
                    break

            consumer.commit()
        except ConsumerTimeout:
            logging.info('Partition %d : No message to Consume' % partition)
            gevent.sleep(NO_MESSAGE_WAIT_TIME)
        except ConnectionError as ce:
            logging.error('Kafka initialization Error %s at partition %d' % (ce, partition))
            gevent.sleep(KAFKA_UNAVAILABLE_WAIT_TIME)
            continue
        except Exception, e:
            logging.error(e)


class GeventConsumer(object):
    def __init__(self, kafka_configs, func):
        """ initialization

        :param kafka_configs: type: dict
        :param func: function to execute on kafka messages
        :return: None
        """
        self.consumer = None
        self.topics = kafka_configs['topics']
        self.partition_count = kafka_configs['partition_count']
        self.configs = kafka_configs['consumer']
        self.func = func
        self.gevent_list = list()
        self.message_queue = []

    def consume(self, *args):
        """ assign one coroutine for one partition to consume

        :return: None
        """
        for topic in self.topics:
            for i in xrange(self.partition_count):
                self.gevent_list.append(gevent.spawn(gevent_consume, i, self.func, topic, self.message_queue, self.configs, *args))
        # return the control right to greenlet
        gevent.joinall(self.gevent_list)

    def close(self):
        self.consumer = None

