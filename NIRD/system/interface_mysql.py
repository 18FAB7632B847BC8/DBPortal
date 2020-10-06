#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    _interface_mysql.db.py
    ~~~~~~~~~~

    this module implements the low level interface of mysql, including multi-threading
    connection access, sql-insertion preventing, transaction and connection maintain
    and release, simple sql operation, etc.

    :copyright (c) 2015 by Han Fu
    :license: BSD, see LICENSE for more details.
"""

import threading
import functools
import logging


class Dict(dict):
    """Dict object inherit dict class implements the operation '.'
    exp. >>> d = Dict(a = 1) >>> d.a
         1
    """

    def __init__(self, names=(), values=(), **kw):
        super(Dict, self).__init__(**kw)
        for k, v in zip(names, values):
            self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value


class DBError(StandardError):
    pass


class _Engine(object):
    """ _Engine object is a engine to control the connection of MySQL
    """

    def __init__(self, connect, config):
        self.config = config
        self._connect = connect

    def connect(self):
        return self._connect()

    def close(self):
        self.connect().close()


engine = None


class _LasyConnection(object):
    """ _LasyConnction object is to ensure that only get the connection when a request
    occurs
    """

    def __init__(self):
        self.connection = None

    def cursor(self):
        global engine
        if self.connection is None:
            self.connection = engine.connect()
            logging.info('connection open:%s' % id(self.connection))
        return self.connection.cursor()

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def cleanup(self):
        if self.connection:
            logging.info('connection closed:%s' % id(self.connection))
            self.connection.close()
            self.connection = None


class _DbContext(threading.local):
    """ context operation between multiple threads
    """

    def __init__(self):
        super(_DbContext, self).__init__()
        self.connection = None
        self.transactions = 0

    def is_init(self):
        return self.connection is not None

    def init(self):
        self.connection = _LasyConnection()
        self.transactions = 0

    def cleanup(self):
        self.connection.cleanup()
        self.transactions = 0

    def cursor(self):
        return self.connection.cursor()


_db_ctx = _DbContext()


def create_engine(user, password, database, host='127.0.0.1', port=3306):
    """ create database connection through network but not unix-cookie

    :param user: user_name
    :param password: password
    :param database: datebase name
    :param host: host ip
    :param port: port number type: int
    :return:
    """
    import pymysql
    global engine
    if engine is not None:
        raise DBError('Database is already initialized')
    params = dict(user=user, password=password, database=database, host=host, port=port)
    if database is None:
        logging.warning('Database not clear')
    # default parameters
    params['use_unicode'] = True
    params['charset'] = 'utf8'
    params['autocommit'] = False
    engine = _Engine(lambda: pymysql.connect(**params), params)
    logging.info('mysql connection is initialized')


def change_database(database):
    import pymysql
    global engine
    assert engine is not None
    config = engine.config
    config['database'] = database
    engine.close()
    engine = _Engine(lambda: pymysql.connect(**config), config)
    logging.info('mysql connection is updated')



class _ConnectionContext(object):
    """ _ConnectionContext object gains and release connection context automatically

    a with clause added
    exp. with connection():
             mysql_operations()
    """

    def __enter__(self):
        global _db_ctx
        self.should_cleanup = False
        if not _db_ctx.is_init():
            _db_ctx.init()
            self.should_cleanup = True
        return self

    def __exit__(self, exctype, excvalue, traceback):
        global _db_ctx
        if self.should_cleanup:
            _db_ctx.cleanup()


def connection():
    return _ConnectionContext()


def with_connection(func):
    """ decorator to add a connection context
    exp.@with_connection
        def select(sql, *args):
           pass
    :param func: function to execute within a connection
    """

    @functools.wraps(func)
    def _wrapper(*args, **kw):
        with _ConnectionContext():
            return func(*args, **kw)

    return _wrapper


class _TransactionContext(object):
    """ _TransactionContext object commit and rollback operation automatically

    a with clause added
    exp. with transaction():
             mysql_operations()

    transactions may be nested. variable transactions is to compute the level inner
    a multi-level transaction
    """

    def __enter__(self):
        global _db_ctx
        self.should_clean_up = False
        if not _db_ctx.is_init():
            _db_ctx.init()
            self.should_clean_up = True
        _db_ctx.transactions += 1
        return self

    def __exit__(self, exctype, excvalue, traceback):
        global _db_ctx
        _db_ctx.transactions -= 1
        try:
            if _db_ctx.transactions == 0:
                if exctype is None:
                    self.commit()
                else:
                    self.rollback()
        finally:
            if self.should_clean_up:
                _db_ctx.cleanup()

    @staticmethod
    def commit():
        global _db_ctx
        try:
            _db_ctx.connection.commit()
            logging.info('transaction commit() total:%d' % _db_ctx.transcations)
        except StandardError:
            _db_ctx.connection.rollback()

    @staticmethod
    def rollback():
        _db_ctx.rollback()
        logging.info('transaction rollback() total:%d' % _db_ctx.transcations)


def transaction():
    return _TransactionContext()


def with_transaction(func):
    """ decorator to add a connection context
    exp.@with_connection
        def select(sql, *args):
           pass
    :param func: function to execute within a transaction
    """

    @functools.wraps(func)
    def _wrapper(*args, **kw):
        with _TransactionContext():
            return func(*args, **kw)

    return _wrapper


def _select(sql, onlyone, *args):
    """ private function to implement select operation

    :param sql: sql string
    :param onlyone: return only one tuple or not
    :param args: values
    :return:
    """
    global _db_ctx
    cursor = None
    sql = sql.replace('?', '%s')
    try:
        cursor = _db_ctx.connection.cursor()
        cursor.execute(sql, args)
        if cursor.description:
            names = [x[0] for x in cursor.description]
        else:
            names = []
        if onlyone:
            values = cursor.fetchone()
            if not values:
                return None
            return Dict(names, values)
        return [Dict(names, x) for x in cursor.fetchall()]
    finally:
        if cursor:
            cursor.close()


def _update(sql, *args):
    """ private function to implement select operation

    :param sql: sql string
    :param args: values
    :return:
    """
    global _db_ctx
    cursor = None
    sql = sql.replace('?', '%s')
    try:
        cursor = _db_ctx.connection.cursor()

        cursor.execute(sql, args)
        rc = cursor.rowcount

        if _db_ctx.transactions == 0:
            _db_ctx.connection.commit()
        return rc
    finally:
        if cursor:
            cursor.close()


@with_connection
def select_one(sql, *args):
    return _select(sql, True, *args)


@with_connection
def select(sql, *args):
    return _select(sql, False, *args)


@with_connection
def count(sql, *args):
    d = _select(sql, True, *args)
    return d.values()[0]


@with_connection
def insert(table_name, **kw):
    [keys, values] = zip(*kw.iteritems())
    sql = 'insert into %s (%s) values (%s)' % (
    table_name, ','.join([' %s ' % key for key in keys]), ','.join(['?' for i in range(len(keys))]))
    return _update(sql, *values)


@with_connection
def update(sql, *args):
    return _update(sql, *args)


