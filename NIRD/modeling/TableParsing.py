#!/usr/bin/env python

from json import load, dumps


def parse_table():
    file_path = "../data/spider/tables.json"

    data = load(open(file_path))

    database = {}
    for table in data:
        table_dict = {
            'db_id': table['db_id'],
            'table_list': [],
            'column_list': [],
            'type_list': [],
        }
        column_names = table['column_names_original']
        assert column_names[0][0] == -1 and column_names[0][1] == "*"
        table_dict['type_list'] = table['column_types']

        for t_name in table['table_names_original']:
            table_dict['table_list'].append({"table_name": t_name.lower(), "columns": []})
        for c_name in column_names:
            table_dict['column_list'].append({"table_id": c_name[0], "column_name": c_name[1].lower()})
            table_dict['table_list'][c_name[0]]['columns'].append(c_name[1])
        assert len(table_dict['column_list']) == len(table_dict['column_list'])
        join_dict = dict()
        for f_k in table['foreign_keys']:
            l_t = table_dict['column_list'][f_k[0]]['table_id']
            r_t = table_dict['column_list'][f_k[1]]['table_id']
            if (l_t, r_t) not in join_dict:
                join_dict[(l_t, r_t)] = [(f_k[0], f_k[1])]
                join_dict[(r_t, l_t)] = [(f_k[1], f_k[0])]
            else:
                join_dict[(l_t, r_t)].append((f_k[0], f_k[1]))
                join_dict[(r_t, l_t)].append((f_k[1], f_k[0]))
        table_dict['join_dict'] = join_dict
        database[table['db_id']] = table_dict

    return database


def dump():
    database = parse_table()
    partition = "dev"
    f_dbid = open("../data/%s.db_id" % partition)
    fw_h = open("../data/%s.header" % partition, 'w')
    fw_t = open("../data/%s.table" % partition, 'w')
    fw_h2t = open("../data/%s.h2t" % partition, 'w')

    for line in f_dbid:
        db_id = line.strip()
        table_dict = database[db_id]
        table_list = ['[EMPTY]']
        column_list = ['[EMPTY]', '[ALL]']
        h2t_list = [0, 0]
        for table in table_dict['table_list']:
            t_name = table['table_name']
            table_list.append(t_name)

        for column in table_dict['column_list'][1:]:
            c_name = column['column_name']
            t_id = column['table_id']
            h2t_list.append(t_id + 1)
            column_list.append(c_name)

        fw_h.write("%s\n" % ' || '.join(column_list))
        fw_t.write("%s\n" % ' || '.join(table_list))
        fw_h2t.write("%s\n" % ' '.join(map(str, h2t_list)))


def dump_column_type():
    database = parse_table()
    partition = "dev"
    f_dbid = open("../data/%s.db_id" % partition)
    f_ht = open("../data/%s.column_types" % partition, 'w')
    f_tf = open("../data/%s.table_foreign" % partition, 'w')
    max_t = 0
    for line in f_dbid:
        db_id = line.strip()
        table_dict = database[db_id]
        h_types = ['EMPTYCOLUMN', 'ALL'] + table_dict['type_list'][1:]
        join_tables = []
        for i in range(len(table_dict['table_list']) + 1):
            join_tables.append(list())
        for pair in table_dict['join_dict']:
            left = pair[0] + 1
            right = pair[1] + 1
            join_tables[left].append(right)
        f_ht.write("%s\n" % ' '.join(h_types))
        f_tf.write("%s\n" % dumps(join_tables))
    print max_t
