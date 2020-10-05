#!/usr/bin/env python
# coding: utf-8

from json import load
from TableParsing import parse_table

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


where_op_dict = {}
for i, wop in enumerate(WHERE_OPS):
    where_op_dict[i] = wop


unit_op_dict = {}
for i, uop in enumerate(UNIT_OPS):
    unit_op_dict[i] = uop


agg_op_dict = {}
for i, aop in enumerate(AGG_OPS):
    agg_op_dict[i] = aop


f_home = "../data"
partition = "dev"
# SELECT
f_sd = open("%s/%s.s_d" % (f_home, partition), 'w')
f_scl = open("%s/%s.s_c_l" % (f_home, partition), 'w')
f_scr = open("%s/%s.s_c_r" % (f_home, partition), 'w')
f_sa = open("%s/%s.s_a" % (f_home, partition), 'w')
f_sal = open("%s/%s.s_a_l" % (f_home, partition), 'w')
f_sar = open("%s/%s.s_a_r" % (f_home, partition), 'w')
f_so = open("%s/%s.s_o" % (f_home, partition), 'w')
f_sdl = open("%s/%s.s_d_l" % (f_home, partition), 'w')
f_sdr = open("%s/%s.s_d_r" % (f_home, partition), 'w')
# FROM
f_ft = open("%s/%s.f_t" % (f_home, partition), 'w')
f_fn = open("%s/%s.f_n" % (f_home, partition), 'w')
# WHERE
f_wc = open("%s/%s.w_c" % (f_home, partition), 'w')
f_wcn = open("%s/%s.w_cn" % (f_home, partition), 'w')
f_wco = open("%s/%s.w_c_o" % (f_home, partition), 'w')
f_wn = open("%s/%s.w_n" % (f_home, partition), 'w')
f_wvl = open("%s/%s.value_l" % (f_home, partition), 'w')
f_wvr = open("%s/%s.value_r" % (f_home, partition), 'w')
f_wvn = open("%s/%s.w_v_n" % (f_home, partition), 'w')
f_wa = open("%s/%s.w_a" % (f_home, partition), 'w')
f_wd = open("%s/%s.w_d" % (f_home, partition), 'w')
# ORDER_BY
f_ocl = open("%s/%s.o_c_l" % (f_home, partition), 'w')
f_ocr = open("%s/%s.o_c_r" % (f_home, partition), 'w')
f_oop = open("%s/%s.o_op" % (f_home, partition), 'w')
f_oo = open("%s/%s.o_o" % (f_home, partition), 'w')
f_oal = open("%s/%s.o_a_l" % (f_home, partition), 'w')
f_oar = open("%s/%s.o_a_r" % (f_home, partition), 'w')
# GROUP_BY
f_gc = open("%s/%s.g_c" % (f_home, partition), 'w')
# LIMIT
f_l = open("%s/%s.l" % (f_home, partition), 'w')
# COMBINE
f_c = open("%s/%s.c" % (f_home, partition), 'w')
# OTHER
f_q = open("%s/%s.question" % (f_home, partition), 'w')
f_s = open("%s/%s.sql" % (f_home, partition), 'w')
f_ps = open("%s/%s.plat_sql " % (f_home, partition), 'w')
f_idx = open("%s/%s.idx" % (f_home, partition), 'w')
f_dbid = open("%s/%s.db_id" % (f_home, partition), 'w')

def parse_select(sql):
    select_clause = sql['select']

    agg_outer = []
    agg_inner_left = []
    agg_inner_right = []
    column_inner_left = []
    column_inner_right = []
    distinct_inner_left = []
    distinct_inner_right = []
    op = []
    for select_unit in select_clause[1]:
        agg_outer.append(select_unit[0])
        val_unit = select_unit[1]
        op.append(val_unit[0])
        col_unit_left = val_unit[1]
        col_unit_right = val_unit[2]
        agg_inner_left.append(col_unit_left[0])
        if col_unit_left[0]:
            print sql
        column_inner_left.append(col_unit_left[1])
        distinct_inner_left.append(col_unit_left[2])

        if col_unit_right:
            agg_inner_right.append(col_unit_right[0])
            column_inner_right.append(col_unit_right[1])
            distinct_inner_right.append(col_unit_right[2])
        else:
            agg_inner_right.append(None)
            column_inner_right.append(None)
            distinct_inner_right.append(None)

    select_dict = {
        'distinct': int(select_clause[0]),
        'agg_outer': agg_outer,
        'agg_inner_left': agg_inner_left,
        'agg_inner_right': agg_inner_right,
        'column_inner_left': column_inner_left,
        'column_inner_right': column_inner_right,
        'distinct_inner_left': distinct_inner_left,
        'distinct_inner_right': distinct_inner_right,
        'op': op
    }

    return select_dict


def parse_from(sql):
    from_clause = sql['from']
    cond_col_left = []
    cond_col_right = []
    tables = []

    if from_clause['table_units'][0][0] == 'sql':
        return True, from_clause['table_units'][0][1]

    for cond in from_clause['conds']:
        if type(cond) is unicode:
            continue
        else:
            val_unit = cond[2]
            col_unit1 = val_unit[1]
            col_unit2 = cond[3]
            cond_col_left.append(col_unit1[1])
            cond_col_right.append(col_unit2[1])

    for table_unit in from_clause['table_units']:
        if table_unit[0] == u'table_unit':
            tables.append(table_unit[1])

    from_dict = {
        'cond_col_left': cond_col_left,
        'cond_col_right': cond_col_right,
        'tables': tables
    }

    return False, from_dict


def parse_group_by(sql):
    columns = []
    for ele in sql['groupBy']:
        columns.append(ele[1])

    groupBy_dict = {
        'columns': columns
    }

    return groupBy_dict


def parse_order_by(sql):
    orders = {
        "asc": 0,
        "desc": 1
    }
    order_by_clause = sql['orderBy']
    if not order_by_clause:
        return {
            'order': 0,
            'agg_inner_left': [],
            'agg_inner_right': [],
            'column_inner_left': [],
            'column_inner_right': [],
            'op': []
        }
    order = orders[str(order_by_clause[0])]
    agg_inner_left = []
    agg_inner_right = []
    column_inner_left = []
    column_inner_right = []
    op = []
    for val_unit in order_by_clause[1]:
        op.append(val_unit[0])
        col_unit_left = val_unit[1]
        col_unit_right = val_unit[2]
        agg_inner_left.append(col_unit_left[0])
        column_inner_left.append(col_unit_left[1])

        if col_unit_right:
            agg_inner_right.append(col_unit_right[0])
            column_inner_right.append(col_unit_right[1])
        else:
            agg_inner_right.append(None)
            column_inner_right.append(None)

    orderBy_dict = {
        'order': order,
        'agg_inner_left': agg_inner_left,
        'agg_inner_right': agg_inner_right,
        'column_inner_left': column_inner_left,
        'column_inner_right': column_inner_right,
        'op': op
    }

    return orderBy_dict


def parse_filter(sql, v_count):
    filter_clause = sql['where'] + sql['having']

    conn_set = set()
    not_list = []
    cond_op = []
    agg = []
    column = []
    distinct = []
    value_left = []
    value_right = []
    nest_count = 0
    nested_values = []
    for cond in filter_clause:
        if type(cond) is unicode:
            conn_set.add(str(cond))
        else:
            not_op, op_id, val_unit, val1, val2 = cond
            assert op_id > 0
            not_list.append(int(not_op))
            cond_op.append(op_id)
            col_unit = val_unit[1]
            agg.append(col_unit[0])
            column.append(col_unit[1])
            distinct.append(col_unit[2])
            if type(val1) is not dict:
                value_left.append(val1)
            else:
                nest_count += 1
                value_left.append("NESTEDVALUE%d" % (nest_count + v_count))
                nested_values.append(val1)
            assert type(val2) is not dict
            value_right.append(val2)
    if not conn_set:
        conn_set.add("and")
    conn = conn_set.pop()
    if conn_set:
        conn = "and"
    filter_dict = {
        "conn": conn,
        "not_list": not_list,
        "cond_op": cond_op,
        "agg": agg,
        "column": column,
        "distinct": distinct,
        "value_left": value_left,
        "value_right": value_right,
    }

    nested_values.reverse()
    return filter_dict, nested_values


def plat_sql(sql_block, last_sql, idx, table_dict, question):

    # FROM
    if type(sql_block['from']) is str:
        from_clause = "FROM %s" % sql_block['from']
    else:
        from_table_list = sql_block['from']['tables']
        from_column_left = sql_block['from']['cond_col_left']
        from_column_right = sql_block['from']['cond_col_right']
        from_clause = "FROM " + " join ".join(map(lambda t_idx: table_dict['table_list'][t_idx]['table_name'], from_table_list))
        last_tidx = from_table_list[0]
        from_clause = "FROM %s" % table_dict['table_list'][last_tidx]['table_name']
        
        for tidx in from_table_list[1:]:
            t_name = table_dict['table_list'][tidx]['table_name']
            from_clause += " JOIN %s " % (t_name)
            
            join_pair = (last_tidx, tidx)
            try:
                join_list = table_dict['join_dict'][join_pair]
            except:
                print question
                print table_dict['db_id']
                print join_pair
                print table_dict['join_dict']
                print "************"
                continue
            from_clause += " and ".join("%s = %s" % (
                table_dict['column_list'][cols[0]]['column_name'],
                table_dict['column_list'][cols[1]]['column_name']) for cols in join_list
                                        )
            last_tidx = tidx
            

    # SELECT
    select_distinct = sql_block['select']['distinct']
    select_agg_outer = sql_block['select']['agg_outer']
    select_agg_inner_left = sql_block['select']['agg_inner_left']
    select_agg_inner_right = sql_block['select']['agg_inner_right']
    select_column_left = sql_block['select']['column_inner_left']
    select_column_right = sql_block['select']['column_inner_right']
    select_distinct_inner_left = sql_block['select']['distinct_inner_left']
    select_distinct_inner_right = sql_block['select']['distinct_inner_right']
    select_op = sql_block['select']['op']

    select_cols = []

    for index, col1 in enumerate(select_column_left):
        col = table_dict['column_list'][col1]['column_name']
        if select_distinct_inner_left[index]:
            col = "distinct " + col
        if select_agg_inner_left[index]:
            col = agg_op_dict[select_agg_inner_left[index]] + "(%s)" % col
        if select_column_right[index]:
            col2 = table_dict['column_list'][select_column_right[index]]['column_name']
            if select_distinct_inner_right[index]:
                col2 = "distinct " + col2
            if select_agg_inner_right[index]:
                col2 = agg_op_dict[select_agg_inner_right[index]] + "(%s)" % col2
            col = "%s %s %s" % (col1, unit_op_dict[select_op[index]], col2)
        if select_agg_outer[index]:
            col = agg_op_dict[select_agg_outer[index]] + "(%s)" % col
        select_cols.append(col)

    select_clause = ", ".join(select_cols)
    if select_distinct:
        select_clause = "SELECT distinct %s" % select_clause
    else:
        select_clause = "SELECT %s" % select_clause

    # FILTER
    where_conn = sql_block['filter']['conn']
    where_not = sql_block['filter']['not_list']
    where_cond_op = sql_block['filter']['cond_op']
    where_agg = sql_block['filter']['agg']
    where_column = sql_block['filter']['column']
    where_distinct = sql_block['filter']['distinct']
    where_value = sql_block['filter']['value_left']

    where_cols = []

    for index, col1 in enumerate(where_column):
        col = table_dict['column_list'][col1]['column_name']
        if where_distinct[index]:
            col = "distinct " + col
        if where_agg[index]:
            col = agg_op_dict[where_agg[index]] + "(%s)" % col
        if where_not[index]:
            col = col + " not"
        col += " %s" % where_op_dict[where_cond_op[index]]
        # if type(where_value[index]) is not float and "VALUE" in where_value[index]:
        #     col += "%s %s" % (col, where_value[index])
        col += " %s" % (where_value[index])
        where_cols.append(col)
    if where_cols:
        where_clause = "WHERE " + (" %s " % where_conn).join(where_cols)
    else:
        where_clause = ""

    # GROUP_BY
    group_by_col = map(lambda col_idx: table_dict['column_list'][col_idx]['column_name'],
                       sql_block['group_by']['columns'])
    if group_by_col:
        group_by_clause = "GROUP BY %s" % " , ".join(group_by_col)
    else:
        group_by_clause = ""

    # ORDER_BY
    order = 'desc' if sql_block['order_by']['order'] else 'asc'
    order_by_col_left = sql_block['order_by']['column_inner_left']
    order_by_col_right = sql_block['order_by']['column_inner_right']
    order_by_agg_left = sql_block['order_by']['agg_inner_left']
    order_by_agg_right = sql_block['order_by']['agg_inner_right']
    order_by_op = sql_block['order_by']['op']

    order_by_col = []
    for index, col1 in enumerate(order_by_col_left):
        col = table_dict['column_list'][col1]['column_name']
        if order_by_agg_left[index]:
            col = agg_op_dict[order_by_agg_left[index]] + "(%s)" % col
            if order_by_agg_right[index]:
                col2 = table_dict['column_list'][order_by_agg_right[index]]['column_name']
                if order_by_col_right[index]:
                    col2 = agg_op_dict[order_by_col_right[index]] + "(%s)" % col2
                col = "%s %s %s" % (col1, unit_op_dict[order_by_op[index]], col2)
        order_by_col.append(col)
    if order_by_col:
        order_by_clause = "ORDER BY %s" % " , ".join(order_by_col)
        order_by_clause += " %s" % order
    else:
        order_by_clause = ""

    # LIMIT
    if sql_block['limit']:
        limit_clause = "LIMIT %d" % sql_block['limit']
    else:
        limit_clause = ""

    combine_clause = ""
    if sql_block['union']:
        combine_clause = "UNION %s" % sql_block['union']
    elif sql_block['intersect']:
        combine_clause = "INTERSECT %s" % sql_block['intersect']
    elif sql_block['except']:
        combine_clause = "EXCEPT %s" % sql_block['except']

    sql = "%s %s" % (select_clause, from_clause)
    if where_clause:
        sql += " %s" % where_clause
    if group_by_clause:
        sql += " %s" % group_by_clause
    if order_by_clause:
        sql += " %s" % order_by_clause
    if limit_clause:
        sql += " %s" % limit_clause
    if combine_clause:
        sql += " %s" % combine_clause

    if not last_sql:
        return sql
    return last_sql.replace(idx, " ( %s ) " % sql).strip()


def preprocess():
    database = parse_table()

    file_path = "D://workspace/spider/%s.json" % partition

    data = load(open(file_path))

    outputs = []
    for sample in data:
        sql = sample['sql']
        question = sample['question']
        db_id = sample['db_id']
        query = sample['query']
        output = {
            "question": question,
            "sql": dict(),
            "db_id": db_id,
            "query": query,
        }
        table_dict = database[db_id]
        stack = [{'sql': sql, 'idx': 'ORI'}]
        sql_plat = ["",]
        sql_list = []

        u_count = 0  # union
        i_count = 0  # intersect
        e_count = 0  # except
        v_count = 0  # nested value
        f_count = 0  # nested from

        while stack:
            stacked_sql = stack.pop()
            sql = stacked_sql['sql']
            idx = stacked_sql['idx']
            union = None
            intersect = None
            excpt = None
            if sql['union']:
                u_count += 1
                stack.append({'sql': sql['union'], 'idx': 'NESTEDUNION%d' % u_count})
                union = 'NESTEDUNION%d' % u_count
            if sql['intersect']:
                i_count += 1
                stack.append({'sql': sql['intersect'], 'idx': 'NESTEDINTERSECT%d' % i_count})
                intersect = 'NESTEDINTERSECT%d' % i_count
            if sql['except']:
                e_count += 1
                stack.append({'sql': sql['except'], 'idx': 'NESTEDEXCEPT%d' % e_count})
                excpt = 'NESTEDEXCEPT%d' % e_count
            select_dict = parse_select(sql)
            filter_dict, nested_values = parse_filter(sql, v_count)
            orderBy_dict = parse_order_by(sql)
            groupBy_dict = parse_group_by(sql)
            nest_f, from_dict = parse_from(sql)

            limit = 0
            if sql['limit']:
                limit = sql['limit']

            while nested_values:
                nested_value = nested_values.pop()
                v_count += 1
                stack.append({'sql': nested_value, 'idx': 'NESTEDVALUE%d' % v_count})

            if nest_f:
                f_count += 1
                stack.append({'sql': from_dict, 'idx': 'NESTEDFROM%d' % f_count})
                from_dict = 'NESTEDFROM%d' % f_count

            sql_block = {
                "db_id": db_id,
                "intersect": intersect,
                "union": union,
                "except": excpt,
                "select": select_dict,
                "from": from_dict,
                "filter": filter_dict,
                "group_by": groupBy_dict,
                "order_by": orderBy_dict,
                "limit": limit
            }

            sql_list.append(sql_block)
            sql_plat.append(plat_sql(sql_block, sql_plat[-1], idx, table_dict, question))
            dump(sql_block, sql_plat[-1], question, query, idx)
        


def dump(sql_block, sql_plat, question, query, idx):
    question = question.replace(u'‘', u"'")
    question = question.replace(u'’', u"'")
    question = question.replace(u'“', u'"')
    question = question.replace(u'”', u'"')
    question = question.replace(u'？', u'?')
    
    try:
        f_q.write("%s\n" % question.lower())
    except:
        print question
    f_s.write("%s\n" % query.encode('utf-8'))
    f_ps.write("%s\n" % sql_plat.encode('utf-8'))
    f_idx.write("%s\n" % idx)
    
    # SELECT
    select_block = sql_block['select']
    
    f_sd.write("%d\n" % select_block['distinct'])
    f_sdl.write("%s\n" % ' '.join(map(lambda x: str(int(x)), select_block['distinct_inner_left'] + [0])))
    dir_list = []
    for dir in select_block['distinct_inner_right']:
        if dir is None:
            dir_list.append(0)
        else:
            dir_list.append(int(dir))
    f_sdr.write("%s\n" % ' '.join(map(str, dir_list + [0])))
    f_sa.write("%s\n" % ' '.join(map(lambda x: str(int(x)), select_block['agg_outer'] + [0])))
    f_sal.write("%s\n" % ' '.join(map(lambda x: str(int(x)), select_block['agg_inner_left'] + [0])))
    ar_list = []
    for ar in select_block['agg_inner_right']:
        if ar is None:
            ar_list.append(0)
        else:
            ar_list.append(int(ar))
    f_sar.write("%s\n" % ' '.join(map(str, ar_list + [0])))
    f_so.write("%s\n" % ' '.join(map(str, select_block['op'] + [0])))
    f_scl.write("%s\n" % ' '.join(map(lambda x: str(x + 1), select_block['column_inner_left']) + ['0']))
    scr_list = []
    for scr in select_block['column_inner_right']:
        if scr is None:
            scr_list.append(0)
        else:
            scr_list.append(scr + 1)
    f_scr.write("%s\n" % ' '.join(map(str, scr_list + [0])))
    # FROM
    from_block = sql_block['from']
    if type(from_block) is str:
        f_ft.write("0\n")
        f_fn.write("1\n")
    else:
        f_ft.write("%s\n" % ' '.join(map(lambda x: str(x + 1), from_block['tables']) + ['0']))
        f_fn.write("0\n")

    # WHERE
    
    where_clause = sql_block['filter']
    
    f_wco.write("%s\n" % (' '.join(map(str, where_clause['cond_op'] + [0]))))
    f_wcn.write("0\n" if where_clause['conn'] == 'and' else "1\n")
    f_wc.write("%s\n" % ' '.join(map(lambda x: str(x + 1), where_clause['column']) + ['0']))
    f_wn.write("%s\n" % ' '.join(map(str, where_clause['not_list'] + [0])))
    f_wd.write("%s\n" % ' '.join(map(lambda x: str(int(x)), where_clause['distinct'] + [0])))
    f_wa.write("%s\n" % ' '.join(map(str, where_clause['agg'] + [0])))
    vl_list = []
    for v_l in where_clause['value_left']:
        if type(v_l) is list:
            vl_list.append("COLUMN_%d" % (v_l[1] + 1))
        else:
            vl_list.append(str(v_l))
    f_wvl.write("%s\n" % ' || '.join(vl_list))
    f_wvr.write("%s\n" % ' || '.join(map(str, where_clause['value_right'])))
    
    
    vn_list = []
    for v_l in where_clause['value_left']:
        if "NESTEDVALUE" in str(v_l):
            vn_list.append("1")
        else:
            vn_list.append("0")
    vn_list.append("0")
    f_wvn.write("%s\n" % ' '.join(vn_list))
    
    # ORDER_BY

    orderby_block = sql_block['order_by']
    f_oo.write("%d\n" % orderby_block['order'])
    f_oop.write("%s\n" % ' '.join(map(str, orderby_block['op'] + [0])))
    f_ocl.write("%s\n" % ' '.join(map(lambda x: str(x + 1), orderby_block['column_inner_left']) + ['0']))
    ocr_list = []
    for ocr in orderby_block['column_inner_right']:
        if ocr is None:
            ocr_list.append(0)
        else:
            ocr_list.append(ocr + 1)
    f_ocr.write("%s\n" % ' '.join(map(str, ocr_list + [0])))
    f_oal.write("%s\n" % ' '.join(map(lambda x: str(int(x)), orderby_block['agg_inner_left'] + [0])))
    oar_list = []
    for oar in orderby_block['agg_inner_right']:
        if oar is None:
            oar_list.append(0)
        else:
            oar_list.append(int(oar))
    f_oar.write("%s\n" % ' '.join(map(str, oar_list + [0])))
    # GROUP_BY
    groupby_block = sql_block['group_by']
    f_gc.write("%s\n" % ' '.join(map(lambda x: str(x + 1), groupby_block['columns']) + ['0']))
    # LIMIT
    f_l.write("%d\n" % sql_block['limit'])
    # COMBINE
    if sql_block['intersect']:
        f_c.write("I\n")
    elif sql_block['union']:
        f_c.write("U\n")
    elif sql_block['except']:
        f_c.write("E\n")
    else:
        f_c.write("N\n")
    
    f_dbid.write("%s\n" % sql_block['db_id'])
    

if __name__ == '__main__':
    preprocess()

