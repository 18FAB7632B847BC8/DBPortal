#!/usr/bin/env python
# coding: utf-8

import theano.tensor as tensor
import theano
import numpy
from collections import OrderedDict


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def get_plat_sql(sql_block, q_toks, table_dict, nv_idx, nf_idx, nu_idx, ni_idx, ne_idx, stack):
    s_type2 = sql_block['s_type2']
    s_distinct = sql_block['s_distinct']
    s_agg = sql_block['s_agg']
    s_op = sql_block['s_op']
    s_column = sql_block['s_column']
    s_value_left_end = sql_block['s_value_left_end']
    s_value_left_start = sql_block['s_value_left_start']
    s_value_right_start = sql_block['s_value_right_start']
    s_value_right_end = sql_block['s_value_right_end']
    s_nested_value = sql_block['s_nested_value']
    nested_from = sql_block['nested_from']
    limit = sql_block['limit']
    distinct = sql_block['distinct']
    order = sql_block['order']
    combine = sql_block['combine']
    connector = sql_block['connector']
    from_table = sql_block['from_table']

    combine_clause = ""
    # COMBINE
    if combine == 0:
        pass
    elif combine == 1:
        ni_idx += 1
        combine_clause = " INTERSECT NESTEDINTERSECT%d" % ni_idx
        stack.append("NESTEDINTERSECT%d" % ni_idx)
    elif combine == 2:
        nu_idx += 1
        combine_clause = " UNION NESTEDUNION%d " % nu_idx
        stack.append("NESTEDUNION%d" % nu_idx)
    elif combine == 3:
        ne_idx += 1
        combine_clause = " EXCEPT NESTEDEXCEPT%d" % ne_idx
        stack.append("NESTEDEXCEPT%d" % ne_idx)
    else:
        raise

    # SELECT
    if distinct == 1:
        select_clause = "SELECT DISTINCT"
    else:
        select_clause = "SELECT"
    select_columns = []
    start = 0
    for i in range(start, len(s_type2)):
        t2 = s_type2[i]
        if t2 > 17:
            start = i
            break
        column = s_column[i]
        if column == 0:
            continue
        column_name = table_dict['column_list'][column - 1]['column_name']
        if column > 1:
            table_name = table_dict['table_list'][table_dict['column_list'][column - 1]['table_id']]['table_name']
            column_name = "%s.%s" % (table_name, column_name)
        dst_in = s_distinct[i]
        if dst_in == 1:
            column_name = "DISTINCT %s" % column_name
        assert column > 0
        agg = s_agg[i]
        if agg > 0:
            column_name = "%s(%s)" % (AGG_OPS[agg], column_name)
        select_columns.append(column_name)
    select_clause = "%s %s" % (select_clause, " , ".join(select_columns))

    where_columns = []
    having_columns = []

    if connector == 1:
        connector = " OR "
    else:
        connector = " AND "
    nested_values = []
    for i in range(start, len(s_type2)):
        t2 = s_type2[i]
        if t2 > 18:
            start = i
            break
        column = s_column[i]
        if column == 0:
            continue
        column_name = table_dict['column_list'][column - 1]['column_name']
        if column > 1:
            table_name = table_dict['table_list'][table_dict['column_list'][column - 1]['table_id']]['table_name']
            column_name = "%s.%s" % (table_name, column_name)
        dst_in = s_distinct[i]
        if dst_in == 1:
            column_name = "DISTINCT %s" % column_name
        assert column > 0
        agg = s_agg[i]
        if agg > 0:
            column_name = "%s(%s)" % (AGG_OPS[agg], column_name)

        op = s_op[i]
        if op == 0:
            continue
        assert op > 0
        column_name = "%s %s" % (column_name, WHERE_OPS[op])
        nested_value = s_nested_value[i]
        if nested_value:
            nv_idx += 1
            column_name = "%s ( NESTEDVALUE%d )" % (column_name, nv_idx)
            nested_values.append("NESTEDVALUE%d" % nv_idx)
        else:
            vl_s = s_value_left_start[i]
            vl_e = s_value_left_end[i]
            value_l = "".join(q_toks[vl_s:vl_e + 1])
            value_l = value_l.encode('utf-8')
            value_l = value_l.replace("Ġ", " ").strip()
            if op == 9:
                value_l = "%" + value_l + "%"
            if not value_l.isdigit():
                value_l = value_l.replace('"', "").strip()
                value_l = '"%s"' % value_l
            column_name = "%s %s" % (column_name, value_l)
            if op == 1:
                vr_s = s_value_right_start[i]
                vr_e = s_value_right_end[i]
                value_r = "".join(q_toks[vr_s:vr_e + 1])
                value_r = value_r.encode('utf-8')
                value_r = value_r.replace("Ġ", " ")
                column_name = "%s AND %s" % (column_name, value_r)
        if agg:
            having_columns.append(column_name)
        else:
            where_columns.append(column_name)
    nested_values.reverse()
    for nv in nested_values:
        stack.append(nv)

    if len(where_columns) > 0:
        where_clause = " WHERE %s" % connector.join(where_columns)
    else:
        where_clause = ""

    if len(having_columns) > 0:
        having_clause = " HAVING %s" % connector.join(having_columns)
    else:
        having_clause = ""

    # ORDER BY

    if order == 0:
        order = "ASC"
    else:
        order = "DESC"

    orderby_columns = []
    for i in range(start, len(s_type2)):
        t2 = s_type2[i]
        if t2 > 19:
            start = i
            break
        column = s_column[i]
        if column == 0:
            continue
        column_name = table_dict['column_list'][column - 1]['column_name']
        if column > 1:
            table_name = table_dict['table_list'][table_dict['column_list'][column - 1]['table_id']]['table_name']
            column_name = "%s.%s" % (table_name, column_name)
        assert column > 0
        agg = s_agg[i]
        if agg > 0:
            column_name = "%s(%s)" % (AGG_OPS[agg], column_name)
        orderby_columns.append(column_name)
    if len(orderby_columns) > 0:
        orderby_clause = " ORDER BY %s %s" % (" , ".join(orderby_columns), order)
    else:
        orderby_clause = ""

    # GROUP BY
    groupby_columns = []
    for i in range(start, len(s_type2)):
        column = s_column[i]
        if column == 0:
            continue
        column_name = table_dict['column_list'][column - 1]['column_name']
        if column > 1:
            table_name = table_dict['table_list'][table_dict['column_list'][column - 1]['table_id']]['table_name']
            column_name = "%s.%s" % (table_name, column_name)
        groupby_columns.append(column_name)
    if len(groupby_columns) > 0:
        groupby_clause = " GROUP BY %s" % " , ".join(groupby_columns)
    else:
        groupby_clause = ""

    # FROM
    from_tables = []

    if nested_from == 1 or from_table[0] == 0:
        nf_idx += 1
        from_clause = " FROM ( NESTEDFROM%d )" % nf_idx
        stack.append("NESTEDFROM%d" % nf_idx)
    else:
        last_idx = -1
        table_idxes = []
        from_table_dict = dict()
        for table_idx in from_table:
            if table_idx < 1:
                break
            if table_idx in from_table_dict and from_table_dict[table_idx] == 2:
                continue
            elif table_idx in from_table_dict:
                from_table_dict[table_idx] += 1
            else:
                from_table_dict[table_idx] = 1
            table_idx = table_idx - 1
            if table_idxes and (table_idxes[-1], table_idx) not in table_dict['join_dict']:
                flag = False
                for pre_tidx in table_idxes[:-1]:
                    if (pre_tidx, table_idx) in table_dict['join_dict']:
                        flag = True
                if not flag:
                    for tidx in range(len(table_dict['table_list'])):
                        if (table_idxes[-1], tidx) in table_dict['join_dict'] and (tidx, table_idx) in table_dict['join_dict']:
                            table_idxes.append(tidx)
            table_idxes.append(table_idx)

        for tcount, table_idx in enumerate(table_idxes):
            table_name = table_dict['table_list'][table_idx]['table_name']
            if last_idx >= 0:
                if (last_idx, table_idx) not in table_dict['join_dict']:
                    for tpid in table_idxes[:tcount]:
                        if (tpid, table_idx) in table_dict['join_dict']:
                            last_idx = tpid
                            break
                if (last_idx, table_idx) in table_dict['join_dict']:
                    join_part = []
                    join_column_pairs = table_dict['join_dict'][(last_idx, table_idx)]
                    for pair in join_column_pairs:
                        column_name = table_dict['column_list'][pair[0]]['column_name']
                        tn = table_dict['table_list'][table_dict['column_list'][pair[0]]['table_id']]['table_name']
                        cln = "%s.%s" % (tn, column_name)
                        column_name = table_dict['column_list'][pair[1]]['column_name']
                        tn = table_dict['table_list'][table_dict['column_list'][pair[1]]['table_id']]['table_name']
                        crn = "%s.%s" % (tn, column_name)
                        join_part.append("%s = %s" % (cln, crn))
                    if len(join_part) > 0:
                        table_name = "%s ON %s" % (table_name, " AND ".join(join_part))

            last_idx = table_idx

            from_tables.append(table_name)
        from_clause = " FROM %s" % " JOIN ".join(from_tables)

    limit_clause = ""
    if limit > 0:
        limit_clause = " LIMIT %d" % limit

    sql = select_clause + from_clause + where_clause + groupby_clause + having_clause + orderby_clause + limit_clause + combine_clause

    return sql, stack, nv_idx, nf_idx, nu_idx, ni_idx, ne_idx


def inspect_inputs(i, node, fn):
    print('>> Inputs: ', i, node, [input[0] for input in fn.inputs])


def inspect_outputs(i, node, fn):
    print('>> Outputs: ', i, node, [input[0] for input in fn.outputs])


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout, scale=0.01, ortho=True):
    if scale == "xavier":
        # Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks."
        # International conference on artificial intelligence and statistics. 2010.
        # http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf
        scale = 1. / numpy.sqrt(nin)
    elif scale == "he":
        # Claimed necessary for ReLU
        # Kaiming He et al. (2015)
        # Delving deep into rectifiers: Surpassing human-level performance on
        # imagenet classification. arXiv preprint arXiv:1502.01852.
        scale = 1. / numpy.sqrt(nin/2.)
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def norm_weight_tensor(shape, sample='uniform', scale=0.01, std=None, mean=0.):
    if sample == 'uniform':
        if std is None:
            assert mean == 0.
            input_shape = numpy.prod(shape[1:])
            out_shape = numpy.prod(shape[0] * numpy.prod(shape[2:]))
            std = numpy.sqrt(2. / (input_shape + out_shape))

        a = mean - numpy.sqrt(3) * std
        b = mean + numpy.sqrt(3) * std

        W = numpy.random.uniform(low=a, high=b, size=shape)
    elif sample == 'normal':
        assert std is not None
        W = numpy.random.uniform(mean, std, size=shape)
    else:
        raise ValueError('sample must in (normal, uniform)')

    return W.astype('float32')





def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def unzip(tparams):
    t_p = OrderedDict()
    for k, v in tparams.iteritems():
        t_p[k] = v.get_value()

    return t_p
