import numpy
import logging


class RetrievalHolder(object):
    def __init__(self, question_path, header_path, table_path, idx_path, from_path, column2table_path,
                 batch_size=128, shuffle_mode='simple', partition="train",
                 cls_token=1, sep_token=2, train_mode="inner"
                 ):
        self.question_path = question_path
        self.header_path = header_path
        self.table_path = table_path
        self.idx_path = idx_path
        self.from_path = from_path
        self.column2table_path = column2table_path
        self.shuffle_mode = shuffle_mode
        self.batch_size = batch_size
        self.partition = partition
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.train_mode = train_mode
        self.questions = []
        self.databases = []
        self.q2d_idxes = []
        self.join_paths = []
        self._idx = None

    def read_data(self):
        logging.info('Data Holder Read Data')
        f1 = open(self.question_path)
        f2 = open(self.header_path)
        f3 = open(self.table_path)
        f4 = open(self.idx_path)
        f5 = open(self.from_path)
        f6 = open(self.column2table_path)

        while True:
            line1 = f1.readline()  # question
            line2 = f2.readline()  # header
            line3 = f3.readline()  # table
            line4 = f4.readline()  # idx
            line5 = f5.readline()  # from
            line6 = f6.readline()  # h2t

            if not line1:
                break
            if "NESTED" in line4:
                table_idxes = map(int, line5.strip().split())[:-1]
                for table_idx in table_idxes:
                    self.join_paths[-1].add(table_idx)
                continue

            self.questions.append(map(int, line1.strip().split()))
            self.join_paths.append(set(map(int, line5.strip().split())[:-1]))
            tables = []
            table_names = [map(int, table_name.strip().split()) for table_name in line3.strip().split(" %d " % self.sep_token)]
            h2t = map(int, line6.strip().split())
            column_names = [map(int, column_name.strip().split()) for column_name in line2.strip().split(" %d " % self.sep_token)]
            for table_name in table_names:
                tables.append({
                    "table_name": table_name,
                    "columns": []
                })
            for idx, zip_idx in enumerate(zip(h2t, column_names)):
                c_idx, c_name = zip_idx
                tables[c_idx]['columns'].append(c_name)
            if len(self.databases) == 0:
                self.databases.append(tables)
                self.q2d_idxes.append(0)
            else:
                if self.databases[-1] == tables:
                    pass
                else:
                    self.databases.append(tables)
                self.q2d_idxes.append(len(self.databases) - 1)

    def __len__(self):
        if not self.questions:
            self.read_data()
        return len(self.questions)

    def reset(self):
        if self.shuffle_mode is None:
            idx = numpy.arange(0, len(self))
            self._idx = [idx[i:i + self.batch_size] for i in xrange(0, len(self), self.batch_size)]
        elif self.shuffle_mode == 'simple':
            idx = numpy.random.permutation(len(self))
            self._idx = [idx[i:i + self.batch_size] for i in xrange(0, len(self), self.batch_size)]
        else:
            raise

    def get_batch_data(self):

        for idxes in self._idx:
            yield self.prepare_data([self.questions[idx] for idx in idxes],
                                    [self.q2d_idxes[idx] for idx in idxes],
                                    [list(self.join_paths[idx]) for idx in idxes],
                                    [self.databases[self.q2d_idxes[idx]] for idx in idxes]
                                    )

    def prepare_data(self, questions, q2d_idxes, join_paths, databases):
        if self.train_mode == "inner" and self.partition == "train":
            sample_size = len(questions)
            max_len_q = max([len(question) for question in questions]) + 2
            max_len_t = 0
            max_num_t = max([len(database) for database in databases])
            for database in databases:
                for table in database:
                    cur_len_t = len(table['table_name']) + 2
                    for column in table['columns']:
                        cur_len_t += (len(column) + 1)
                    if cur_len_t > max_len_t:
                        max_len_t = cur_len_t
            x_q = numpy.zeros((sample_size, max_len_q), dtype="int64")
            x_q_mask = numpy.zeros_like(x_q, dtype="float32")
            x_c = numpy.zeros((sample_size, max_num_t, max_len_t), dtype="int64")
            x_c_mask = numpy.zeros_like(x_c, dtype="float32")

            f_mask = numpy.zeros((sample_size, max_num_t, max_num_t), dtype="float32")
            for i, data in enumerate(zip(questions, join_paths, databases)):
                question, join_path, database = data
                question = [self.cls_token] + question + [self.sep_token]
                x_q[i, :len(question)] = question
                x_q_mask[i, :len(question)] = 1.
                for j, table in enumerate(database):
                    header = [self.cls_token] + table['table_name'] + [self.sep_token]
                    for column in table['columns']:
                        header.extend(column + [self.sep_token])
                    x_c[i, j, :len(header)] = header
                    x_c_mask[i, j, :len(header)] = 1.
                    if j in join_path:
                        for k in xrange(len(database)):
                            if k not in join_path and k != 0:
                                f_mask[i, j, k] = 1.

                x_c_mask[i, len(database):, 0] = 1.
            return x_q, x_q_mask, x_c, x_c_mask, f_mask
        elif self.train_mode == "all" and self.partition == "train":
            assert self.train_mode == "all" or self.partition != "train"
            sample_size = len(questions)
            max_len_q = max([len(question) for question in questions]) + 2
            max_len_t = 0
            num_t = 0
            for i, database in enumerate(databases):
                num_t += len(database)
                for table in database:
                    cur_len_t = len(table['table_name']) + 2
                    for column in table['columns']:
                        cur_len_t += (len(column) + 1)
                    if cur_len_t > max_len_t:
                        max_len_t = cur_len_t

            x_q = numpy.zeros((sample_size, max_len_q), dtype="int64")
            x_q_mask = numpy.zeros_like(x_q, dtype="float32")
            x_c = numpy.zeros((num_t, max_len_t), dtype="int64")
            x_c_mask = numpy.zeros_like(x_c, dtype="float32")
            f_mask = numpy.zeros((sample_size, num_t, num_t), dtype="float32")
            t_idx = 0
            for i, data in enumerate(zip(questions, join_paths, databases)):
                question, join_path, database = data
                question = [self.cls_token] + question + [self.sep_token]
                x_q[i, :len(question)] = question
                x_q_mask[i, :len(question)] = 1.
                for j, table in enumerate(database):
                    header = [self.cls_token] + table['table_name'] + [self.sep_token]
                    for column in table['columns']:
                        header.extend(column + [self.sep_token])
                    x_c[t_idx, :len(header)] = header
                    x_c_mask[t_idx, :len(header)] = 1.
                    if j in join_path:
                        t_idx_inner = 0
                        for k, db_inner in enumerate(databases):
                            if k == i:
                                for m in xrange(len(db_inner)):
                                    if m not in join_path and m != 0:
                                        f_mask[i, t_idx, t_idx_inner] = 1.
                                    t_idx_inner += 1
                            elif q2d_idxes[i] == q2d_idxes[k]:
                                t_idx_inner += len(db_inner)
                            else:
                                f_mask[i, t_idx, t_idx_inner + 1:t_idx_inner + len(db_inner)] = 1.
                                t_idx_inner += len(db_inner)
                        assert t_idx_inner == num_t

                    t_idx += 1

            assert t_idx == num_t
            return x_q, x_q_mask, x_c, x_c_mask, f_mask
        elif self.partition == "dev":
            sample_size = len(questions)
            max_len_q = max([len(question) for question in questions]) + 2
            max_len_t = 0
            max_num_t = max([len(database) for database in databases])
            for database in databases:
                for table in database:
                    cur_len_t = len(table['table_name']) + 2
                    for column in table['columns']:
                        cur_len_t += (len(column) + 1)
                    if cur_len_t > max_len_t:
                        max_len_t = cur_len_t
            x_q = numpy.zeros((sample_size, max_len_q), dtype="int64")
            x_q_mask = numpy.zeros_like(x_q, dtype="float32")
            x_c = numpy.zeros((sample_size, max_num_t, max_len_t), dtype="int64")
            x_c_mask = numpy.zeros_like(x_c, dtype="float32")
            t_mask = numpy.zeros((sample_size, max_num_t), dtype="float32")

            for i, data in enumerate(zip(questions, databases)):
                question, database = data
                question = [self.cls_token] + question + [self.sep_token]
                x_q[i, :len(question)] = question
                x_q_mask[i, :len(question)] = 1.
                for j, table in enumerate(database):
                    header = [self.cls_token] + table['table_name'] + [self.sep_token]
                    for column in table['columns']:
                        header.extend(column + [self.sep_token])
                    x_c[i, j, :len(header)] = header
                    x_c_mask[i, j, :len(header)] = 1.
                    t_mask[i, j] = 1.

                x_c_mask[i, len(database):, 0] = 1.
            return x_q, x_q_mask, x_c, x_c_mask, t_mask, join_paths
        else:
            raise

"""
if __name__ == '__main__':
    data_holder = RetrievalHolder(
        question_path="../data/build/build_train.question",
        header_path="../data/build/build_train.header",
        table_path="../data/build/build_train.table",
        idx_path="../data/build/build_train.idx",
        from_path="../data/build/build_train.f_t",
        column2table_path="../data/build/build_train.h2t",
        batch_size=4,
        train_mode="all"

    )

    data_holder.read_data()
    data_holder.reset()
    for data in data_holder.get_batch_data():
        x_q, x_q_mask, x_c, x_c_mask, f_mask = data
"""