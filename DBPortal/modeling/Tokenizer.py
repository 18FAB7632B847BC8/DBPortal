from ChineseTokenizer import BasicTokenizer, WordpieceTokenizer


def match(x, y):
    flag = True
    for i, j in zip(x, y):
        if i != j:
            flag = False
            break
    return flag


def segment_match(q_t, v_t):
    count = 0
    len_v = len(v_t)
    for i in range(len(q_t)):
        if match(q_t[i:i + len_v], v_t):
            count += 1
    return count


def tokenize():
    partition = "train"

    fr_q = open("../data/%s.question" % partition)
    fr_v = open("../data/clean_%s.value_l" % partition)

    bt = BasicTokenizer(True)
    idx = 0
    for line_q, line_v in zip(fr_q, fr_v):
        idx += 1
        question = line_q.strip().decode('utf-8')
        values = line_v.strip().decode('utf-8').split(' || ')
        q_t = bt.tokenize(question)
        for value in values:
            if not value or value == "None" or 'NESTED' in value or 'COLUMN' in value:
                continue
            v_t = bt.tokenize(value)
            count = segment_match(q_t, v_t)
            if count != 1:
                print question

def get_dict():
    from json import load
    fr = open("../bert.base/bert-base-vocab.json")
    fw = open("../bert.base/vocab.txt", 'w')

    vocab_r = load(fr, encoding="utf-8")
    vocab = dict()
    for k, v in vocab_r.iteritems():
        vocab[v] = k

    for i in range(len(vocab)):
        fw.write("%s\n" % vocab[i].encode('utf-8'))

    fr.close()
    fw.flush()
    fw.close()


def test_dict():
    s = set()
    with open("../bert.base/vocab.txt") as f:
        for line in f:
            s.add(line.strip())
    print len(s)

    # test_dict()