import re
import sys

ori_ = sys.argv[1]
out_ = sys.argv[2]

fsp = open('./specialToken.txt', 'r', encoding='utf-8')
config = {'maxlen': 60, 'special_token': [x.strip() for x in fsp]}
fsp.close()

'''def get_labels(sent):
    labels = []
    for x in sent:
        labels += ['0'] * (len(x) - 1) + ['1']
    return labels'''

def get_single_data(data, config):
    sents = list()
    # labels = list()
    for sent in data:
        Left = 0
        sent = sent.split()
        for idx,word in enumerate(sent):
            if word not in config['special_token']:
                if len(re.sub('\W','',word,flags=re.U))==0:
                    if idx > Left:
                        slen = len(list(''.join(sent[Left:idx])))
                        if slen <= config['maxlen']:
                            sents.append(sent[Left:idx])
                            # sents.append(''.join(list(sent[Left:idx])))
                            # labels.append(get_labels(sent[Left:idx]))
                    Left = idx+1
        if Left!=len(sent):
            slen = len(list(''.join(sent[Left:])))
            if slen <= config['maxlen']:
                sents.append(sent[Left:])
                # sents.append(''.join(list(sent[Left:])))
                # labels.append(get_labels(sent[Left:]))
    # return sents, labels
    return sents

fin = open(ori_, 'r', encoding='utf-8')
data = [s.strip() for s in fin]
fin.close()

sents = get_single_data(data, config)

fout = open(out_, 'w', encoding='utf-8')
for s in sents:
    fout.write(' '.join(s) + '\n')
fout.close()
