import math
import numpy as np
import random
import sys

from transformers import AutoConfig, AutoModel, AutoTokenizer

model_name = 'bert-base-chinese'
config = AutoConfig.from_pretrained(model_name, cache_dir=None)
model = AutoModel.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    cache_dir=None
)
embeddings = list(model.embeddings.word_embeddings.weight.detach().cpu().numpy())

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)
wv = {}
dist = {}
for w in tokenizer.vocab:
    idx = tokenizer.vocab[w]
    if idx >= 106:
        wv[w] = embeddings[idx]
        dist[w] = np.dot(wv[w], wv[w])

def get_labels(sent):
    labels = []
    for x in sent:
        labels += [0] * (len(x) - 1) + [1]
    return labels

random.seed(0)

N_MRT = 10
MASK_COUNT = 2

fin = open('train.cut', 'r')
fpred = open('pred_mlm.txt', 'r', encoding='utf-8')
fout = open('train.mrt', 'w', encoding='utf-8')
frisk = open('train.risk', 'w')

for idx, __s in enumerate(fin):
    s = __s.strip()
    if len(s) == 0:
        for i in range(0, N_MRT):
            fout.write('\n')
            frisk.write('0.0\n')
        continue

    S = s.split(' ')
    l = get_labels(S)
    g = s.replace(' ', '')
    
    len_l = len(l)
    min_pos = 1
    max_pos = max(int(len_l * 0.1), 1)
    cands = [l]
    
    D = {}
    while True:
        p = fpred.readline().strip().split(' ')
        if p[0] == '-':
            break
        start = int(p[0])
        end = int(p[1])
        D[(start, end)] = []
        for i, q in enumerate(p[2 : ]):
            pos = q.find('(')
            prob = math.exp(-float(q[ : pos]))
            pred_token = q[pos + 1 : -1]
            gold_token = g[i + start]
            if (gold_token in wv) and (pred_token in wv):
                sim = float(np.dot(wv[pred_token], wv[gold_token]) / np.sqrt(dist[pred_token] * dist[gold_token]))
                diff = min(1.0 - sim, 1.0)
            else:
                diff = 1.0
            D[(start, end)].append(prob * diff)
    
    for i in range(1, N_MRT):
        n_pos = random.randint(min_pos, max_pos)
        pos = random.sample(list(range(len_l)), n_pos)
        ll = l.copy()
        for p in pos:
            ll[p] = 1 - ll[p]
        cands.append(ll)
    
    for ll in cands:
        ll[-1] = 1
        for j in range(0, len_l):
            fout.write(g[j])
            if (j < len_l - 1) and (ll[j] == 1):
                fout.write(' ')
        fout.write('\n')
        mask_start = []
        mask_end = []
        start = 0
        for i, x in enumerate(ll):
            if x == 1:
                end = i + 1
                if end - start <= MASK_COUNT:
                    mask_start.append(start)
                    mask_end.append(end)
                else:
                    for j in range(start, end - MASK_COUNT + 1):
                        mask_start.append(j)
                        mask_end.append(j + MASK_COUNT)
                start = end

        scores = []
        for start, end in zip(mask_start, mask_end):
            if (start, end) in D:
                scores += D[(start, end)]
        risk = float(np.mean(scores))
        frisk.write('%f\n' % risk)

fin.close()
fout.close()
frisk.close()
