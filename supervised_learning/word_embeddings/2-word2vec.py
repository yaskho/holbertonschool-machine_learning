#!/usr/bin/env python3
import gensim
from gensim.test.utils import common_texts

kw = dict(vector_size=100, min_count=1, window=5, negative=5,
          seed=0, workers=1)

# A: my current version (build_vocab, then train)
a = gensim.models.Word2Vec(sg=0, **kw)
a.build_vocab(common_texts)
a.train(common_texts, total_examples=a.corpus_count, epochs=5)

# B: everything in the constructor
b = gensim.models.Word2Vec(common_texts, sg=0, epochs=5, **kw)

# C: constructor, then an extra train call (trains twice)
c = gensim.models.Word2Vec(common_texts, sg=0, **kw)
c.train(common_texts, total_examples=c.corpus_count, epochs=5)

# D: same as A but with Skip-gram
d = gensim.models.Word2Vec(sg=1, **kw)
d.build_vocab(common_texts)
d.train(common_texts, total_examples=d.corpus_count, epochs=5)

for name, m in (("A", a), ("B", b), ("C", c), ("D", d)):
    print(name, m.wv["computer"][:2])