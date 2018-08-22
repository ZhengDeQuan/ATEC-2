import word2vec
import numpy as np
from numpy import linalg
def load_word_embedding():
    # word_embedding:[clusters=None,vectors,vocab,vocab_hash]
    word_embedding = word2vec.load('./data/word_embedding.bin')
    return word_embedding

if __name__=='__main__':
    word_em=load_word_embedding()
    while True:
        a=input()
        b=input()
        try:
            v1=np.array(word_em.vocab_hash[a])
            v2=np.array(word_em.vocab_hash[b])
            dist = linalg.norm(v1 - v2)
            sim = 1.0 / (1.0 + dist)  # 归一化
            print(sim)
        except:
            print('wrong')
    print('all work has finished')