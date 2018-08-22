#/usr/bin/env python
#coding=utf-8
import jieba
import pickle
import word2vec
import random

from googletrans import Translator
from sklearn.metrics.classification import classification_report
import jieba.posseg as pseg
import time

class match_data:
    def __init__(self,t1,t2,label=None):
        self.t1=t1
        self.t2=t2
        self.label=label


def csv_reader():
    class0=0
    class1=0
    data=[]
    with open('./data/atec_nlp_sim_train.csv','r',encoding='utf-8') as f:
        for line in f:
            line=line.strip('\n')
            strs=line.split('\t')
            data.append(match_data(strs[1],strs[2],int(strs[3])))
            if int(strs[3])==0:
                class0+=1
            else:
                class1+=1
    print(class0)
    print(class1)
    with open('./data/atec_nlp_sim_train_add.csv','r',encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            strs = line.split('\t')
            data.append(match_data(strs[1], strs[2], int(strs[3])))
            if int(strs[3]) == 0:
                class0 += 1
            else:
                class1 += 1
    print(class0)
    print(class1)
    return data


def gen_data_for_word2vec_and_seg_data(data):
    jieba.load_userdict("./data/userdict.txt")
    # jieba.cut
    with open('./data/word2vec_corpus.txt','w+',encoding='utf-8') as f:
        for d in data:
            rst=jieba.cut(d.t1,cut_all=False)
            r=' '.join(rst)
            f.write(r+"\n")
            d.t1=r
            rst = jieba.cut(d.t2, cut_all=False)
            r = ' '.join(rst)
            f.write(r + "\n")
            d.t2=r
    pickle.dump(data,open('./data/seg_data.pkl','wb+'),protocol=True)


def pre_train_word_embedding():
    word2vec.word2vec('./data/word2vec_corpus.txt', './data/word_embedding.bin', size=200, window=6, sample='1e-5',
                      cbow=0, save_vocab='./data/worddict', min_count=1, iter_=30)


def load_word_embedding():
    # word_embedding:[clusters=None,vectors,vocab,vocab_hash]
    word_embedding = word2vec.load('./data/word_embedding.bin')
    return word_embedding


def seg_to_index():
    seg_data=pickle.load(open('./data/seg_data.pkl','rb'))
    word_embedding = load_word_embedding()
    for d in seg_data:
        tmp=d.t1.split(' ')
        d.t1=[word_embedding.vocab_hash[t] for t in tmp if t in word_embedding.vocab_hash]
        tmp = d.t2.split(' ')
        d.t2 = [word_embedding.vocab_hash[t] for t in tmp if t in word_embedding.vocab_hash]
    pickle.dump(seg_data,open('./data/index_data.pkl','wb+'),protocol=True)
    pickle.dump(word_embedding.vectors,open('./data/word_embedding.pkl','wb+'),protocol=True)


def stat():
    total_len=0
    max_len=0
    data=pickle.load(open('./data/index_data.pkl', 'rb'))
    for d in data:
        if max_len<len(d.t1):
            max_len=len(d.t1)
        if max_len<len(d.t2):
            max_len=len(d.t2)
        total_len+=(len(d.t1)+len(d.t2))
    print(total_len/(2*len(data)))
    print(max_len)


def partition():
    data = pickle.load(open('./data/index_data.pkl', 'rb'))
    positive=[d for d in data if d.label==1]
    negative=[d for d in data if d.label==0]
    random.shuffle(positive)
    random.shuffle(negative)
    val=[positive[i] for i in range(0,len(positive)) if i<1500]+[negative[i] for i in range(0,len(negative)) if i<6000]
    train=[positive[i] for i in range(0,len(positive)) if i>=1500]*6+[negative[i] for i in range(0,len(negative)) if i>=6000]
    random.shuffle(val)
    random.shuffle(train)
    val_q=[d.t1 for d in val]
    val_r=[d.t2 for d in val]
    val_label=[d.label for d in val]
    train_q = [d.t1 for d in train]
    train_r = [d.t2 for d in train]
    train_label = [d.label for d in train]
    pickle.dump([train_q,train_r,train_label],open('./data/train.pkl','wb+'),protocol=True)
    pickle.dump([val_q,val_r,val_label], open('./data/val.pkl', 'wb+'), protocol=True)


def compare():
    with open('./output.txt','r',encoding='utf-8') as f:
        pred_label=[]
        for line in f:
            arr=line.strip('\n').split('\t')
            pred_label.append(int(arr[1]))
    with open('./data/test.csv','r',encoding='utf-8') as f:
        label=[]
        for line in f:
            lineno, sen1, sen2, tmp = line.strip().split('\t')
            label.append(int(tmp))
    print(classification_report(label,pred_label))


def dig_synonyms_and_antonym():
    data=csv_reader()
    for d in data:
        text_seg1 = pseg.cut(d.t1)
        text_seg2 = pseg.cut(d.t2)
        text_words1=[w.word for w in text_seg1]
        text_pos1=[w.flag for w in text_seg1]
        text_words2=[w.word for w in text_seg2]
        text_pos2=[w.word for w in text_seg2]
        if d.label==1:#进入同义词统计模式
            for i in range(0,len(text_words1)):
                for j in range(0,len(text_words2)):
                    if text_words1[i]==text_words2[j] and text_pos1[i]==text_pos2[j]:
                        break
        else:
            pass
        #todo


def load_label():
    file_src_dict = {'embedding_file': './data/word_embedding.pkl', 'train_file': './data/train.pkl',
                     'evaluate_file': './data/val.pkl'}
    with open(file_src_dict['evaluate_file'], 'rb') as f:
        val_q, val_r, val_labels = pickle.load(f)
    with open(file_src_dict['train_file'], 'rb') as f:
        train_q, train_r, train_labels = pickle.load(f)
    return val_labels,train_labels


if __name__=="__main__":
    # train=pickle.load(open('./data0/train.pkl','rb'))
    # text1=[]
    # text2=[]
    # all_utterances=[]
    # for i in range(0,len(train[0])):
    #     t1, t2, label =train[0][i],train[1][i],train[2][i]
    #     all_utterances.append(t1)
    #     all_utterances.append(t2)
    #     if label==1:
    #         text1.append(t1)
    #         text2.append(t2)
    # pickle.dump([text1,text2],open('./data0/random_train.pkl','wb+'),protocol=True)
    # pickle.dump(all_utterances,open('./data0/all_utterances','wb+'),protocol=True)
    # print('all work has finished')
    translate_new_data2()

