#/user/bin/env python
#coding=utf-8
# import jieba
# import pickle
# import word2vec
# import random
# from googletrans import Translator
# from sklearn.metrics.classification import classification_report
# import jieba.posseg as pseg
#
#
# class match_data:
#     def __init__(self,t1,t2,label=None):
#         self.t1=t1
#         self.property1 = []
#         self.t2=t2
#         self.property2 = []
#         self.label=label
#
#
# def csv_reader():
#     class0=0
#     class1=0
#     data=[]
#     with open('./data3/atec_nlp_sim_train.csv','r',encoding='utf-8') as f:
#         for line in f:
#             line=line.strip('\n')
#             strs=line.split('\t')
#             data.append(match_data(strs[1], strs[2], int(strs[3])))
#             if int(strs[3])==0:
#                 class0+=1
#             else:
#                 class1+=1
#     print(class0)
#     print(class1)
#     with open('./data3/atec_nlp_sim_train_add.csv','r',encoding='utf-8') as f:
#         for line in f:
#             line = line.strip('\n')
#             strs = line.split('\t')
#             data.append(match_data(strs[1], strs[2], int(strs[3])))
#             if int(strs[3]) == 0:
#                 class0 += 1
#             else:
#                 class1 += 1
#     print(class0)
#     print(class1)
#     trans_data = []
#     translate_to_data(open('./data3/initialtext1.txt', 'r', encoding='utf-8'),
#                       open('./data3/translate1.txt', 'r', encoding='utf-8'), trans_data)
#     translate_to_data(open('./data3/initialtext2.txt', 'r', encoding='utf-8'),
#                       open('./data3/translate2.txt', 'r', encoding='utf-8'), trans_data)
#     translate_to_data(open('./data3/initialtext3.txt', 'r', encoding='utf-8'),
#                       open('./data3/translate3.txt', 'r', encoding='utf-8'), trans_data)
#     translate_to_data(open('./data3/initialtext4.txt', 'r', encoding='utf-8'),
#                       open('./data3/translate4.txt', 'r', encoding='utf-8'), trans_data)
#     translate_to_data(open('./data3/add1.txt', 'r', encoding='utf-8'),
#                       open('./data3/addtranslate1.txt', 'r', encoding='utf-8'), trans_data)
#     translate_to_data(open('./data3/add2.txt', 'r', encoding='utf-8'),
#                       open('./data3/addtranslate2.txt', 'r', encoding='utf-8'), trans_data)
#     translate_to_data(open('./data3/add3.txt', 'r', encoding='utf-8'),
#                       open('./data3/addtranslate3.txt', 'r', encoding='utf-8'), trans_data)
#     return data, trans_data
#
#
# def translate_to_data(initial_input, trans_input, data):
#     for line in initial_input:
#         line = line.strip('\n').split('\t')
#         data.append(match_data(line[1], trans_input.readline().strip('\n'), 1))
#         data.append(match_data(line[2], trans_input.readline().strip('\n'), 1))
#
#
# def translate_new_data(infile, outfile):
#     translator = Translator()
#     i = 0
#     fin = open(infile, 'r', encoding='utf-8')
#     fout = open(outfile, 'a+', encoding='utf-8')
#     lines = fin.readlines(100000)
#     length = len(lines)
#     last_length = 0
#     state = 1
#     while True:
#         try:
#             if state == 1:
#                 if i == length:
#                     lines = fin.readlines(100000)
#                     if len(lines) == 0:
#                         break
#                     else:
#                         last_length = length
#                         length += len(lines)
#                 line = lines[i - last_length]
#                 print(i + 1)
#                 line = lines[i]
#                 fields = line.strip('\n').split('\t')
#                 english = translator.translate(fields[1], 'en', 'zh-cn')
#                 chinese = translator.translate(english.text, 'zh-cn', 'en')
#                 txt1 = chinese.text
#                 #fout.write(chinese.text + '\n')
#                 english = translator.translate(fields[2], 'en', 'zh-cn')
#                 chinese = translator.translate(english.text, 'zh-cn', 'en')
#                 txt2 = chinese.text
#                 #fout.write(chinese.text + '\n')
#                 i += 1
#             elif state == 0:
#                 print(i + 1)
#                 fields = line.strip('\n').split('\t')
#                 english = translator.translate(fields[1], 'en', 'zh-cn')
#                 chinese = translator.translate(english.text, 'zh-cn', 'en')
#                 txt1 = chinese.text
#                 #fout.write(chinese.text + '\n')
#                 english = translator.translate(fields[2], 'en', 'zh-cn')
#                 chinese = translator.translate(english.text, 'zh-cn', 'en')
#                 txt2 = chinese.text
#                 #fout.write(chinese.text + '\n')
#                 i += 1
#                 state = 1
#         except:
#             print('error error error error')
#             state = 0
#         else:
#             fout.write(txt1 + '\n')
#             fout.write(txt2 + '\n')
#     fin.close()
#     fout.close()
#
#
#
def load_property():
   property_index = {}
   with open('./data3/property.txt', 'r', encoding='UTF-8') as f:
       for line in f:
           line = line.strip('\n').split(' ')
           property_index[line[0]] = int(line[1])
   return property_index
#
#
# def save_property():
#     property = []
#     tmp = [0] * 41
#     property.append(tmp)
#     for i in range(41):
#         tmp[i] = 1
#         property.append(tmp)
#         tmp[i] = 0
#     pickle.dump(property, open('./data3/property.pkl', 'wb'), protocol=True)
#
#
# def gendata_segdata_addproperty(data, trans_data):
#     property_index = load_property()
#     jieba.load_userdict("./data3/userdict.txt")
#     with open('./data3/word2vec_corpus.txt', 'w', encoding='utf-8') as f:
#         for d in data + trans_data:
#             words = []
#             flags = []
#             word_and_flag = pseg.cut(d.t1)
#             for word, flag in word_and_flag:
#                 if (flag in property_index) and (not word.isspace()):
#                     words.append(word)
#                     flags.append(property_index[flag])
#             r = ' '.join(words)
#             f.write(r + '\n')
#             d.t1 = words
#             d.property1 = flags
#             words = []
#             flags = []
#             word_and_flag = pseg.cut(d.t2)
#             for word, flag in word_and_flag:
#                 if flag in property_index:
#                     words.append(word)
#                     flags.append(property_index[flag])
#             r = ' '.join(words)
#             f.write(r + '\n')
#             d.t2 = words
#             d.property2 = flags
#         pickle.dump(data, open('./data3/seg_data.pkl', 'wb'), protocol=True)
#         pickle.dump(trans_data, open('./data3/seg_trans_data.pkl', 'wb'), protocol=True)
#
#
# def pre_train_word_embedding():
#     word2vec.word2vec('./data3/word2vec_corpus.txt', './data3/word_embedding.bin', size=200, window=6, sample='1e-5',
#                       cbow=0, save_vocab='./data3/worddict', min_count=1, iter_=30)
#
#
# def load_word_embedding():
#     # word_embedding:[clusters=None,vectors,vocab,vocab_hash]
#     word_embedding = word2vec.load('./data3/word_embedding.bin')
#     return word_embedding
#
#
# def seg_to_index():
#     seg_data=pickle.load(open('./data3/seg_data.pkl','rb'))
#     seg_trans_data = pickle.load(open('./data3/seg_trans_data.pkl', 'rb'))
#     word_embedding=load_word_embedding()
#     for d in seg_data + seg_trans_data:
#         tmp = d.t1
#         words = []
#         flags = []
#         for i in range(len(tmp)):
#             if tmp[i] in word_embedding.vocab_hash:
#                 words.append(word_embedding.vocab_hash[tmp[i]])
#                 flags.append(d.property1[i])
#         d.t1 = words
#         d.property1 = flags
#         words = []
#         flags = []
#         tmp = d.t2
#         for i in range(len(tmp)):
#             if tmp[i] in word_embedding.vocab_hash:
#                 words.append(word_embedding.vocab_hash[tmp[i]])
#                 flags.append(d.property2[i])
#         d.t2 = words
#         d.property2 = flags
#     pickle.dump(seg_data,open('./data3/index_data.pkl','wb'),protocol=True)
#     pickle.dump(seg_trans_data, open('./data3/index_trans_data.pkl', 'wb'), protocol=True)
#     pickle.dump(word_embedding.vectors,open('./data3/word_embedding.pkl','wb'),protocol=True)
#
#
# def stat():
#     total_len=0
#     max_len=0
#     data=pickle.load(open('./data3/index_trans_data.pkl', 'rb'))
#     for d in data:
#         if max_len<len(d.t1):
#             max_len=len(d.t1)
#         if max_len<len(d.t2):
#             max_len=len(d.t2)
#         total_len+=(len(d.t1)+len(d.t2))
#     print(total_len/(2*len(data)))
#     print(max_len)
#
#
# def partition():
#     data = pickle.load(open('./data3/index_data.pkl', 'rb'))
#     positive=[d for d in data if d.label==1]
#     negative=[d for d in data if d.label==0]
#     random.shuffle(positive)
#     random.shuffle(negative)
#     val=[positive[i] for i in range(0,len(positive)) if i<1500]+[negative[i] for i in range(0,len(negative)) if i<6000]
#     train=[positive[i] for i in range(0,len(positive)) if i>=1500]*4+[negative[i] for i in range(0,len(negative)) if i>=6000]
#     random.shuffle(val)
#     random.shuffle(train)
#     val_q=[d.t1 for d in val]
#     val_pro_q = [d.property1 for d in val]
#     val_r=[d.t2 for d in val]
#     val_pro_r = [d.property2 for d in val]
#     val_label=[d.label for d in val]
#     train_q = [d.t1 for d in train]
#     train_pro_q = [d.property1 for d in train]
#     train_r = [d.t2 for d in train]
#     train_pro_r = [d.property2 for d in train]
#     train_label = [d.label for d in train]
#     pickle.dump([train_q, train_pro_q, train_r, train_pro_r, train_label],open('./data3/train.pkl','wb'),protocol=True)
#     pickle.dump([val_q, val_pro_q, val_r, val_pro_r, val_label], open('./data3/val.pkl', 'wb'), protocol=True)
#     index_trans_data = pickle.load(open('./data3/index_trans_data.pkl', 'rb'))
#     trans_q = [d.t1 for d in index_trans_data]
#     trans_q_pro = [d.property1 for d in index_trans_data]
#     trans_r = [d.t2 for d in index_trans_data]
#     trans_r_pro = [d.property2 for d in index_trans_data]
#     trans_label = [d.label for d in index_trans_data]
#     pickle.dump([trans_q, trans_q_pro, trans_r, trans_r_pro, trans_label],
#                 open('./data3/trans_train.pkl', 'wb'), protocol=True)
#
#
# def compare():
#     with open('./output.txt','r',encoding='utf-8') as f:
#         pred_label=[]
#         for line in f:
#             arr=line.strip('\n').split('\t')
#             pred_label.append(int(arr[1]))
#     with open('./data/test.csv','r',encoding='utf-8') as f:
#         label=[]
#         for line in f:
#             lineno, sen1, sen2, tmp = line.strip().split('\t')
#             label.append(int(tmp))
#     print(classification_report(label,pred_label))
#
#
# def dig_synonyms_and_antonym():
#     data=csv_reader()
#     for d in data:
#         text_seg1 = pseg.cut(d.t1)
#         text_seg2 = pseg.cut(d.t2)
#         text_words1=[w.word for w in text_seg1]
#         text_pos1=[w.flag for w in text_seg1]
#         text_words2=[w.word for w in text_seg2]
#         text_pos2=[w.word for w in text_seg2]
#         if d.label==1:#进入同义词统计模式
#             for i in range(0,len(text_words1)):
#                 for j in range(0,len(text_words2)):
#                     if text_words1[i]==text_words2[j] and text_pos1[i]==text_pos2[j]:
#                         break
#         else:
#             pass
#
#
#
# def load_label():
#     file_src_dict = {'embedding_file': './data/word_embedding.pkl', 'train_file': './data/train.pkl',
#                      'evaluate_file': './data/val.pkl'}
#     with open(file_src_dict['evaluate_file'], 'rb') as f:
#         val_q, val_r, val_labels = pickle.load(f)
#     with open(file_src_dict['train_file'], 'rb') as f:
#         train_q, train_r, train_labels = pickle.load(f)
#     return val_labels,train_labels
#
# def random_train():
#     train = pickle.load(open('./data3/train.pkl', 'rb'))
#     text1 = []
#     text1_pro = []
#     text2 = []
#     text2_pro = []
#     all_utterances = []
#     all_utterances_pro = []
#     for i in range(0, len(train[0])):
#         t1, p1, t2, p2, label = train[0][i], train[1][i], train[2][i], train[3][i], train[4][i]
#         all_utterances.append(t1)
#         all_utterances.append(t2)
#         all_utterances_pro.append(p1)
#         all_utterances_pro.append(p2)
#         if label == 1:
#             text1.append(t1)
#             text1_pro.append(p1)
#             text2.append(t2)
#             text2_pro.append(p2)
#     pickle.dump([text1, text1_pro, text2, text2_pro], open('./data3/random_train.pkl', 'wb'), protocol=True)
#     pickle.dump(all_utterances, open('./data3/all_utterances', 'wb'), protocol=True)
#     pickle.dump(all_utterances_pro, open('./data3/all_utterances_pro', 'wb'), protocol=True)
#     print('all work has finished')
#
#
# def count(str, c):
#     i = 0
#     for s in str:
#         if s == c:
#             i += 1
#     return i
#
# def stat_abnormal(c):
#     paths = ['./data3/translate1.txt', './data3/translate2.txt', './data3/translate3.txt', './data3/translate4.txt',
#              './data3/addtranslate1.txt', './data3/addtranslate2.txt', './data3/addtranslate3.txt']
#     for i in range(len(paths)):
#         print()
#         print()
#         print(paths[i])
#         with open(paths[i], 'r', encoding='utf-8') as f:
#             j = 1
#             for line in f:
#                 if count(line, c) > 8:
#                     print('initial:', int((j - 1) / 2) + 1, 'translate:', j, line)
#                 j += 1
#
#
# if __name__=="__main__":
    #save_property()
    # print('csv_read')
    #data, trans_data = csv_reader()
    #gendata_segdata_addproperty(data, trans_data)
    #print('train word2vec')
    #pre_train_word_embedding()
    # print('seg data')
    # gendata_segdata_addproperty(data)
    # print('seg to index')
    # seg_to_index()
    # print('partition')
    # partition()
    # print('random train')
    # random_train()
    #stat()
    #translate_new_data('./data2/initialtext3.txt', './data2/translate3.txt')
    #print('handle initial_text4')
    #translate_new_data('./data2/initialtext4.txt', './data2/translate4.txt')
    #stat_abnormal('*')