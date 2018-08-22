#/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib import rnn
import pickle
import time
from datetime import timedelta
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.classification import classification_report
import sys
import jieba
import os
from gcforest.gcforest import GCForest
import random

class ARNN:
    def __init__(self,embedding):
        self.learning_rate=1e-3#1e-4到了0.63
        self.batch_size=128
        self.epoch_num=20
        self.print_batch = 40
        self.total_words = embedding.shape[0]
        self.embedding = embedding
        self.word_embedding_size=embedding.shape[1]
        self.max_sentence_len=100000
        self.l2=0.0
        self.kp=1.0

    def build_base_model(self,rnn_units=200,interaction_dims=16,training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                          word_embedding_size), dtype=tf.float32,
                                              trainable=True)  # 我发现这个任务embedding设为trainable很重要
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_r)
        with tf.variable_scope('first_encodeing'):
            sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer())
            sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(sentence_GRU, output_keep_prob=self.keep_prob)
            q_gru, q_last_state = tf.nn.dynamic_rnn(sentence_GRU, q_embedding,
                                                  sequence_length=self.q_sequence_len,
                                                  dtype=tf.float32)
            r_gru, r_last_state = tf.nn.dynamic_rnn(sentence_GRU, r_embedding,
                                                  sequence_length=self.r_sequence_len,
                                                  dtype=tf.float32)
        with tf.variable_scope('interaction'):
            interaction_weight = tf.get_variable(name='interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            ileft = tf.tensordot(r_last_state, interaction_weight, axes=(1, 0))
            ileft = tf.expand_dims(ileft, axis=2)
            iright = tf.stack([q_last_state] * interaction_dims, axis=1)
            iright = tf.expand_dims(iright, axis=3)
            interaction = tf.matmul(ileft, iright)[:, :, -1, -1]
            self.final_matching_vector=interaction
        with tf.variable_scope('output'):
            logits = tf.layers.dense(self.final_matching_vector, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     name='output')
            self.y_pred = tf.nn.softmax(logits)
            self.class_label_pred = tf.argmax(self.y_pred, 1)  # 预测类别
        with tf.variable_scope('optimze'):
            self.total_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits))
            tf.summary.scalar('loss', self.total_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)
        if training:
            i = 0
            while os.path.exists('./base' + str(i)):
                i += 1
            os.makedirs('./base' + str(i))
            return './base' + str(i)

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def get_sequences_length(self,sequences, maxlen):
        sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
        return sequences_length

    def predict(self,model_path,data_q,data_r):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 只分配40%的显存
        all_pred_label=[]
        all_pred_value=[]
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            batch_size_for_val = 1000
            low=0
            while True:
                n_sample = min(low + batch_size_for_val, len(data_q)) - low
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_q = pad_sequences(data_q[low:low + n_sample], padding='post')
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r = pad_sequences(data_r[low:low + n_sample], padding='post')
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.keep_prob:1.0
                }
                pred_label,pred_value = sess.run([self.class_label_pred,self.y_pred], feed_dict=feed_dict)
                all_pred_label.append(pred_label)
                all_pred_value.append(pred_value)
                low = low + batch_size_for_val
                if low >= len(data_q):
                    break
            all_pred_label = np.concatenate(all_pred_label, axis=0)
            all_pred_value = np.concatenate(all_pred_value,axis=0)
            return all_pred_label,all_pred_value

    def gen_feature(self,model_path,data_q,data_r):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 只分配40%的显存
        all_feature=[]
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            batch_size_for_val = 300
            low=0
            while True:
                n_sample = min(low + batch_size_for_val, len(data_q)) - low
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_q = pad_sequences(data_q[low:low + n_sample], padding='post')
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r = pad_sequences(data_r[low:low + n_sample], padding='post')
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.keep_prob:1.0
                }
                feature = sess.run(self.final_matching_vector, feed_dict=feed_dict)
                all_feature.append(feature)
                low = low + batch_size_for_val
                if low >= len(data_q):
                    break
            all_feature = np.concatenate(all_feature, axis=0)
            #print(all_feature.shape)
            return all_feature

    def evaluate_val_for_train(self, sess, data):
        val_q, val_r, val_labels = data
        all_pred_label = []
        low = 0
        batch_size_for_val=1500
        while True:
            n_sample = min(low + batch_size_for_val, len(val_labels)) - low
            batch_q_len = self.get_sequences_length(val_q[low:low + n_sample], maxlen=self.max_sentence_len)
            batch_q = pad_sequences(val_q[low:low + n_sample], padding='post')
            batch_r_len = self.get_sequences_length(val_r[low:low + n_sample], maxlen=self.max_sentence_len)
            batch_r = pad_sequences(val_r[low:low + n_sample], padding='post')
            feed_dict = {
                self.input_q: np.array(batch_q),
                self.q_sequence_len: np.array(batch_q_len),
                self.input_r: np.array(batch_r),
                self.r_sequence_len: np.array(batch_r_len),
                self.input_y: np.array(val_labels[low:low + n_sample]),
                self.keep_prob:1.0
            }
            pred_label,loss = sess.run([self.class_label_pred,self.total_loss], feed_dict=feed_dict)
            all_pred_label.append(pred_label)
            low = low + batch_size_for_val
            if low >= len(val_labels):
                break
        all_pred_label = np.concatenate(all_pred_label, axis=0)
        return loss,classification_report(val_labels,all_pred_label)

    def train_model(self, file_src_dict, store_path, continue_train=False,
                                    previous_model_path="model"):
        with open(store_path+'/papr.txt','w+') as f:
            f.write('lr:'+str(self.learning_rate)+'\n')
            f.write('kp'+str(self.kp)+'\n')
            f.write('l2'+str(self.l2) + '\n')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 只分配40%的显存
        # prepare data for val:
        with open(file_src_dict['evaluate_file'], 'rb') as f:
            val_q, val_r, val_labels = pickle.load(f)
        val_data=[val_q, val_r, val_labels]
        with tf.Session(config=config) as sess:
            file=open(store_path+'/record.txt','w+')
            train_writer = tf.summary.FileWriter(store_path, sess.graph)
            # prepare data for train:
            with open(file_src_dict['train_file'], 'rb') as f:
                train_q, train_r, train_labels = pickle.load(f)
            if continue_train is False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
            else:
                saver.restore(sess, previous_model_path)
            low = 0
            epoch = 1
            start_time = time.time()
            sess.graph.finalize()
            best_score=100
            while epoch < self.epoch_num:
                # low means the start location of the array of data should be feed in next
                # n_samples means how many group-samples will be feed in next time
                # one group-samples means one context and its true response and some neg responses
                n_sample = min(low + self.batch_size, len(train_labels)) - low
                batch_q_len = self.get_sequences_length(self.copy_list(train_q[low:low + n_sample]), maxlen=self.max_sentence_len)
                batch_q = pad_sequences(self.copy_list(train_q[low:low + n_sample]), padding='post')
                batch_r_len = self.get_sequences_length(self.copy_list(train_r[low:low + n_sample]), maxlen=self.max_sentence_len)
                batch_r = pad_sequences(self.copy_list(train_r[low:low + n_sample]), padding='post')
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.input_y: np.array(train_labels[low:low + n_sample]),
                    self.keep_prob:self.kp
                }
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                if low % (self.batch_size * self.print_batch) == 0:
                    time_dif = self.__get_time_dif(start_time)
                    loss,clf_report=self.evaluate_val_for_train(sess, val_data)
                    if best_score>loss:
                        best_score=loss
                        saver.save(sess, store_path+"/model_best.{0}".format(low))
                    train_loss=sess.run(self.total_loss, feed_dict=feed_dict)
                    print("train loss:", train_loss, "; val evaluation:"
                          ,loss,clf_report, "time:", time_dif)
                    file.write("train loss:"+str(train_loss)+ "; val evaluation:"+str(loss)+'\n'+clf_report+"\ntime:"+str(time_dif)+'\n')
                if low >= len(train_labels):  # 即low>=total conversations number
                    low = 0
                    saver.save(sess, store_path+"/model.{0}".format(epoch))
                    print('epoch={i}'.format(i=epoch), 'ended')
                    epoch += 1
            f.close()

    def copy_list(self,list):
        new_list=[]
        for l in list:
            if type(l)==type([0]) or type(l)==np.array([0]):
                new_list.append(self.copy_list(l))
            else:
                new_list.append(l)
        return new_list


def train_onehotkey():
    print('start')
    file_src_dict = {'embedding_file': './data/word_embedding.pkl','train_random_file':'./data/random_train.pkl','evaluate_file':'./data/val.pkl',
                     'all_utterances':'./data/all_utterances','train_fix_file':'./data/train.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    arnn = ARNN(embedding=embeddings)
    print('build graph')
    path=arnn.build_base_model()
    print('start train')
    arnn.train_model(file_src_dict=file_src_dict,store_path=path)

def process(inpath,outpath):
    jieba.load_userdict("./data/userdict.txt")
    vocab_hash=pickle.load(open('./data/word_dict.pkl','rb'))
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        data_q=[]
        data_r=[]
        line_nums=[]
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            line_nums.append(lineno)
            words1 = [w for w in jieba.cut(sen1) if w.strip()]
            words2 = [w for w in jieba.cut(sen2) if w.strip()]
            index1 = [vocab_hash[w] for w in words1 if w in vocab_hash]
            index2 = [vocab_hash[w] for w in words2 if w in vocab_hash]
            data_q.append(index1)
            data_r.append(index2)
        result=ensemble(data_q, data_r)
        for r,l in zip(result,line_nums):
            if r==1:
                fout.write(l + '\t1\n')
            else:
                fout.write(l + '\t0\n')

def ensemble(val_q,val_r,val_label=None):
    file_src_dict = {'embedding_file': './data/word_embedding.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    all_pred_score=[]
    all_pred_label=[]
    models = ['./base/model.5']
    model_name = ['base']
    graphs = [tf.Graph() for i in range(0, len(models))]
    for i in range(0,len(graphs)):
        with graphs[i].as_default():
            arnn = ARNN(embedding=embeddings)
            if model_name[i]=='base':
                arnn.build_base_model(training=False)
            pred_label, pred_score = arnn.predict(model_path=models[i],data_q=val_q,data_r=val_r)
            all_pred_score.append(pred_score)
            all_pred_label.append(pred_label)
            del arnn
    final_score=(all_pred_score[0]+all_pred_score[1])
    for i in range(2,len(all_pred_score)):
        final_score+=all_pred_score[i]
    final_score/=len(all_pred_score)
    final_label=[int(s[1]>s[0]) for s in final_score]
    if val_label is not None:
        for pred_l in all_pred_label:
            print(classification_report(val_label,pred_l))
        print(classification_report(val_label,final_label))
    return final_label


def ensemble_test():
    file_src_dict = {'embedding_file': './data/word_embedding.pkl'}
    with open(file_src_dict['embedding_file'], 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        embeddings = pickle.load(f)
    with open('./data/val.pkl', 'rb') as f:  # embedding is a 2d-list with size :vocab_size*dim
        val_q, val_r, val_label = pickle.load(f)
    all_pred_score=[]
    all_pred_label=[]
    models = ['./base/model.1']
    model_name = ['base']
    graphs = [tf.Graph() for i in range(0, len(models))]
    for i in range(0,len(graphs)):
        with graphs[i].as_default():
            arnn = ARNN(embedding=embeddings)
            if model_name[i]=='base':
                arnn.build_base_model(training=False)
            pred_label, pred_score = arnn.predict(model_path=models[i],data_q=val_q,data_r=val_r)
            all_pred_score.append(pred_score)
            all_pred_label.append(pred_label)
            del arnn
    final_score=(all_pred_score[0]+all_pred_score[1])
    for i in range(2,len(all_pred_score)):
        final_score+=all_pred_score[i]
    final_score/=len(all_pred_score)
    final_label=[int(s[1]>s[0]) for s in final_score]
    if val_label is not None:
        for pred_l in all_pred_label:
            print(classification_report(val_label,pred_l))
        print(classification_report(val_label,final_label))

if __name__=='__main__':
    process(sys.argv[1], sys.argv[2])
