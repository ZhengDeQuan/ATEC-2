#coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, flatten
from tensorflow.contrib import rnn
import pickle
import time
from datetime import timedelta
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.classification import classification_report
import pandas as pd
from pandas import Series,DataFrame


class ARNN:
    def __init__(self,embedding, simple_embedding):
        self.learning_rate=1e-3#1e-4到了0.63
        self.batch_size=32
        self.epoch_num=1
        self.print_batch = 40
        self.total_words = embedding.shape[0]
        self.embedding = embedding
        self.simple_embedding = simple_embedding
        self.word_embedding_size=embedding.shape[1]
        self.max_sentence_len=100000
        self.max_cut_len = 40
        self.l2=0.0
        self.kp=1.0


    def build_cross_encoding_model(self,rnn_units=200,interaction_dims=16,training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            # inpput simple cut data
            self.input_q_simple = tf.placeholder(tf.int32, [None, None], name='input_q_simple')
            self.input_r_simple = tf.placeholder(tf.int32, [None, None], name='input_r_simple')
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.simple_embedding_ph = tf.placeholder(tf.float32,
                                                      shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                      word_embedding_size), dtype=tf.float32,
                                          trainable=True)#我发现这个任务embedding设为trainable很重要
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_r)
            # simple cut data word_embedding
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                              shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                              dtype=tf.float32, trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
        with tf.variable_scope('first_encodeing'):
            sentence_GRU=rnn.GRUCell(rnn_units,kernel_initializer=tf.orthogonal_initializer(),
                                     name='sentence_GRU')
            sentence_GRU=tf.nn.rnn_cell.DropoutWrapper(sentence_GRU,output_keep_prob=self.keep_prob)
            q_gru, q_last_state = tf.nn.dynamic_rnn(sentence_GRU, q_embedding,
                                                                     sequence_length=self.q_sequence_len,
                                                                     dtype=tf.float32)
            r_gru, r_last_state = tf.nn.dynamic_rnn(sentence_GRU, r_embedding,
                                                  sequence_length=self.r_sequence_len,
                                                  dtype=tf.float32)
            #simple cut data first_encoding
            simple_sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                       name='simple_sentence_GRU')
            simple_sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_sentence_GRU, output_keep_prob=self.keep_prob)
            simple_q_gru, simple_q_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_q_embedding,
                                                  sequence_length=self.simple_q_sequence_len,
                                                  dtype=tf.float32)
            simple_r_gru, simple_r_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_r_embedding,
                                                  sequence_length=self.simple_r_sequence_len,
                                                  dtype=tf.float32)
        with tf.variable_scope('attention_and_interaction'):
            attention_weight = tf.get_variable(name='attention_weight', shape=(rnn_units, rnn_units),
                                               dtype=tf.float32, initializer=xavier_initializer())
            interaction_weight = tf.get_variable(name='interaction_weight',shape=(rnn_units,interaction_dims,rnn_units))
            #use r embedding and q gru
            left=tf.tensordot(r_embedding,attention_weight,axes=(2,0))
            right=tf.transpose(q_gru,perm=(0,2,1))
            reqg_attention=tf.matmul(left,right)
            reqg_attention=tf.nn.softmax(reqg_attention)
            reqg_q_vector=tf.matmul(reqg_attention,q_gru)
            ileft=tf.tensordot(r_embedding,interaction_weight,axes=(2,0))
            ileft=tf.expand_dims(ileft,axis=3)
            iright=tf.stack([reqg_q_vector]*interaction_dims,axis=2)
            iright=tf.expand_dims(iright,axis=4)
            reqg_interaction=tf.matmul(ileft,iright)[:,:,:,-1,-1]
            #use q embedding and r gru
            left = tf.tensordot(q_embedding, attention_weight, axes=(2, 0))
            right = tf.transpose(r_gru, perm=(0, 2, 1))
            qerg_attention = tf.matmul(left, right)
            qerg_attention = tf.nn.softmax(qerg_attention)
            qerg_r_vector = tf.matmul(qerg_attention, r_gru)
            ileft = tf.tensordot(q_embedding, interaction_weight, axes=(2, 0))
            ileft = tf.expand_dims(ileft, axis=3)
            iright = tf.stack([qerg_r_vector] * interaction_dims, axis=2)
            iright = tf.expand_dims(iright, axis=4)
            qerg_interaction = tf.matmul(ileft, iright)[:, :, :, -1, -1]
            #simple cut data attention_and_interaction
            simple_attention_weight = tf.get_variable(name='simple_attention_weight', shape=(rnn_units, rnn_units),
                                               dtype=tf.float32, initializer=xavier_initializer())
            simple_interaction_weight = tf.get_variable(name='simple_interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            simple_left = tf.tensordot(simple_r_embedding, simple_attention_weight, axes=(2, 0))
            simple_right = tf.transpose(simple_q_gru, perm=(0, 2, 1))
            simple_reqg_attention = tf.matmul(simple_left, simple_right)
            simple_reqg_attention = tf.nn.softmax(simple_reqg_attention)
            simple_reqg_q_vector = tf.matmul(simple_reqg_attention, simple_q_gru)
            simple_ileft = tf.tensordot(simple_r_embedding, simple_interaction_weight, axes=(2, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=3)
            simple_iright = tf.stack([simple_reqg_q_vector] * interaction_dims, axis=2)
            simple_iright = tf.expand_dims(simple_iright, axis=4)
            simple_reqg_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, :, -1, -1]
            simple_left = tf.tensordot(simple_q_embedding, simple_attention_weight, axes=(2, 0))
            simple_right = tf.transpose(simple_r_gru, perm=(0, 2, 1))
            simple_qerg_attention = tf.matmul(simple_left, simple_right)
            simple_qerg_attention = tf.nn.softmax(simple_qerg_attention)
            simple_qerg_r_vector = tf.matmul(simple_qerg_attention, simple_r_gru)
            simple_ileft = tf.tensordot(simple_q_embedding, simple_interaction_weight, axes=(2, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=3)
            simple_iright = tf.stack([simple_qerg_r_vector] * interaction_dims, axis=2)
            simple_iright = tf.expand_dims(simple_iright, axis=4)
            simple_qerg_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, :, -1, -1]
        with tf.variable_scope('interaction_accumulation'):
            accumulation_GRU = rnn.GRUCell(interaction_dims, kernel_initializer=tf.orthogonal_initializer(),
                                           name='accumulation_GRU')
            accumulation_GRU = tf.nn.rnn_cell.DropoutWrapper(accumulation_GRU, output_keep_prob=self.keep_prob)
            output, re_accu = tf.nn.dynamic_rnn(accumulation_GRU, reqg_interaction,
                                                                sequence_length=self.r_sequence_len, dtype=tf.float32, )
            output, qe_accu = tf.nn.dynamic_rnn(accumulation_GRU, qerg_interaction,
                                                sequence_length=self.q_sequence_len, dtype=tf.float32, )
            final_vector = tf.concat([re_accu,qe_accu],axis=1)
            #simple cut data interaction_accumulation
            simple_accumulation_GRU = rnn.GRUCell(interaction_dims, kernel_initializer=tf.orthogonal_initializer(),
                                           name='simple_accumulation_GRU')
            simple_accumulation_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_accumulation_GRU, output_keep_prob=self.keep_prob)
            simple_output, simple_re_accu = tf.nn.dynamic_rnn(simple_accumulation_GRU, simple_reqg_interaction,
                                                sequence_length=self.simple_r_sequence_len, dtype=tf.float32, )
            simple_output, simple_qe_accu = tf.nn.dynamic_rnn(simple_accumulation_GRU, simple_qerg_interaction,
                                                sequence_length=self.simple_q_sequence_len, dtype=tf.float32, )
            simple_final_vector = tf.concat([simple_re_accu, simple_qe_accu], axis=1)
            #self.final_matching_vector = tf.concat([final_vector, simple_final_vector], axis=1)
        with tf.variable_scope('output'):
            # change part begin
            wsub = tf.abs(tf.subtract(q_last_state, r_last_state))
            wmul = tf.matmul(tf.expand_dims(q_last_state, axis=1), tf.expand_dims(r_last_state, axis=2))[:, :, -1]
            wmax = tf.maximum(tf.matmul(tf.expand_dims(q_last_state, axis=1), tf.expand_dims(q_last_state, axis=2))[:, :, -1],
                              tf.matmul(tf.expand_dims(r_last_state, axis=1), tf.expand_dims(r_last_state, axis=2))[:, :, -1])
            wfeature = tf.concat((wsub, wmul, wmax), axis=1)

            csub = tf.abs(tf.subtract(simple_q_last_state, simple_r_last_state))
            cmul = tf.matmul(tf.expand_dims(simple_q_last_state, axis=1), tf.expand_dims(simple_r_last_state, axis=2))[:, :, -1]
            cmax = tf.maximum(tf.matmul(tf.expand_dims(simple_q_last_state, axis=1), tf.expand_dims(simple_q_last_state, axis=2))[:, :, -1],
                              tf.matmul(tf.expand_dims(simple_r_last_state, axis=1), tf.expand_dims(simple_r_last_state, axis=2))[:, :, -1])
            cfeature = tf.concat((csub, cmul, cmax), axis=1)
            self.final_matching_vector = tf.concat((final_vector, simple_final_vector, wfeature, cfeature), axis=1)
            self.final_matching_vector = tf.concat(
                (tf.layers.dense(self.final_matching_vector, 24, activation=tf.nn.relu),
                 tf.layers.dense(self.final_matching_vector, 16, activation=tf.nn.sigmoid)),
                axis=1
            )
            logits = tf.layers.dense(self.final_matching_vector, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     name='output',
                                     activation=tf.nn.sigmoid)
            # change part end
            # logits = tf.layers.dense(self.final_matching_vector, 2,
            #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
            #                          bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
            #                          name='output')
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
            while os.path.exists('./charCross' + str(i)):
                i += 1
            os.makedirs('./charCross' + str(i))
            return './charCross' + str(i)


    def build_direct_encoding_model(self,rnn_units=200,interaction_dims=16,training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_q_simple = tf.placeholder(tf.int32, [None, None], name='input_q_simple')  # placeholder只存储一个batch的数据
            self.input_r_simple = tf.placeholder(tf.int32, [None, None], name='input_r_simple')  # placeholder只存储一个batch的数据
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.simple_embedding_ph = tf.placeholder(tf.float32,
                                                      shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                      word_embedding_size), dtype=tf.float32,
                                          trainable=True)#我发现这个任务embedding设为trainable很重要#或许前面我想错了
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_r)
            # simple cut data word_embedding
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                                     shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                                     dtype=tf.float32, trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
        with tf.variable_scope('attention_and_interaction'):
            attention_weight = tf.get_variable(name='attention_weight', shape=(rnn_units, rnn_units),
                                               dtype=tf.float32, initializer=xavier_initializer())
            interaction_weight = tf.get_variable(name='interaction_weight',shape=(rnn_units,interaction_dims,rnn_units))
            #use r embedding and q gru
            left=tf.tensordot(r_embedding,attention_weight,axes=(2,0))
            right=tf.transpose(q_embedding,perm=(0,2,1))
            r2q_attention=tf.matmul(left,right)
            r2q_attention=tf.nn.softmax(r2q_attention)
            r2q_vector=tf.matmul(r2q_attention,q_embedding)
            ileft=tf.tensordot(r_embedding,interaction_weight,axes=(2,0))
            ileft=tf.expand_dims(ileft,axis=3)
            iright=tf.stack([r2q_vector]*interaction_dims,axis=2)
            iright=tf.expand_dims(iright,axis=4)
            r2q_interaction=tf.matmul(ileft,iright)[:,:,:,-1,-1]
            #use q embedding and r gru
            q2r_attention = tf.transpose(r2q_attention,perm=(0,2,1))
            q2r_vector = tf.matmul(q2r_attention, r_embedding)
            ileft = tf.tensordot(q_embedding, interaction_weight, axes=(2, 0))
            ileft = tf.expand_dims(ileft, axis=3)
            iright = tf.stack([q2r_vector] * interaction_dims, axis=2)
            iright = tf.expand_dims(iright, axis=4)
            q2r_interaction = tf.matmul(ileft, iright)[:, :, :, -1, -1]
            #
            r2q_interaction=tf.nn.leaky_relu(tf.negative(r2q_interaction))
            q2r_interaction=tf.nn.leaky_relu(tf.negative(q2r_interaction))
            # simple cut data attention_interaction
            simple_attention_weight = tf.get_variable(name='simple_attention_weight', shape=(rnn_units, rnn_units),
                                               dtype=tf.float32, initializer=xavier_initializer())
            simple_interaction_weight = tf.get_variable(name='simple_interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            simple_left = tf.tensordot(simple_r_embedding, simple_attention_weight, axes=(2, 0))
            simple_right = tf.transpose(simple_q_embedding, perm=(0, 2, 1))
            simple_r2q_attention = tf.matmul(simple_left, simple_right)
            simple_r2q_attention = tf.nn.softmax(simple_r2q_attention)
            simple_r2q_vector = tf.matmul(simple_r2q_attention, simple_q_embedding)
            simple_ileft = tf.tensordot(simple_r_embedding, simple_interaction_weight, axes=(2, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=3)
            simple_iright = tf.stack([simple_r2q_vector] * interaction_dims, axis=2)
            simple_iright = tf.expand_dims(simple_iright, axis=4)
            simple_r2q_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, :, -1, -1]
            simple_q2r_attention = tf.transpose(simple_r2q_attention, perm=(0, 2, 1))
            simple_q2r_vector = tf.matmul(simple_q2r_attention, simple_r_embedding)
            simple_ileft = tf.tensordot(simple_q_embedding, simple_interaction_weight, axes=(2, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=3)
            simple_iright = tf.stack([simple_q2r_vector] * interaction_dims, axis=2)
            simple_iright = tf.expand_dims(simple_iright, axis=4)
            simple_q2r_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, :, -1, -1]
            #
            simple_r2q_interaction = tf.nn.leaky_relu(tf.negative(simple_r2q_interaction))
            simple_q2r_interaction = tf.nn.leaky_relu(tf.negative(simple_q2r_interaction))
        with tf.variable_scope('interaction_accumulation'):
            accumulation_GRU = rnn.GRUCell(interaction_dims, kernel_initializer=tf.orthogonal_initializer(),
                                           name='accumulation_GRU')
            accumulation_GRU = tf.nn.rnn_cell.DropoutWrapper(accumulation_GRU, output_keep_prob=self.keep_prob)
            output, re_accu = tf.nn.dynamic_rnn(accumulation_GRU, r2q_interaction,
                                                sequence_length=self.r_sequence_len, dtype=tf.float32)
            output, qe_accu = tf.nn.dynamic_rnn(accumulation_GRU, q2r_interaction,
                                                sequence_length=self.q_sequence_len, dtype=tf.float32)
            final_vector=tf.concat([re_accu,qe_accu],axis=1)
            # simple cut data interaction_accumulation
            simple_accumulation_GRU = rnn.GRUCell(interaction_dims, kernel_initializer=tf.orthogonal_initializer(),
                                           name='simple_accumulation_GRU')
            simple_accumulation_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_accumulation_GRU, output_keep_prob=self.keep_prob)
            simple_output, simple_re_accu = tf.nn.dynamic_rnn(simple_accumulation_GRU, simple_r2q_interaction,
                                                sequence_length=self.simple_r_sequence_len, dtype=tf.float32)
            simple_output, simple_qe_accu = tf.nn.dynamic_rnn(simple_accumulation_GRU, simple_q2r_interaction,
                                                sequence_length=self.simple_q_sequence_len, dtype=tf.float32)
            simple_final_vector = tf.concat([simple_re_accu, simple_qe_accu], axis=1)
            self.final_matching_vector = tf.concat([final_vector, simple_final_vector], axis=1)
        with tf.variable_scope('output'):
            hidden = tf.layers.dense(self.final_matching_vector, interaction_dims*2 ,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     activation=tf.nn.tanh,
                                     name='hidden')
            logits = tf.layers.dense(hidden, 2,
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
            while os.path.exists('./charDirect' + str(i)):
                i += 1
            os.makedirs('./charDirect' + str(i))
            return './charDirect' + str(i)


    def build_slice_encoding_model(self,rnn_units=200,interaction_dims=16,training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_q_simple = tf.placeholder(tf.int32, [None, None],
                                                 name='input_q_simple')  # placeholder只存储一个batch的数据
            self.input_r_simple = tf.placeholder(tf.int32, [None, None],
                                                 name='input_r_simple')  # placeholder只存储一个batch的数据
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.simple_embedding_ph = tf.placeholder(tf.float32,
                                               shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                      word_embedding_size), dtype=tf.float32,
                                          trainable=True)#我发现这个任务embedding设为trainable很重要
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_r)
            # simple cut data word_embedding
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                              shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                              dtype=tf.float32,trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
        with tf.variable_scope('first_encodeing'):
            sentence_GRU=rnn.GRUCell(rnn_units,kernel_initializer=tf.orthogonal_initializer(),
                                     name='sentence_GRU')
            sentence_GRU=tf.nn.rnn_cell.DropoutWrapper(sentence_GRU,output_keep_prob=self.keep_prob)
            q_gru, last_state = tf.nn.dynamic_rnn(sentence_GRU, q_embedding,
                                                                     sequence_length=self.q_sequence_len,
                                                                     dtype=tf.float32)
            r_gru, last_state = tf.nn.dynamic_rnn(sentence_GRU, r_embedding,
                                                  sequence_length=self.r_sequence_len,
                                                  dtype=tf.float32)
            # simple cut data first_encoding
            simple_sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                              name='simple_sentence_GRU')
            simple_sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_sentence_GRU, output_keep_prob=self.keep_prob)
            simple_q_gru, simple_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_q_embedding,
                                                  sequence_length=self.simple_q_sequence_len,
                                                  dtype=tf.float32)
            simple_r_gru, simple_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_r_embedding,
                                                  sequence_length=self.simple_r_sequence_len,
                                                  dtype=tf.float32)
        with tf.variable_scope('attention_and_interaction'):
            attention_weight = tf.get_variable(name='attention_weight', shape=(rnn_units, rnn_units),
                                               dtype=tf.float32, initializer=xavier_initializer())
            interaction_weight = tf.get_variable(name='interaction_weight',shape=(rnn_units,interaction_dims,rnn_units))
            #use r embedding and q gru
            left=tf.tensordot(r_gru,attention_weight,axes=(2,0))
            right=tf.transpose(q_gru,perm=(0,2,1))
            r2q_attention=tf.matmul(left,right)
            r2q_attention=tf.nn.softmax(r2q_attention)
            r2q_vector=tf.matmul(r2q_attention,q_gru)
            ileft=tf.tensordot(r_gru,interaction_weight,axes=(2,0))
            ileft=tf.expand_dims(ileft,axis=3)
            iright=tf.stack([r2q_vector]*interaction_dims,axis=2)
            iright=tf.expand_dims(iright,axis=4)
            r2q_interaction=tf.matmul(ileft,iright)[:,:,:,-1,-1]
            #use q embedding and r gru
            q2r_attention = tf.transpose(r2q_attention,perm=(0,2,1))
            q2r_vector = tf.matmul(q2r_attention, r_gru)
            ileft = tf.tensordot(q_gru, interaction_weight, axes=(2, 0))
            ileft = tf.expand_dims(ileft, axis=3)
            iright = tf.stack([q2r_vector] * interaction_dims, axis=2)
            iright = tf.expand_dims(iright, axis=4)
            q2r_interaction = tf.matmul(ileft, iright)[:, :, :, -1, -1]
            # simple cut data attention_and_interaction
            simple_attention_weight = tf.get_variable(name='simple_attention_weight', shape=(rnn_units, rnn_units),
                                               dtype=tf.float32, initializer=xavier_initializer())
            simple_interaction_weight = tf.get_variable(name='simple_interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            simple_left = tf.tensordot(simple_r_gru, simple_attention_weight, axes=(2, 0))
            simple_right = tf.transpose(simple_q_gru, perm=(0, 2, 1))
            simple_r2q_attention = tf.matmul(simple_left, simple_right)
            simple_r2q_attention = tf.nn.softmax(simple_r2q_attention)
            simple_r2q_vector = tf.matmul(simple_r2q_attention, simple_q_gru)
            simple_ileft = tf.tensordot(simple_r_gru, simple_interaction_weight, axes=(2, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=3)
            simple_iright = tf.stack([simple_r2q_vector] * interaction_dims, axis=2)
            simple_iright = tf.expand_dims(simple_iright, axis=4)
            simple_r2q_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, :, -1, -1]
            # use q embedding and r gru
            simple_q2r_attention = tf.transpose(simple_r2q_attention, perm=(0, 2, 1))
            simple_q2r_vector = tf.matmul(simple_q2r_attention, simple_r_gru)
            simple_ileft = tf.tensordot(simple_q_gru, simple_interaction_weight, axes=(2, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=3)
            simple_iright = tf.stack([simple_q2r_vector] * interaction_dims, axis=2)
            simple_iright = tf.expand_dims(simple_iright, axis=4)
            simple_q2r_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, :, -1, -1]
        with tf.variable_scope('interaction_accumulation'):
            accumulation_GRU = rnn.GRUCell(interaction_dims, kernel_initializer=tf.orthogonal_initializer(),
                                           name='accumulation_GRU')
            accumulation_GRU = tf.nn.rnn_cell.DropoutWrapper(accumulation_GRU, output_keep_prob=self.keep_prob)
            output, re_accu = tf.nn.dynamic_rnn(accumulation_GRU, r2q_interaction,
                                                                sequence_length=self.r_sequence_len, dtype=tf.float32, )
            output, qe_accu = tf.nn.dynamic_rnn(accumulation_GRU, q2r_interaction,
                                                sequence_length=self.q_sequence_len, dtype=tf.float32, )
            final_vector=tf.concat([re_accu,qe_accu],axis=1)
            # simple cut data interaction_accumulation
            simple_accumulation_GRU = rnn.GRUCell(interaction_dims, kernel_initializer=tf.orthogonal_initializer(),
                                                  name='simple_accumulation_GRU')
            simple_accumulation_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_accumulation_GRU, output_keep_prob=self.keep_prob)
            simple_output, simple_re_accu = tf.nn.dynamic_rnn(simple_accumulation_GRU, simple_r2q_interaction,
                                                sequence_length=self.simple_r_sequence_len, dtype=tf.float32, )
            simple_output, simple_qe_accu = tf.nn.dynamic_rnn(simple_accumulation_GRU, simple_q2r_interaction,
                                                sequence_length=self.simple_q_sequence_len, dtype=tf.float32, )
            simple_final_vector = tf.concat([simple_re_accu, simple_qe_accu], axis=1)
            self.final_matching_vector = tf.concat([final_vector, simple_final_vector], axis=1)
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
            while os.path.exists('./charSlice' + str(i)):
                i += 1
            os.makedirs('./charSlice' + str(i))
            return './charSlice' + str(i)


    def build_recommended_model(self, rnn_units=200, training=True):
        with tf.variable_scope('common'):
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('char_level'):
            self.input_q_simple = tf.placeholder(tf.int32, [None, None], name='input_q_simple')
            self.input_r_simple = tf.placeholder(tf.int32, [None, None], name='input_r_simple')
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.simple_embedding_ph = tf.placeholder(tf.float32, shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                                     shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                                     dtype=tf.float32, trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
            # simple_sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
            #                                   name='chargru')
            # simple_sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_sentence_GRU, output_keep_prob=self.keep_prob)
            # simple_q_gru, simple_q_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_q_embedding,
            #                                                       sequence_length=self.simple_q_sequence_len,
            #                                                       dtype=tf.float32)
            # simple_r_gru, simple_r_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_r_embedding,
            #                                                       sequence_length=self.simple_r_sequence_len,
            #                                                       dtype=tf.float32)
            # change part begin
            simple_GRU_fw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                 name='simple_forwardCell')
            simple_GRU_fw = tf.nn.rnn_cell.DropoutWrapper(simple_GRU_fw, output_keep_prob=self.keep_prob)
            simple_GRU_bw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                 name='simple_backwordCell')
            simple_GRU_bw = tf.nn.rnn_cell.DropoutWrapper(simple_GRU_bw, output_keep_prob=self.keep_prob)
            simple_q_gru, simple_q_last_state = tf.nn.bidirectional_dynamic_rnn(simple_GRU_fw, simple_GRU_bw, simple_q_embedding,
                                                                  sequence_length=self.simple_q_sequence_len,
                                                                  dtype=tf.float32)
            simple_r_gru, simple_r_last_state = tf.nn.bidirectional_dynamic_rnn(simple_GRU_fw, simple_GRU_bw, simple_r_embedding,
                                                                  sequence_length=self.simple_r_sequence_len,
                                                                  dtype=tf.float32)
            simple_q_last_state = tf.concat(simple_q_last_state, axis=1)
            simple_r_last_state = tf.concat(simple_r_last_state, axis=1)
            # change part end
            char_key_weight = tf.get_variable(name='char_key_weight', shape=(rnn_units, 10))
            char_q_key = tf.nn.tanh(tf.tensordot(simple_q_embedding, char_key_weight, axes=(2, 0))) #b w 10
            char_attention_weight = tf.get_variable(name='chat_att_w', shape=(10, 1))
            char_att_q_w = tf.tensordot(char_q_key, char_attention_weight, axes=(2, 0))  # b*w*1
            char_att_q = tf.matmul(tf.transpose(char_att_q_w, perm=(0, 2, 1)), simple_q_embedding)[:, -1, :]  # b*e
            # char att r:
            char_r_key = tf.nn.tanh(tf.tensordot(simple_r_embedding, char_key_weight, axes=(2, 0)))
            char_att_r_w = tf.tensordot(char_r_key, char_attention_weight, axes=(2, 0))  # b*w*1
            char_att_r = tf.matmul(tf.transpose(char_att_r_w, perm=(0, 2, 1)), simple_r_embedding)[:, -1, :]  # b*e
            v1c = tf.concat((simple_q_last_state, char_att_q), axis=1)
            v2c = tf.concat((simple_r_last_state, char_att_r), axis=1)
            # matching features:
            csub = tf.abs(tf.subtract(v1c, v2c))
            cmul = tf.matmul(tf.expand_dims(v1c, axis=1), tf.expand_dims(v2c, axis=2))[:, :, -1]
            cmax = tf.maximum(tf.matmul(tf.expand_dims(v1c, axis=1), tf.expand_dims(v1c, axis=2))[:, :, -1],
                              tf.matmul(tf.expand_dims(v2c, axis=1), tf.expand_dims(v2c, axis=2))[:, :, -1])
            cfeature = tf.concat((csub, cmul, cmax), axis=1)
        with tf.variable_scope('word_level'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.embedding.shape[0], self.embedding.shape[1]))
            word_embeddings = tf.get_variable('word_embeddings',
                                              shape=(
                                                  self.embedding.shape[0], self.embedding.shape[1]),
                                              dtype=tf.float32, trainable=True)
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_r)

            # sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
            #                            name='wordgru')
            # sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(sentence_GRU, output_keep_prob=self.keep_prob)
            # q_gru, q_last_state = tf.nn.dynamic_rnn(sentence_GRU, q_embedding,
            #                                         sequence_length=self.q_sequence_len,
            #                                         dtype=tf.float32)
            # r_gru, r_last_state = tf.nn.dynamic_rnn(sentence_GRU, r_embedding,
            #                                         sequence_length=self.r_sequence_len,
            #                                         dtype=tf.float32)
            # change part start
            GRU_fw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                 name='forwardCell')
            GRU_fw = tf.nn.rnn_cell.DropoutWrapper(GRU_fw, output_keep_prob=self.keep_prob)
            GRU_bw = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                 name='backwordCell')
            GRU_bw = tf.nn.rnn_cell.DropoutWrapper(GRU_bw, output_keep_prob=self.keep_prob)
            q_gru, q_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, q_embedding,
                                                                  sequence_length=self.q_sequence_len,
                                                                  dtype=tf.float32)
            r_gru, r_last_state = tf.nn.bidirectional_dynamic_rnn(GRU_fw, GRU_bw, r_embedding,
                                                                  sequence_length=self.r_sequence_len,
                                                                  dtype=tf.float32)
            q_last_state = tf.concat(q_last_state, axis=1)
            r_last_state = tf.concat(r_last_state, axis=1)
            # change part end
            # char attr q
            word_key_weight = tf.get_variable(name='word_key_weight', shape=(rnn_units, 10))
            word_q_key = tf.nn.tanh(tf.tensordot(q_embedding, word_key_weight, axes=(2, 0)))
            word_attention_weight = tf.get_variable(name='word_att_w', shape=(10, 1))
            word_att_q_w = tf.tensordot(word_q_key, word_attention_weight, axes=(2, 0))  # b*w*1 b 1 w b w
            word_att_q = tf.matmul(tf.transpose(word_att_q_w, perm=(0, 2, 1)), q_embedding)[:, -1, :]  # b*e
            # char att r:
            word_r_key = tf.nn.tanh(tf.tensordot(r_embedding, word_key_weight, axes=(2, 0)))
            word_att_r_w = tf.tensordot(word_r_key, word_attention_weight, axes=(2, 0))  # b*w*1
            word_att_r = tf.matmul(tf.transpose(word_att_r_w, perm=(0, 2, 1)), r_embedding)[:, -1, :]  # b*e
            v1 = tf.concat((q_last_state, word_att_q), axis=1)
            v2 = tf.concat((r_last_state, word_att_r), axis=1)
            # matching features:
            wsub = tf.abs(tf.subtract(v1, v2))
            wmul = tf.matmul(tf.expand_dims(v1, axis=1), tf.expand_dims(v2, axis=2))[:, :, -1]
            wmax = tf.maximum(tf.matmul(tf.expand_dims(v1, axis=1), tf.expand_dims(v1, axis=2))[:, :, -1],
                              tf.matmul(tf.expand_dims(v2, axis=1), tf.expand_dims(v2, axis=2))[:, :, -1])
            wfeature = tf.concat((wsub, wmul, wmax), axis=1)
        with tf.variable_scope('output'):
            self.final_matching_vector = tf.concat((cfeature, wfeature), axis=1)
            self.final_matching_vector = tf.concat(
                (tf.layers.dense(self.final_matching_vector, 16, activation=tf.nn.relu),
                 tf.layers.dense(self.final_matching_vector, 24, activation=tf.nn.sigmoid)), axis=1)
            logits = tf.layers.dense(self.final_matching_vector, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     name='output',
                                     activation=tf.nn.sigmoid)
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
            while os.path.exists('./charRecmd' + str(i)):
                i += 1
            os.makedirs('./charRecmd' + str(i))
            return './charRecmd' + str(i)


    def build_base_model(self,rnn_units=200,interaction_dims=16,training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_q_simple = tf.placeholder(tf.int32, [None, None], name='input_q_simple')
            self.input_r_simple = tf.placeholder(tf.int32, [None, None], name='input_r_simple')
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.simple_embedding_ph = tf.placeholder(tf.float32,
                                                      shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                          word_embedding_size), dtype=tf.float32,
                                              trainable=True)  # 我发现这个任务embedding设为trainable很重要
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            # simple word_embeddings
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                                     shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                                     dtype=tf.float32, trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            q_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_r)
            # simple embedding lookup
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
        with tf.variable_scope('first_encodeing'):
            sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                       name='sentence_GRU')
            sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(sentence_GRU, output_keep_prob=self.keep_prob)
            q_gru, q_last_state = tf.nn.dynamic_rnn(sentence_GRU, q_embedding,
                                                  sequence_length=self.q_sequence_len,
                                                  dtype=tf.float32)
            r_gru, r_last_state = tf.nn.dynamic_rnn(sentence_GRU, r_embedding,
                                                  sequence_length=self.r_sequence_len,
                                                  dtype=tf.float32)
            # simple dynamic_rnn
            simple_sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                              name='simple_sentence_GRU')
            simple_sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_sentence_GRU, output_keep_prob=self.keep_prob)
            simple_q_gru, simple_q_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_q_embedding,
                                                                  sequence_length=self.simple_q_sequence_len, dtype=tf.float32)
            simple_r_gru, simple_r_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_r_embedding,
                                                                  sequence_length=self.simple_r_sequence_len, dtype=tf.float32)
        with tf.variable_scope('interaction'):
            interaction_weight = tf.get_variable(name='interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            ileft = tf.tensordot(r_last_state, interaction_weight, axes=(1, 0))
            ileft = tf.expand_dims(ileft, axis=2)
            iright = tf.stack([q_last_state] * interaction_dims, axis=1)
            iright = tf.expand_dims(iright, axis=3)
            interaction = tf.matmul(ileft, iright)[:, :, -1, -1]
            # simple interaction
            simple_interaction_weight = tf.get_variable(name='simple_interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            simple_ileft = tf.tensordot(simple_r_last_state, simple_interaction_weight, axes=(1, 0))
            simple_ileft = tf.expand_dims(simple_ileft, axis=2)
            simple_iright = tf.stack([simple_q_last_state] * interaction_dims, axis=1)
            simple_iright = tf.expand_dims(simple_iright, axis=3)
            simple_interaction = tf.matmul(simple_ileft, simple_iright)[:, :, -1, -1]
            self.final_matching_vector=tf.concat((interaction, simple_interaction), axis=1)
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
            while os.path.exists('./charBase' + str(i)):
                i += 1
            os.makedirs('./charBase' + str(i))
            return './charBase' + str(i)


    def build_interaction_cnn_model(self,rnn_units=200,interaction_dims=8,training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, None], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, None], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_q_simple = tf.placeholder(tf.int32, [None, None], name='input_q_simple')  # placeholder只存储一个batch的数据
            self.input_r_simple = tf.placeholder(tf.int32, [None, None], name='input_r_simple')  # placeholder只存储一个batch的数据
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.simple_embedding_ph = tf.placeholder(tf.float32,
                                               shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                          word_embedding_size), dtype=tf.float32,
                                              trainable=True)  # 我发现这个任务embedding设为trainable很重要#或许前面我想错了
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding = tf.nn.embedding_lookup(word_embeddings, self.input_r)
            # simple cut data word_embedding
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                                     shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                                     dtype=tf.float32, trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
        with tf.variable_scope('interaction'):
            interaction_weight = tf.get_variable(name='interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            left=tf.tensordot(q_embedding,interaction_weight,axes=(2,0))
            left=tf.transpose(left,perm=(0,2,1,3))
            right=tf.stack([r_embedding]*interaction_dims,axis=1)
            right=tf.transpose(right,perm=(0,1,3,2))
            interaction=tf.matmul(left,right)
            interaction=tf.transpose(interaction,perm=(0,2,3,1))
            interaction=tf.nn.tanh(interaction)
            #interaction=tf.layers.batch_normalization(interaction,training=training)
            # simple cut data interaction
            simple_interaction_weight = tf.get_variable(name='simple_interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units))
            simple_left = tf.tensordot(simple_q_embedding, simple_interaction_weight, axes=(2, 0))
            simple_left = tf.transpose(simple_left, perm=(0, 2, 1, 3))
            simple_right = tf.stack([simple_r_embedding] * interaction_dims, axis=1)
            simple_right = tf.transpose(simple_right, perm=(0, 1, 3, 2))
            simple_interaction = tf.matmul(simple_left, simple_right)
            simple_interaction = tf.transpose(simple_interaction, perm=(0, 2, 3, 1))
            simple_interaction = tf.nn.tanh(simple_interaction)
        with tf.variable_scope('pooling'):
            max_pool1= tf.reduce_max(interaction, axis=(1))
            max_pool2 = tf.reduce_max(interaction, axis=(2))
            min_pool1= tf.reduce_min(max_pool1,axis=(1))
            min_pool2=tf.reduce_min(max_pool2,axis=(1))
            final_vector=tf.concat([min_pool1,min_pool2],axis=1)
            # simple cut data pooling
            simple_max_pool1 = tf.reduce_max(simple_interaction, axis=(1))
            simple_max_pool2 = tf.reduce_max(simple_interaction, axis=(2))
            simple_min_pool1 = tf.reduce_min(simple_max_pool1, axis=(1))
            simple_min_pool2 = tf.reduce_min(simple_max_pool2, axis=(1))
            simple_final_vector = tf.concat([simple_min_pool1, simple_min_pool2], axis=1)
            self.final_matching_vector = tf.concat([final_vector, simple_final_vector], axis=1)
        with tf.variable_scope('output'):
            logits = tf.layers.dense(self.final_matching_vector, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     #activation=tf.nn.tanh,
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
            while os.path.exists('./charIcnn' + str(i)):
                i += 1
            os.makedirs('./charIcnn' + str(i))
            return './charIcnn' + str(i)


    def build_gru_cnn_model(self, rnn_units=200, interaction_dims=8, training=True):
        with tf.variable_scope('placeholders'):
            self.input_q = tf.placeholder(tf.int32, [None, self.max_cut_len], name='input_q')  # placeholder只存储一个batch的数据
            self.input_r = tf.placeholder(tf.int32, [None, self.max_cut_len], name='input_r')  # placeholder只存储一个batch的数据
            self.q_sequence_len = tf.placeholder(tf.int32, [None], name='q_sequence_len')
            self.r_sequence_len = tf.placeholder(tf.int32, [None], name='r_sequence_len')
            self.input_q_simple = tf.placeholder(tf.int32, [None, self.max_cut_len],
                                          name='input_q_simple')  # placeholder只存储一个batch的数据
            self.input_r_simple = tf.placeholder(tf.int32, [None, self.max_cut_len],
                                          name='input_r_simple')  # placeholder只存储一个batch的数据
            self.simple_q_sequence_len = tf.placeholder(tf.int32, [None], name='simple_q_sequence_len')
            self.simple_r_sequence_len = tf.placeholder(tf.int32, [None], name='simple_r_sequence_len')
            self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
            self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))
            self.simple_embedding_ph = tf.placeholder(tf.float32,
                                               shape=(self.simple_embedding.shape[0], self.simple_embedding.shape[1]))
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.
                                                                      word_embedding_size), dtype=tf.float32,
                                          trainable=True)#我发现这个任务embedding设为trainable很重要#或许前面我想错了
            self.embedding_init = word_embeddings.assign(self.embedding_ph)
            q_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_q)
            r_embedding=tf.nn.embedding_lookup(word_embeddings, self.input_r)
            # simple cut data word_embedding
            simple_word_embeddings = tf.get_variable('simple_word_embeddings',
                                                     shape=(
                                                     self.simple_embedding.shape[0], self.simple_embedding.shape[1]),
                                                     dtype=tf.float32, trainable=True)
            self.simple_embedding_init = simple_word_embeddings.assign(self.simple_embedding_ph)
            simple_q_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_q_simple)
            simple_r_embedding = tf.nn.embedding_lookup(simple_word_embeddings, self.input_r_simple)
        with tf.variable_scope('first_encodeing'):
            sentence_GRU=rnn.GRUCell(rnn_units,kernel_initializer=tf.orthogonal_initializer(),
                                     name='sentence_GRU')
            sentence_GRU=tf.nn.rnn_cell.DropoutWrapper(sentence_GRU,output_keep_prob=self.keep_prob)
            q_gru, last_state = tf.nn.dynamic_rnn(sentence_GRU, q_embedding,
                                                                     sequence_length=self.q_sequence_len,
                                                                     dtype=tf.float32)
            r_gru, last_state = tf.nn.dynamic_rnn(sentence_GRU, r_embedding,
                                                  sequence_length=self.r_sequence_len,
                                                  dtype=tf.float32)
            simple_sentence_GRU = rnn.GRUCell(rnn_units, kernel_initializer=tf.orthogonal_initializer(),
                                       name='simple_sentence_GRU')
            simple_sentence_GRU = tf.nn.rnn_cell.DropoutWrapper(simple_sentence_GRU, output_keep_prob=self.keep_prob)
            simple_q_gru, simple_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_q_embedding,
                                                  sequence_length=self.simple_q_sequence_len,
                                                  dtype=tf.float32)
            simple_r_gru, simple_last_state = tf.nn.dynamic_rnn(simple_sentence_GRU, simple_r_embedding,
                                                  sequence_length=self.simple_r_sequence_len,
                                                  dtype=tf.float32)
        with tf.variable_scope('interaction'):
            interaction_weight = tf.get_variable('interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units), dtype=tf.float32)
            # word_embedding of sentence and word_embedding of sentence
            left = tf.tensordot(q_embedding, interaction_weight, axes=(2, 0))
            left = tf.transpose(left, [0, 2, 1, 3])
            right = tf.stack([r_embedding] * interaction_dims, axis=1)
            right = tf.transpose(right, [0, 1, 3, 2])
            ileftright = tf.matmul(left, right)
            self.ileftright = tf.transpose(ileftright, [0, 2, 3, 1])
            #gru of sentence and gru of sentence
            gleft = tf.tensordot(q_gru, interaction_weight, axes=(2, 0))
            gleft = tf.transpose(gleft, [0, 2, 1, 3])
            gright = tf.stack([r_gru] * interaction_dims, axis=1)
            gright = tf.transpose(gright, [0, 1, 3, 2])
            igleftgright = tf.matmul(gleft, gright)
            self.igleftgright = tf.transpose(igleftgright, [0, 2, 3, 1])
            # simple cut data interaction
            simple_interaction_weight = tf.get_variable('simple_interaction_weight',
                                                 shape=(rnn_units, interaction_dims, rnn_units), dtype=tf.float32)
            simple_left = tf.tensordot(simple_q_embedding, simple_interaction_weight, axes=(2, 0))
            simple_left = tf.transpose(simple_left, [0, 2, 1, 3])
            simple_right = tf.stack([simple_r_embedding] * interaction_dims, axis=1)
            simple_right = tf.transpose(simple_right, [0, 1, 3, 2])
            simple_ileftright = tf.matmul(simple_left, simple_right)
            self.simple_ileftright = tf.transpose(simple_ileftright, [0, 2, 3, 1])
            # gru of sentence and gru of sentence
            simple_gleft = tf.tensordot(simple_q_gru, simple_interaction_weight, axes=(2, 0))
            simple_gleft = tf.transpose(simple_gleft, [0, 2, 1, 3])
            simple_gright = tf.stack([simple_r_gru] * interaction_dims, axis=1)
            simple_gright = tf.transpose(simple_gright, [0, 1, 3, 2])
            simple_igleftgright = tf.matmul(simple_gleft, simple_gright)
            self.simple_igleftgright = tf.transpose(simple_igleftgright, [0, 2, 3, 1])
        with tf.variable_scope('conv2d_and_maxpooling'):
            W_conv1 = tf.get_variable('conv_weight1',
                                     shape=(5, 5, interaction_dims, 8), dtype=tf.float32)
            W_conv2 = tf.get_variable('conv_weight2',
                                     shape=(5, 5, interaction_dims, 8), dtype=tf.float32)
            # word embedding conv
            conv_out_word = tf.nn.conv2d(self.ileftright, W_conv1,
                                    strides=[1, 1, 1, 1], padding='VALID')
            pool_out_word = tf.nn.max_pool(conv_out_word,
                                      ksize=[1, 18, 18, 1], strides=[1, 18, 18, 1], padding='VALID')
            #gru conv
            conv_out_slice = tf.nn.conv2d(self.igleftgright, W_conv2,
                                          strides=[1, 1, 1, 1], padding='VALID')
            pool_out_slice = tf.nn.max_pool(conv_out_slice,
                                            ksize=[1, 18, 18, 1], strides=[1, 18, 18, 1], padding='VALID')
            flatten_pool_out_word = flatten(pool_out_word)
            flatten_pool_out_slice = flatten(pool_out_slice)
            final_vector = tf.concat([flatten_pool_out_word, flatten_pool_out_slice], axis=1)
            # simple cut data conv2d_and_maxpooling
            simple_W_conv1 = tf.get_variable('simple_conv_weight1',
                                      shape=(5, 5, interaction_dims, 8), dtype=tf.float32)
            simple_W_conv2 = tf.get_variable('simple_conv_weight2',
                                      shape=(5, 5, interaction_dims, 8), dtype=tf.float32)
            # word embedding conv
            simple_conv_out_word = tf.nn.conv2d(self.simple_ileftright, simple_W_conv1,
                                         strides=[1, 1, 1, 1], padding='VALID')
            simple_pool_out_word = tf.nn.max_pool(simple_conv_out_word,
                                           ksize=[1, 18, 18, 1], strides=[1, 18, 18, 1], padding='VALID')
            # gru conv
            simple_conv_out_slice = tf.nn.conv2d(self.simple_igleftgright, simple_W_conv2,
                                          strides=[1, 1, 1, 1], padding='VALID')
            simple_pool_out_slice = tf.nn.max_pool(simple_conv_out_slice,
                                            ksize=[1, 18, 18, 1], strides=[1, 18, 18, 1], padding='VALID')
            simple_flatten_pool_out_word = flatten(simple_pool_out_word)
            simple_flatten_pool_out_slice = flatten(simple_pool_out_slice)
            simple_final_vector = tf.concat([simple_flatten_pool_out_word, simple_flatten_pool_out_slice], axis=1)
            self.final_matching_vector = tf.concat([final_vector, simple_final_vector], axis=1)
        with tf.variable_scope('output'):
            logits = tf.layers.dense(self.final_matching_vector, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(self.l2),
                                     #activation=tf.nn.tanh,
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
            while os.path.exists('./charGrucnn' + str(i)):
                i += 1
            os.makedirs('./charGrucnn' + str(i))
            return './charGrucnn' + str(i)


    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))


    def get_sequences_length(self,sequences, maxlen):
        sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
        return sequences_length


    def predict(self,model_path,data_q,data_r, simple_data_q, simple_data_r):
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        all_pred_label=[]
        all_pred_value=[]
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_path)
            batch_size_for_val = 1000
            low=0
            while True:
                n_sample = min(low + batch_size_for_val, len(data_q)) - low
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_q = pad_sequences(self.copy_list(data_q[low:low + n_sample]), padding='post')
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r = pad_sequences(self.copy_list(data_r[low:low + n_sample]), padding='post')
                simple_batch_q_len = self.get_sequences_length(simple_data_q[low:low + n_sample], maxlen=self.max_sentence_len)
                simple_batch_q = pad_sequences(self.copy_list(simple_data_q[low:low + n_sample]), padding='post')
                simple_batch_r_len = self.get_sequences_length(simple_data_r[low:low + n_sample], maxlen=self.max_sentence_len)
                simple_batch_r = pad_sequences(self.copy_list(simple_data_r[low:low + n_sample]), padding='post')
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.input_q_simple: np.array(simple_batch_q),
                    self.simple_q_sequence_len: np.array(simple_batch_q_len),
                    self.input_r_simple: np.array(simple_batch_r),
                    self.simple_r_sequence_len: np.array(simple_batch_r_len),
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


    def predict_cut(self, model_path, data_q, data_r, simple_data_q, simple_data_r):
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
                batch_q_len = self.get_sequences_length(data_q[low:low + n_sample],
                                                        maxlen=self.max_cut_len)
                batch_q = pad_sequences(self.copy_list(data_q[low:low + n_sample]),
                                        padding='post', maxlen=self.max_cut_len)
                batch_r_len = self.get_sequences_length(data_r[low:low + n_sample],
                                                        maxlen=self.max_cut_len)
                batch_r = pad_sequences(self.copy_list(data_r[low:low + n_sample]),
                                        padding='post', maxlen=self.max_cut_len)
                simple_batch_q_len = self.get_sequences_length(simple_data_q[low:low + n_sample],
                                                               maxlen=self.max_cut_len)
                simple_batch_q = pad_sequences(self.copy_list(simple_data_q[low:low + n_sample]),
                                               padding='post', maxlen=self.max_cut_len)
                simple_batch_r_len = self.get_sequences_length(simple_data_r[low:low + n_sample],
                                                               maxlen=self.max_cut_len)
                simple_batch_r = pad_sequences(self.copy_list(simple_data_r[low:low + n_sample]),
                                               padding='post', maxlen=self.max_cut_len)
                feed_dict = {
                    self.input_q: np.array(batch_q),
                    self.q_sequence_len: np.array(batch_q_len),
                    self.input_r: np.array(batch_r),
                    self.r_sequence_len: np.array(batch_r_len),
                    self.input_q_simple: np.array(simple_batch_q),
                    self.simple_q_sequence_len: np.array(simple_batch_q_len),
                    self.input_r_simple: np.array(simple_batch_r),
                    self.simple_r_sequence_len: np.array(simple_batch_r_len),
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


    def copy_list(self,list):
        new_list=[]
        for l in list:
            if type(l)==type([0]) or type(l)==type(np.array([0])):
                new_list.append(self.copy_list(l))
            else:
                new_list.append(l)
        return new_list


    def evaluate_val_for_train(self, sess, data, simple_data):
        val_q, val_r, val_labels = data
        simple_val_q, simple_val_r, simple_val_labels = simple_data
        all_pred_label = []
        low = 0
        batch_size_for_val=1000
        while True:
            n_sample = min(low + batch_size_for_val, len(val_labels)) - low
            batch_q_len = self.get_sequences_length(val_q[low:low + n_sample], maxlen=self.max_sentence_len)
            batch_q = pad_sequences(self.copy_list(val_q[low:low + n_sample]), padding='post')
            batch_r_len = self.get_sequences_length(val_r[low:low + n_sample], maxlen=self.max_sentence_len)
            batch_r = pad_sequences(self.copy_list(val_r[low:low + n_sample]), padding='post')
            # simple val data
            simple_batch_q_len = self.get_sequences_length(simple_val_q[low:low + n_sample], maxlen=self.max_sentence_len)
            simple_batch_q = pad_sequences(self.copy_list(simple_val_q[low:low + n_sample]), padding='post')
            simple_batch_r_len = self.get_sequences_length(simple_val_r[low:low + n_sample], maxlen=self.max_sentence_len)
            simple_batch_r = pad_sequences(self.copy_list(simple_val_r[low:low + n_sample]), padding='post')
            feed_dict = {
                self.input_q: np.array(batch_q),
                self.q_sequence_len: np.array(batch_q_len),
                self.input_r: np.array(batch_r),
                self.r_sequence_len: np.array(batch_r_len),
                self.input_q_simple: simple_batch_q,
                self.simple_q_sequence_len: simple_batch_q_len,
                self.input_r_simple: simple_batch_r,
                self.simple_r_sequence_len: simple_batch_r_len,
                self.input_y: np.array(val_labels[low:low + n_sample]),
                self.keep_prob:1.0
            }
            pred_label,loss = sess.run([self.class_label_pred,self.total_loss], feed_dict=feed_dict)
            all_pred_label.append(pred_label)
            low = low + batch_size_for_val
            if low >= len(val_labels):
                break
        all_pred_label = np.concatenate(all_pred_label, axis=0)
        precision, recall, f_score, true_sum = precision_recall_fscore_support(val_labels, all_pred_label)
        return loss, f_score[1], classification_report(val_labels, all_pred_label)


    def evaluate_val_for_train_cut(self, sess, data, simple_data):
        val_q, val_r, val_labels = data
        simple_val_q, simple_val_r, simple_val_labels = simple_data
        all_pred_label = []
        low = 0
        batch_size_for_val=1000
        while True:
            n_sample = min(low + batch_size_for_val, len(val_labels)) - low
            batch_q_len = self.get_sequences_length(val_q[low:low + n_sample], maxlen=self.max_cut_len)
            batch_q = pad_sequences(self.copy_list(val_q[low:low + n_sample]), padding='post', maxlen=self.max_cut_len)
            batch_r_len = self.get_sequences_length(val_r[low:low + n_sample], maxlen=self.max_cut_len)
            batch_r = pad_sequences(self.copy_list(val_r[low:low + n_sample]), padding='post', maxlen=self.max_cut_len)
            simple_batch_q_len = self.get_sequences_length(simple_val_q[low:low + n_sample],
                                                           maxlen=self.max_cut_len)
            simple_batch_q = pad_sequences(self.copy_list(simple_val_q[low:low + n_sample]),
                                    padding='post', maxlen=self.max_cut_len)
            simple_batch_r_len = self.get_sequences_length(simple_val_r[low:low + n_sample],
                                                    maxlen=self.max_cut_len)
            simple_batch_r = pad_sequences(self.copy_list(simple_val_r[low:low + n_sample]),
                                           padding='post', maxlen=self.max_cut_len)
            feed_dict = {
                self.input_q: np.array(batch_q),
                self.q_sequence_len: np.array(batch_q_len),
                self.input_r: np.array(batch_r),
                self.r_sequence_len: np.array(batch_r_len),
                self.input_y: np.array(val_labels[low:low + n_sample]),
                self.input_q_simple: np.array(simple_batch_q),
                self.simple_q_sequence_len: np.array(simple_batch_q_len),
                self.input_r_simple: np.array(simple_batch_r),
                self.simple_r_sequence_len: np.array(simple_batch_r_len),
                self.keep_prob:1.0
            }
            pred_label,loss = sess.run([self.class_label_pred,self.total_loss], feed_dict=feed_dict)
            all_pred_label.append(pred_label)
            low = low + batch_size_for_val
            if low >= len(val_labels):
                break
        all_pred_label = np.concatenate(all_pred_label, axis=0)
        _, _, f_score, _ = precision_recall_fscore_support(val_labels, all_pred_label)
        return loss, f_score[1], classification_report(val_labels,all_pred_label)


    def train_model_with_fixed_data(self, file_src_dict, store_path, continue_train=False,
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

                if low >= len(train_labels):  # 即low>=total conversations number
                    low = 0
                    saver.save(sess, store_path+"/model.{0}".format(epoch))
                    print('epoch={i}'.format(i=epoch), 'ended')
                    epoch += 1
            f.close()


    def train_model_with_random_sample(self, file_src_dict, store_path, continue_train=False,
                                    previous_model_path="model",negative_samples=1):
        with open(store_path+'/papr.txt','w+') as f:
            f.write('lr:'+str(self.learning_rate)+'\n')
            f.write('kp'+str(self.kp)+'\n')
            f.write('l2'+str(self.l2) + '\n')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 只分配40%的显存
        # prepare data for val:
        with open(file_src_dict['evaluate_file'], 'rb') as f:
            val_q, val_r, val_labels = pickle.load(f)
        val_data=[val_q, val_r, val_labels]
        with tf.Session(config=config) as sess:
            file=open(store_path+'/record.txt','w+')
            train_writer = tf.summary.FileWriter(store_path, sess.graph)
            # prepare data for train:
            with open(file_src_dict['train_random_file'], 'rb') as f:
                train_q, train_r = pickle.load(f)
            with open(file_src_dict['all_utterances'],'rb')  as f:
                all_utterances=pickle.load(f)
            with open(file_src_dict['train_fix_file'], 'rb') as f:
                fix_q, fix_r, fix_labels = pickle.load(f)
            all_utterances_len=self.get_sequences_length(all_utterances, maxlen=self.max_sentence_len)
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
            #modify mistake
            n_times = len(fix_q) / len(train_q) + 1
            low_fix = 0
            while epoch < self.epoch_num:
                # low means the start location of the array of data should be feed in next
                # n_samples means how many group-samples will be feed in next time
                # one group-samples means one context and its true response and some neg responses
                n_sample = min(low + self.batch_size, len(train_q)) - low
                batch_q_len = self.get_sequences_length(train_q[low:low + n_sample], maxlen=self.max_sentence_len)
                #batch_q = pad_sequences(self.copy_list(train_q[low:low + n_sample]), padding='post')
                batch_r_len = self.get_sequences_length(train_r[low:low + n_sample], maxlen=self.max_sentence_len)
                #batch_r = pad_sequences(self.copy_list(train_r[low:low + n_sample]), padding='post')
                batch_label = [1]*n_sample+[0]*n_sample*2*negative_samples
                negative_indices = [np.random.randint(0, len(all_utterances), n_sample) for _ in
                                    range(negative_samples)]  #
                # negs's shape is negative_samples*n_sample*sentence_max_len
                tmp=np.array(all_utterances)
                negs = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                tmp=np.array(all_utterances_len)
                negs_len = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                new_batch_q=train_q[low:low + n_sample]
                new_batch_q_len=self.copy_list(batch_q_len)
                new_batch_r=train_r[low:low + n_sample]
                new_batch_r_len=self.copy_list(batch_r_len)
                for neg,neg_len in zip(negs,negs_len):
                    new_batch_q+=neg
                    new_batch_q_len+=neg_len
                    new_batch_r+=train_r[low:low + n_sample]
                    new_batch_r_len+=batch_r_len
                for neg,neg_len in zip(negs,negs_len):
                    new_batch_r+=neg
                    new_batch_r_len+=neg_len
                    new_batch_q+=train_q[low:low + n_sample]
                    new_batch_q_len+=batch_q_len
                batch_fix_q_len = self.get_sequences_length(fix_q[low_fix:low_fix + n_sample * n_times],
                                                        maxlen=self.max_sentence_len)
                batch_fix_q = fix_q[low_fix:low_fix + n_sample * n_times]
                batch_fix_r_len = self.get_sequences_length(fix_r[low_fix:low_fix + n_sample * n_times],
                                                        maxlen=self.max_sentence_len)
                batch_fix_r = fix_r[low_fix:low_fix + n_sample * n_times]
                new_batch_q+=batch_fix_q
                new_batch_r+=batch_fix_r
                new_batch_q+=batch_fix_r
                new_batch_r+=batch_fix_q
                new_batch_q_len+=batch_fix_q_len
                new_batch_r_len+=batch_fix_r_len
                new_batch_q_len+=batch_fix_r_len
                new_batch_r_len+=batch_fix_q_len
                batch_label+=fix_labels[low_fix:low_fix + n_sample * n_times]
                batch_label+=fix_labels[low_fix:low_fix + n_sample * n_times]
                '''new_batch_q+=batch_fix_q
                new_batch_r+=batch_fix_q
                new_batch_q+=batch_fix_r
                new_batch_r+=batch_fix_r
                new_batch_q_len+=batch_fix_q_len
                new_batch_r_len+=batch_fix_q_len
                new_batch_q_len+=batch_fix_r_len
                new_batch_r_len+=batch_fix_r_len
                new_batch_q += train_q[low:low + n_sample]
                new_batch_r += train_q[low:low + n_sample]
                new_batch_q += train_r[low:low + n_sample]
                new_batch_r += train_r[low:low + n_sample]
                new_batch_q_len += batch_q_len
                new_batch_r_len += batch_q_len
                new_batch_q_len += batch_r_len
                new_batch_r_len += batch_r_len
                batch_label+=[1]*n_sample*4'''
                new_batch_q=pad_sequences(self.copy_list(new_batch_q), padding='post')
                new_batch_r=pad_sequences(self.copy_list(new_batch_r), padding='post')
                feed_dict = {
                    self.input_q: np.array(new_batch_q),
                    self.q_sequence_len: np.array(new_batch_q_len),
                    self.input_r: np.array(new_batch_r),
                    self.r_sequence_len: np.array(new_batch_r_len),
                    self.input_y: np.array(batch_label),
                    self.keep_prob:self.kp
                }
                #print(feed_dict[self.input_q].shape)
                #print(feed_dict[self.input_r].shape)
                #print(feed_dict[self.q_sequence_len].shape)
                #print(feed_dict[self.r_sequence_len].shape)
                #print(feed_dict[self.input_y].shape)
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                low_fix += (n_sample * n_times)
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
                if low >= len(train_q):  # 即low>=total conversations number
                    low = 0
                    saver.save(sess, store_path+"/model.{0}".format(epoch))
                    print('epoch={i}'.format(i=epoch), 'ended')
                    epoch += 1
            f.close()


    def train_model_with_random_sample_random(self, df_train , df_valid, store_path, continue_train=False,
                                    previous_model_path="model",negative_samples=1):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=20)
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99
        # prepare data for val:



        # with open(file_src_dict['evaluate_file'], 'rb') as f:
        #     val_q, val_r, val_labels = pickle.load(f)
        val_q , val_r , val_labels = get_example(df_valid, "word")
        # with open(simple_file_src_dict['evaluate_file'], 'rb') as f:
        #     simple_val_q, simple_val_r, simple_val_labels = pickle.load(f)
        simple_val_q , simple_val_r , simple_val_labels = get_example(df_valid , "char")
        val_data=[val_q, val_r, val_labels]
        simple_val_data = [simple_val_q, simple_val_r, simple_val_labels]





        with tf.Session(config=config) as sess:
            train_writer = tf.summary.FileWriter(store_path, sess.graph)
            # prepare data for train:
            # with open(file_src_dict['train_random_file'], 'rb') as f:
            #     #全1
            #     train_q, train_r = pickle.load(f)

            # with open(file_src_dict['all_utterances'],'rb')  as f:
            #     all_utterances=pickle.load(f)

            train_q, train_r, all_utterances = get_positive_example(df_train, "word")


            # with open(file_src_dict['train_fix_file'], 'rb') as f:
            #     #原始的
            #     fix_q, fix_r, fix_labels = pickle.load(f)
            fix_q, fix_r, fix_labels = get_example(df_train , "word")





            # prepare simple data for train
            # with open(simple_file_src_dict['train_random_file'], 'rb') as f:
            #     simple_train_q, simple_train_r = pickle.load(f)
            # with open(simple_file_src_dict['all_utterances'], 'rb') as f:
            #     simple_all_utterances = pickle.load(f)
            simple_train_q, simple_train_r , simple_all_utterances = get_positive_example(df_train,"char")
            # with open(simple_file_src_dict['train_fix_file'], 'rb') as f:
            #     simple_fix_q, simple_fix_r, simple_fix_labels = pickle.load(f)
            simple_fix_q, simple_fix_r, simple_fix_labels = get_example(df_train , "char")
            all_utterances_len=self.get_sequences_length(all_utterances, maxlen=self.max_sentence_len)
            simple_all_utterances_len = self.get_sequences_length(simple_all_utterances, maxlen=self.max_sentence_len)
            if continue_train is False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
                sess.run(self.simple_embedding_init, feed_dict={self.simple_embedding_ph: self.simple_embedding})
            else:
                saver.restore(sess, previous_model_path)
            low = 0
            epoch = 1
            start_time = time.time()
            sess.graph.finalize()
            best_f_score = 0.52
            n_of_batch = 1
            #best_score=100
            #modify mistake
            n_times = len(fix_q) / len(train_q) + 1
            low_fix = 0
            while epoch < self.epoch_num:
                # low means the start location of the array of data should be feed in next
                # n_samples means how many group-samples will be feed in next time
                # one group-samples means one context and its true response and some neg responses
                n_sample = min(low + self.batch_size, len(train_q)) - low
                batch_q_len = self.get_sequences_length(train_q[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_r_len = self.get_sequences_length(train_r[low:low + n_sample], maxlen=self.max_sentence_len)
                # the len of simple batch q and r
                simple_batch_q_len = self.get_sequences_length(simple_train_q[low:low + n_sample], maxlen=self.max_sentence_len)
                simple_batch_r_len = self.get_sequences_length(simple_train_r[low:low + n_sample], maxlen=self.max_sentence_len)
                batch_label = [1]*n_sample+[0]*n_sample*2*negative_samples
                negative_indices = [np.random.randint(0, len(all_utterances), n_sample) for _ in
                                    range(negative_samples)]  #
                # negs's shape is negative_samples*n_sample*sentence_max_len
                tmp=np.array(all_utterances)
                negs = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                tmp=np.array(all_utterances_len)
                negs_len = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                new_batch_q=train_q[low:low + n_sample]
                new_batch_q_len=self.copy_list(batch_q_len)
                new_batch_r=train_r[low:low + n_sample]
                new_batch_r_len=self.copy_list(batch_r_len)
                for neg,neg_len in zip(negs,negs_len):
                    new_batch_q+=neg
                    new_batch_q_len+=neg_len
                    new_batch_r+=train_r[low:low + n_sample]
                    new_batch_r_len+=batch_r_len
                for neg,neg_len in zip(negs,negs_len):
                    new_batch_r+=neg
                    new_batch_r_len+=neg_len
                    new_batch_q+=train_q[low:low + n_sample]
                    new_batch_q_len+=batch_q_len

                # simple negs's shape is negative_samples*n_sample*sentence_max_len
                tmp = np.array(simple_all_utterances)
                simple_negs = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                tmp = np.array(simple_all_utterances_len)
                simple_negs_len = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                simple_new_batch_q = simple_train_q[low:low + n_sample]
                simple_new_batch_q_len = self.copy_list(simple_batch_q_len)
                simple_new_batch_r = simple_train_r[low:low + n_sample]
                simple_new_batch_r_len = self.copy_list(simple_batch_r_len)
                for simple_neg, simple_neg_len in zip(simple_negs, simple_negs_len):
                    simple_new_batch_q += simple_neg
                    simple_new_batch_q_len += simple_neg_len
                    simple_new_batch_r += simple_train_r[low:low + n_sample]
                    simple_new_batch_r_len += simple_batch_r_len
                for simple_neg, simple_neg_len in zip(simple_negs, simple_negs_len):
                    simple_new_batch_r += simple_neg
                    simple_new_batch_r_len += simple_neg_len
                    simple_new_batch_q += simple_train_q[low:low + n_sample]
                    simple_new_batch_q_len += simple_batch_q_len

                random_fix = np.random.randint(0, len(fix_q), n_sample)
                fix_q_tmp = np.array(fix_q)
                batch_fix_q_len = self.get_sequences_length(fix_q_tmp[random_fix],
                                                        maxlen=self.max_sentence_len)
                batch_fix_q = list(fix_q_tmp[random_fix])
                fix_r_tmp = np.array(fix_r)
                batch_fix_r_len = self.get_sequences_length(fix_r_tmp[random_fix],
                                                        maxlen=self.max_sentence_len)
                batch_fix_r = list(fix_r_tmp[random_fix])
                fix_labels_tmp = np.array(fix_labels)
                batch_fix_labels = list(fix_labels_tmp[random_fix])
                new_batch_q+=batch_fix_q
                new_batch_r+=batch_fix_r
                new_batch_q+=batch_fix_r
                new_batch_r+=batch_fix_q
                new_batch_q_len+=batch_fix_q_len
                new_batch_r_len+=batch_fix_r_len
                new_batch_q_len+=batch_fix_r_len
                new_batch_r_len+=batch_fix_q_len
                batch_label+=batch_fix_labels
                batch_label+=batch_fix_labels

                # simple train fix data
                simple_fix_q_tmp = np.array(simple_fix_q)
                simple_batch_fix_q_len = self.get_sequences_length(simple_fix_q_tmp[random_fix],
                                                            maxlen=self.max_sentence_len)
                simple_batch_fix_q = list(simple_fix_q_tmp[random_fix])
                simple_fix_r_tmp = np.array(simple_fix_r)
                simple_batch_fix_r_len = self.get_sequences_length(simple_fix_r_tmp[random_fix],
                                                            maxlen=self.max_sentence_len)
                simple_batch_fix_r = list(simple_fix_r_tmp[random_fix])
                simple_new_batch_q += simple_batch_fix_q
                simple_new_batch_r += simple_batch_fix_r
                simple_new_batch_q += simple_batch_fix_r
                simple_new_batch_r += simple_batch_fix_q
                simple_new_batch_q_len += simple_batch_fix_q_len
                simple_new_batch_r_len += simple_batch_fix_r_len
                simple_new_batch_q_len += simple_batch_fix_r_len
                simple_new_batch_r_len += simple_batch_fix_q_len

                new_batch_q=pad_sequences(self.copy_list(new_batch_q), padding='post')
                new_batch_r=pad_sequences(self.copy_list(new_batch_r), padding='post')
                simple_new_batch_q = pad_sequences(self.copy_list(simple_new_batch_q), padding='post')
                simple_new_batch_r = pad_sequences(self.copy_list(simple_new_batch_r), padding='post')
                feed_dict = {
                    self.input_q: np.array(new_batch_q),
                    self.q_sequence_len: np.array(new_batch_q_len),
                    self.input_r: np.array(new_batch_r),
                    self.r_sequence_len: np.array(new_batch_r_len),
                    self.input_q_simple: np.array(simple_new_batch_q),
                    self.simple_q_sequence_len: np.array(simple_new_batch_q_len),
                    self.input_r_simple: np.array(simple_new_batch_r),
                    self.simple_r_sequence_len: np.array(simple_new_batch_r_len),
                    self.input_y: np.array(batch_label),
                    self.keep_prob:self.kp
                }
                #print(feed_dict[self.input_q].shape)
                #print(feed_dict[self.input_r].shape)
                #print(feed_dict[self.q_sequence_len].shape)
                #print(feed_dict[self.r_sequence_len].shape)
                #print(feed_dict[self.input_y].shape)
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                n_of_batch += 1
                low_fix += (n_sample * n_times)
                if low % (self.batch_size * self.print_batch) == 0:
                    time_dif = self.__get_time_dif(start_time)
                    loss, f_score, clf_report = self.evaluate_val_for_train(sess, val_data, simple_val_data)
                    if f_score >= best_f_score:
                        best_f_score = f_score
                        saver.save(sess, store_path + '/model_best.{0}.{1}'.format(epoch, n_of_batch))
                        print("*****************************")
                        print(store_path + '/model_best.{0}.{1}'.format(epoch, n_of_batch))
                        print("*****************************")
                    train_loss=sess.run(self.total_loss, feed_dict=feed_dict)
                    print("train loss:", train_loss, "; val evaluation:"
                          ,loss, clf_report, "time:", time_dif)
                if low >= len(train_q):  # 即low>=total conversations number
                    low = 0
                    n_of_batch = 1
                    print('epoch={i}'.format(i=epoch), 'ended')
                    epoch += 1



    def train_model_with_random_sample_random_cut(self, file_src_dict, simple_file_src_dict, store_path, continue_train=False,
                                    previous_model_path="model",negative_samples=1):

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 只分配40%的显存
        # prepare data for val:
        with open(file_src_dict['evaluate_file'], 'rb') as f:
            val_q, val_r, val_labels = pickle.load(f)
        with open(simple_file_src_dict['evaluate_file'], 'rb') as f:
            simple_val_q, simple_val_r, simple_val_labels = pickle.load(f)
        val_data=[val_q, val_r, val_labels]
        simple_val_data = [simple_val_q, simple_val_r, simple_val_labels]
        with tf.Session(config=config) as sess:
            file=open(store_path+'/record.txt','w+')
            train_writer = tf.summary.FileWriter(store_path, sess.graph)
            # prepare data for train:
            with open(file_src_dict['train_random_file'], 'rb') as f:
                train_q, train_r = pickle.load(f)
            with open(file_src_dict['all_utterances'],'rb')  as f:
                all_utterances=pickle.load(f)
            with open(file_src_dict['train_fix_file'], 'rb') as f:
                fix_q, fix_r, fix_labels = pickle.load(f)
            # prepare data for simple cut train
            with open(simple_file_src_dict['train_random_file'], 'rb') as f:
                simple_train_q, simple_train_r = pickle.load(f)
            with open(simple_file_src_dict['all_utterances'],'rb')  as f:
                simple_all_utterances=pickle.load(f)
            with open(simple_file_src_dict['train_fix_file'], 'rb') as f:
                simple_fix_q, simple_fix_r, simple_fix_labels = pickle.load(f)
            all_utterances_len=self.get_sequences_length(all_utterances, maxlen=self.max_cut_len)
            simple_all_utterances_len = self.get_sequences_length(simple_all_utterances, maxlen=self.max_cut_len)
            if continue_train is False:
                sess.run(init)
                sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
                sess.run(self.simple_embedding_init, feed_dict={self.simple_embedding_ph: self.simple_embedding})
            else:
                saver.restore(sess, previous_model_path)
            low = 0
            epoch = 1
            start_time = time.time()
            sess.graph.finalize()
            #best_score=100
            best_f_score = 0.52
            n_of_batch = 1
            #modify mistake
            n_times = len(fix_q) / len(train_q) + 1
            low_fix = 0
            while epoch < self.epoch_num:
                # low means the start location of the array of data should be feed in next
                # n_samples means how many group-samples will be feed in next time
                # one group-samples means one context and its true response and some neg responses
                n_sample = min(low + self.batch_size, len(train_q)) - low
                batch_q_len = self.get_sequences_length(train_q[low:low + n_sample], maxlen=self.max_cut_len)
                #batch_q = pad_sequences(self.copy_list(train_q[low:low + n_sample]), padding='post')
                batch_r_len = self.get_sequences_length(train_r[low:low + n_sample], maxlen=self.max_cut_len)
                #batch_r = pad_sequences(self.copy_list(train_r[low:low + n_sample]), padding='post')
                simple_batch_q_len = self.get_sequences_length(simple_train_q[low:low + n_sample],
                                                               maxlen=self.max_cut_len)
                simple_batch_r_len = self.get_sequences_length(simple_train_r[low : low + n_sample],
                                                               maxlen=self.max_cut_len)
                batch_label = [1]*n_sample+[0]*n_sample*2*negative_samples
                negative_indices = [np.random.randint(0, len(all_utterances), n_sample) for _ in
                                    range(negative_samples)]  #
                # negs's shape is negative_samples*n_sample*sentence_max_len
                tmp=np.array(all_utterances)
                negs = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                tmp=np.array(all_utterances_len)
                negs_len = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                new_batch_q=train_q[low:low + n_sample]
                new_batch_q_len=self.copy_list(batch_q_len)
                new_batch_r=train_r[low:low + n_sample]
                new_batch_r_len=self.copy_list(batch_r_len)
                for neg,neg_len in zip(negs,negs_len):
                    new_batch_q+=neg
                    new_batch_q_len+=neg_len
                    new_batch_r+=train_r[low:low + n_sample]
                    new_batch_r_len+=batch_r_len
                for neg,neg_len in zip(negs,negs_len):
                    new_batch_r+=neg
                    new_batch_r_len+=neg_len
                    new_batch_q+=train_q[low:low + n_sample]
                    new_batch_q_len+=batch_q_len

                # simple cut data random data
                tmp = np.array(simple_all_utterances)
                simple_negs = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                tmp = np.array(simple_all_utterances_len)
                simple_negs_len = [list(tmp[negative_indices[i]]) for i in range(negative_samples)]
                simple_new_batch_q = simple_train_q[low:low + n_sample]
                simple_new_batch_q_len = self.copy_list(simple_batch_q_len)
                simple_new_batch_r = simple_train_r[low:low + n_sample]
                simple_new_batch_r_len = self.copy_list(simple_batch_r_len)
                for simple_neg, simple_neg_len in zip(simple_negs, simple_negs_len):
                    simple_new_batch_q += simple_neg
                    simple_new_batch_q_len += simple_neg_len
                    simple_new_batch_r += simple_train_r[low:low + n_sample]
                    simple_new_batch_r_len += simple_batch_r_len
                for simple_neg, simple_neg_len in zip(simple_negs, simple_negs_len):
                    simple_new_batch_r += simple_neg
                    simple_new_batch_r_len += simple_neg_len
                    simple_new_batch_q += simple_train_q[low:low + n_sample]
                    simple_new_batch_q_len += simple_batch_q_len

                random_fix = np.random.randint(0, len(fix_q), n_sample)
                fix_q_tmp = np.array(fix_q)
                batch_fix_q_len = self.get_sequences_length(fix_q_tmp[random_fix],
                                                        maxlen=self.max_cut_len)
                batch_fix_q = list(fix_q_tmp[random_fix])
                fix_r_tmp = np.array(fix_r)
                batch_fix_r_len = self.get_sequences_length(fix_r_tmp[random_fix],
                                                        maxlen=self.max_cut_len)
                batch_fix_r = list(fix_r_tmp[random_fix])
                fix_labels_tmp = np.array(fix_labels)
                batch_fix_labels = list(fix_labels_tmp[random_fix])
                new_batch_q+=batch_fix_q
                new_batch_r+=batch_fix_r
                new_batch_q+=batch_fix_r
                new_batch_r+=batch_fix_q
                new_batch_q_len+=batch_fix_q_len
                new_batch_r_len+=batch_fix_r_len
                new_batch_q_len+=batch_fix_r_len
                new_batch_r_len+=batch_fix_q_len
                batch_label+=batch_fix_labels
                batch_label+=batch_fix_labels

                # simple train fix data
                simple_fix_q_tmp = np.array(simple_fix_q)
                simple_batch_fix_q_len = self.get_sequences_length(simple_fix_q_tmp[random_fix],
                                                                   maxlen=self.max_cut_len)
                simple_batch_fix_q = list(simple_fix_q_tmp[random_fix])
                simple_fix_r_tmp = np.array(simple_fix_r)
                simple_batch_fix_r_len = self.get_sequences_length(simple_fix_r_tmp[random_fix],
                                                                   maxlen=self.max_cut_len)
                simple_batch_fix_r = list(simple_fix_r_tmp[random_fix])
                simple_new_batch_q += simple_batch_fix_q
                simple_new_batch_r += simple_batch_fix_r
                simple_new_batch_q += simple_batch_fix_r
                simple_new_batch_r += simple_batch_fix_q
                simple_new_batch_q_len += simple_batch_fix_q_len
                simple_new_batch_r_len += simple_batch_fix_r_len
                simple_new_batch_q_len += simple_batch_fix_r_len
                simple_new_batch_r_len += simple_batch_fix_q_len

                new_batch_q=pad_sequences(self.copy_list(new_batch_q), padding='post', maxlen=self.max_cut_len)
                new_batch_r=pad_sequences(self.copy_list(new_batch_r), padding='post', maxlen=self.max_cut_len)
                simple_new_batch_q = pad_sequences(self.copy_list(simple_new_batch_q), padding='post', maxlen=self.max_cut_len)
                simple_new_batch_r = pad_sequences(self.copy_list(simple_new_batch_r), padding='post', maxlen=self.max_cut_len)
                feed_dict = {
                    self.input_q: np.array(new_batch_q),
                    self.q_sequence_len: np.array(new_batch_q_len),
                    self.input_r: np.array(new_batch_r),
                    self.r_sequence_len: np.array(new_batch_r_len),
                    self.input_y: np.array(batch_label),
                    self.input_q_simple: np.array(simple_new_batch_q),
                    self.simple_q_sequence_len: np.array(simple_new_batch_q_len),
                    self.input_r_simple: np.array(simple_new_batch_r),
                    self.simple_r_sequence_len: np.array(simple_new_batch_r_len),
                    self.keep_prob:self.kp
                }
                #print(feed_dict[self.input_q].shape)
                #print(feed_dict[self.input_r].shape)
                #print(feed_dict[self.q_sequence_len].shape)
                #print(feed_dict[self.r_sequence_len].shape)
                #print(feed_dict[self.input_y].shape)
                _, summary = sess.run([self.train_op, merged], feed_dict=feed_dict)
                train_writer.add_summary(summary)
                low += n_sample
                n_of_batch += 1
                low_fix += (n_sample * n_times)
                if low % (self.batch_size * self.print_batch) == 0:
                    time_dif = self.__get_time_dif(start_time)
                    loss, f_score, clf_report=self.evaluate_val_for_train_cut(sess, val_data, simple_val_data)
                    if f_score > best_f_score:
                        best_f_score = f_score
                        saver.save(sess, store_path+'/model_best.{0}.{1}'.format(epoch, n_of_batch))
                    train_loss=sess.run(self.total_loss, feed_dict=feed_dict)
                    print("train loss:", train_loss, "; val evaluation:"
                          ,loss,clf_report, "time:", time_dif)
                    file.write("train loss:"+str(train_loss)+ "; val evaluation:"+str(loss)+'\n'+clf_report+"\ntime:"+str(time_dif)+'\n')
                if low >= len(train_q):  # 即low>=total conversations number
                    low = 0
                    n_of_batch = 1
                    print('epoch={i}'.format(i=epoch), 'ended')
                    file.write('epoch={i}'.format(i=epoch) + ' ended\n')
                    epoch += 1



def get_positive_example(df , side):
    data = df[["sent1id_"+side , "sent2id_"+side , "label"]]
    sample = data[data.label == 1] #如果label是int型的话
    sent1 = list(sample["sent1id_"+side])
    sent2 = list(sample["sent2id_"+side])
    sent1 = [[int(word.strip())] for sen in sent1 for word in sen.split('/') if len(word.strip()) > 0]
    sent2 = [[int(word.strip())] for sen in sent2 for word in sen.split('/') if len(word.strip()) > 0]
    q = df["sent1id_"+side]
    r = df["sent2id_"+side]
    all_sen = q.append(r)
    del q
    del r
    all_sen = [ [int(word.strip())] for sen in all_sen for word in sen.split('/') if len(word.strip()) > 0]
    return sent1 , sent2 , all_sen

def get_example(df , side):
    sent1 = list(df["sent1id_"+side])
    sent2 = list(df["sent2id_"+side])
    label = list(df['label'])
    sent1 = [[int(word.strip())] for sen in sent1 for word in sen.split('/') if len(word.strip()) > 0]
    sent2 = [[int(word.strip())] for sen in sent2 for word in sen.split('/') if len(word.strip()) > 0]
    return sent1 , sent2 , label





#四个输入桩df1, df2, df3, df4

#默认df3  给的词向量、df4 给的字向量、
# df3给的train数据，df4给的valid数据 , 两者都是id化了的
word_embedding =  np.array(df3.ix[:,1:])
char_embedding = np.array(df4.ix[:,1:])
arnn = ARNN(embedding=word_embedding, simple_embedding=char_embedding)
# 测试
# pos_char_q, pos_char_r ,all_char_sen = get_positive_example(df3 , "char")
# pos_word_q, pos_word_r ,all_word_sen = get_positive_example(df3 , "word")
# print(pos_char_q[0:10])
path = arnn.build_recommended_model()
print('start train')
path = model_dir + 'model'
#arnn.train_model_with_fixed_data(file_src_dict=file_src_dict,store_path=path)
#arnn.train_model_with_random_sample_random(df_train = df1 , df_valid= df2,store_path=model_dir+"model",continue_train=True,previous_model_path=model_dir + "model/model_best.1.2")

df_valid = df2
# with open(file_src_dict['evaluate_file'], 'rb') as f:
#     val_q, val_r, val_labels = pickle.load(f)
val_q , val_r , val_labels = get_example(df_valid, "word")
# with open(simple_file_src_dict['evaluate_file'], 'rb') as f:
#     simple_val_q, simple_val_r, simple_val_labels = pickle.load(f)
simple_val_q , simple_val_r , simple_val_labels = get_example(df_valid , "char")
# val_data=[val_q, val_r, val_labels]
# simple_val_data = [simple_val_q, simple_val_r, simple_val_labels]
all_pred_label,all_pred_value = arnn.predict(model_path = model_dir + "model/model_best.1.2",data_q=val_q,data_r=val_r, simple_data_q=simple_val_q, simple_data_r=simple_val_r)
df_valid.insert(len(df_valid.keys()),"yes_prob" , DataFrame(all_pred_value[:,1]) )
topai(1 , df_valid)

