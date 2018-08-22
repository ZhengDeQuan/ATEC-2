# -*- coding:utf-8 -*-
# print("num_of_train_sample = ",len(df1))
# print("num_of_label = ",len(df1['label']))
# print("num_of_sent1 = ",len(df1['sent1']))
# print("num_of_sent2 = ",len(df1["sent2"]))
# print("num_of_sent1id_word = ",len(df1['sent1id_word']))
# print("num_of_sent2id_word = ",len(df1['sent2id_word']))
# print("num_of_sent1id_char = ",len(df1['sent1id_char']))
# print("num_of_sent2id_char = ",len(df1["sent2id_char"]))
#a = "101010101"
# b = a.split("0")
# print("b = ",b)
# c = list(b)
# print("c = ",c)
# c = list(map(int , b))
# print("c = ",c)
# import numpy as np
# d = np.array(c)
#
# print(d.shape)

# import os
# a = os.listdir(model_dir+"model")
# print(a)

# import tensorflow as tf
# import numpy as np
#
# a = np.array(range(2 * 3 * 4)).reshape(2,3,4)
# print("a = ",a)
# b = np.array(range(2 * 4 * 1)).reshape(2, 4 ,1)
# print("b = ",b)
# a = tf.constant(value = a, dtype=tf.float32)
# b = tf.transpose(tf.constant(value = b, dtype=tf.float32),perm=(0,2,1))
# c = a + b
# with tf.Session():
#     print("transpose_b = ",b.eval())
#     print("c = ",c.eval())


import tensorflow  as tf
import numpy as np
a = np.array(range(12)).reshape(2,3,2)
print("a = ",a)
b = np.ones((12)).reshape(2,2,3)
a = tf.constant(value = a, dtype=tf.float32)
b = tf.constant(value = b, dtype=tf.float32)
c = tf.matmul(a,b)
d = tf.reduce_mean(a,axis = 1)
with tf.Session():
    #print("transpose_b = ",b.eval())
    #print("c = ",c.eval())
    print("d = ",d.eval())

