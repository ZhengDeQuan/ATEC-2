# -*- coding:utf-8 -*-
# import pandas as pd
# from pandas import DataFrame
# data = {'pop': [1.5, 1.7, 3.6, 2.4, 2.9],'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],'year': [2000, 2001, 2002, 2001, 2002]}
# df = DataFrame(data)
# df1 = df[['state','year']]
# df2 = df['state','year']
# print(df1)
# print(df2)

import pandas as pd
from pandas import Series,DataFrame
data = {'pop': [1.5, 1.7, 3.6, 2.4, 2.9],'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],'year': [2000, 2001, 2002, 2001, 2002]}
df = DataFrame(data)
print(df)
print(len(df))
# df1 = df[['state','year']]
# print(df1)
# df2 = df[['pop']]
# print(df2)
# df1.insert(len(df1.keys()),'yes_prob',df2)
# print(df1)
# print(len(df1.keys()))
# import numpy as np
# a = np.ones(len(df1[df1.keys()[0]]))
# a = np.ones(10)
# a = DataFrame(a)
# print(" a = ",a)
# print("over")
# df1.insert(len(df1.keys()),'no_prob',a)
# print(df1)

