import numpy as n,sklearn.model_selection as s
f='ratings.csv'
T,t=s.train_test_split(n.recfromcsv(f),test_size=.1)
n.savetxt('train_'+f,T)
n.savetxt('test_'+f,t)