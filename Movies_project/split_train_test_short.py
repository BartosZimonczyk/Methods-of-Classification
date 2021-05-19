import pandas as p,sklearn.model_selection as s
r='ratings.csv'
t=p.read_csv(r)
x,y=s.train_test_split(t,test_size=.1,stratify=t['userId'])
x.to_csv('train_'+r)
y.to_csv('test_'+r)