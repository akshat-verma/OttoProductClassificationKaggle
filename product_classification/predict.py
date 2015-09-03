'''
Created on May 11, 2015

@author: vermaa8
'''


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.naive_bayes import BernoulliNB



output_dictionary = {'Class_1':'1,0,0,0,0,0,0,0,0','Class_2':'0,1,0,0,0,0,0,0,0','Class_3':'0,0,1,0,0,0,0,0,0','Class_4':'0,0,0,1,0,0,0,0,0','Class_5':'0,0,0,0,1,0,0,0,0','Class_6':'0,0,0,0,0,1,0,0,0','Class_7':'0,0,0,0,0,0,1,0,0','Class_8':'0,0,0,0,0,0,0,1,0','Class_9':'0,0,0,0,0,0,0,0,1'}
training_data = pd.DataFrame.from_csv('train.csv')
test_set = pd.DataFrame.from_csv('test.csv')
training_set = training_data.ix[:,:-1]
target_label = training_data.target

'''
m = RandomForestClassifier()
m.fit(training_set,target_label)
sorted_features = sorted(zip(list(training_set.columns.values),m.feature_importances_),key=itemgetter(1),reverse=True)
selected_features = []
for feature,value in sorted_features:
    if(value > 0.01 ):
        selected_features.append(feature)
selected_training_set =  training_set[selected_features].values
'''


rf_clf = RandomForestClassifier(n_estimators=50)
rf_clf.fit(training_set,target_label)
#scores = cross_validation.cross_val_score(rf_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
rf_out = rf_clf.predict_proba(test_set).tolist()

bg_clf = BaggingClassifier(n_estimators=50)
bg_clf.fit(training_set,target_label)
#scores = cross_validation.cross_val_score(bg_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
bg_out = bg_clf.predict_proba(test_set).tolist()

et_clf = ExtraTreesClassifier(n_estimators=50)
et_clf.fit(training_set,target_label)
#scores = cross_validation.cross_val_score(et_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
et_out = et_clf.predict_proba(test_set).tolist()


gd_clf = GradientBoostingClassifier(n_estimators=100)
gd_clf.fit(training_set,target_label)
#scores = cross_validation.cross_val_score(gd_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
gd_out = gd_clf.predict_proba(test_set).tolist()

'''
ad_clf = AdaBoostClassifier(n_estimators=100)
ad_clf.fit(training_set,target_label)
scores = cross_validation.cross_val_score(ad_clf, training_set, target_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#ad_out = ad_clf.predict_proba(test_set).tolist()


lr_clf = LogisticRegression()
lr_clf.fit(training_set,target_label)
scores = cross_validation.cross_val_score(lr_clf, training_set, target_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#lr_out = lr_clf.predict_proba(test_set).tolist()
'''


final_out =[]
for rf,bg,et,gd in zip(rf_out,bg_out,et_out,gd_out):
    temp_list = []
    for val in zip(rf,bg,et,gd):
        temp_list.append("{0:.2f}".format(np.mean(val)))
    final_out.append(temp_list)
f=open('submission.csv','w')
f.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
for ids, probs in zip(map(str,test_set.index.tolist()), final_out):
            probas = ids +','+','.join(list(map(str, probs)))
            f.write(probas)
            f.write('\n')
f.close()

    