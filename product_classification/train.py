'''
Created on May 5, 2015

@author: vermaa8
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
from sklearn import svm
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter






output_dictionary = {'Class_1':'1,0,0,0,0,0,0,0,0','Class_2':'0,1,0,0,0,0,0,0,0','Class_3':'0,0,1,0,0,0,0,0,0','Class_4':'0,0,0,1,0,0,0,0,0','Class_5':'0,0,0,0,1,0,0,0,0','Class_6':'0,0,0,0,0,1,0,0,0','Class_7':'0,0,0,0,0,0,1,0,0','Class_8':'0,0,0,0,0,0,0,1,0','Class_9':'0,0,0,0,0,0,0,0,1'}
training_data = pd.DataFrame.from_csv('train.csv')
test_set = pd.DataFrame.from_csv('test.csv')
training_set = training_data.ix[:,:-1]
target_label = training_data.target


m = RandomForestClassifier()
m.fit(training_set,target_label)
sorted_features = sorted(zip(list(training_set.columns.values),m.feature_importances_),key=itemgetter(1),reverse=True)
selected_training_set =  training_set[sorted_features[0:50]].values

print "feature selection done"
rf_clf = RandomForestClassifier(n_estimators=50)
rf_clf.fit(training_set,target_label)
scores = cross_validation.cross_val_score(rf_clf, selected_training_set, target_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#rf_out = rf_clf.predict(test_set)


'''clf = BaggingClassifier()
bg_clf.fit(training_set,target_label)
scores = cross_validation.cross_val_score(bg_clf, training_set, target_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))'''

'''et_clf = ExtraTreesClassifier(n_estimators=50)
et_clf.fit(training_set,target_label)'''
#scores = cross_validation.cross_val_score(et_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
svm_clf = svm.LinearSVC()
svm_clf.fit(training_set, target_label)
#scores = cross_validation.cross_val_score(svm_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svm_out = svm_clf.predict(test_set)'''



'''
gd_clf = GradientBoostingClassifier(n_estimators=200)
gd_clf.fit(training_set,target_label)
scores = cross_validation.cross_val_score(gd_clf, training_set, target_label, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#gd_out = gd_clf.predict(test_set)'''

'''
ad_clf = AdaBoostClassifier()
ad_clf.fit(training_set,target_label)
#scores = cross_validation.cross_val_score(ad_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
ad_out = ad_clf.predict(test_set)


lr_clf = LogisticRegression()
lr_clf.fit(training_set,target_label)
#scores = cross_validation.cross_val_score(lr_clf, training_set, target_label, cv=10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
lr_out = lr_clf.predict(test_set)


def majority_class(inp_tup):
    data = Counter(inp_tup)
    return data.most_common(1)[0][0]


#concat_list = map(list.__add__,svm_out,rf_out,ad_out,gd_out,lr_out)
concat_list = zip(gd_out.tolist(),rf_out.tolist(),svm_out.tolist(),lr_out.tolist(),ad_out.tolist())
final_list = []
for val in concat_list:
    final_list.append(majority_class(val))


#tup = zip(map(str,test_set.index.tolist()),[output_dictionary[x] for x in rf_clf.predict(test_set[selected_features].values)])'''

'''tup = zip(map(str,test_set.index.tolist()),[output_dictionary[x] for x in gd_clf.predict(test_set)])
output = 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9' +'\n'+ '\n'.join('%s,%s' % t for t in tup)
f=open('submission.csv','w')
f.write(output)
f.close()'''
