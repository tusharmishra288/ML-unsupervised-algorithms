
import pandas as pd
#import seaborn as sm
r=pd.read_csv('bank_contacts.csv')
#p=r['credit_application'].value_counts()
#sm.countplot(p)
x=pd.DataFrame(r.iloc[:,:-1])
y=pd.DataFrame(r.iloc[:,-1])
import sklearn.model_selection as k
x_train,x_test,y_train,y_test=k.train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.naive_bayes import GaussianNB
t=GaussianNB()
pred=t.fit(x_train,y_train)
y_pred=t.predict(x_test)
print('accuracy: %d',t.score(x_test,y_test))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,y_pred))

