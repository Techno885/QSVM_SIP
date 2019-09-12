import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Dataset definition
dataset = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[dataset['data'], dataset['target']], columns=np.append(dataset['feature_names'],
                                                                                      ['target']))
sns.pairplot(df_cancer, hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                                            'mean smoothness'])
plt.show()

# Normalization
X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

# Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9896, random_state=20)

X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = (X_train_max - X_train_min)

X_train_scaled = (X_train - X_train_min) /(X_train_range)

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min) / X_test_range

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer', 'predicted_healthy'])
print(confusion)
sns.heatmap(confusion, annot=True, fmt="d")
plt.show()

print(classification_report(y_test, y_predict))

