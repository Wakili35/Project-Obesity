import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.inspection import permutation_importance

Df = pd.read_excel('MCS_Revised.xlsx')
Obesity_raw = pd.DataFrame(Df, columns=['Mother_BMI', 'Mother_SEC', 'Father_SEC', 'Father_BMI', 'BirthWeight',
                                        'Gender', 'BreastFed', 'BMI_Age3', 'Obesity_Flag_Age3', 'BMI_Age_5',
                                        'Obesity_Flag_Age5', 'Obesity_Flag_Age7'])

Obesity = Obesity_raw.drop(['Mother_SEC', 'Father_SEC', 'Gender', 'BreastFed'], axis='columns')
print(Obesity.describe().transpose())

# standardization
z = np.abs(stats.zscore(Obesity))
'''threshold = 3
Obesity = Obesity[(z < 3).all(axis=1)]'''

# Creating a two class label for the obesity flags
Obesity.loc[Obesity['Obesity_Flag_Age3'] == 2] = 1
Obesity.loc[Obesity['Obesity_Flag_Age5'] == 2] = 1
Obesity.loc[Obesity['Obesity_Flag_Age7'] == 2] = 1

# label the data
y = Obesity['Obesity_Flag_Age7']
x = Obesity.drop(['Obesity_Flag_Age7'], axis=1)

# over sampling strategy
smote = SMOTE(sampling_strategy='not majority', random_state=42)

# splitting into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=30)
x_res, y_res = smote.fit_resample(x_train, y_train)

# random forest
RFC = RandomForestClassifier(max_depth=8, random_state=0)
y_predict = RFC.fit(x_res, y_res).predict(x_test.values)
print('Random Forest Classifier')
print(classification_report_imbalanced(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

importance = RFC.feature_importances_
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

'''shap_values = shap.TreeExplainer(RFC).shap_values(x_test)
print(shap.summary_plot(shap_values, x_test, plot_type = 'dot', feature_names = Obesity.columns))
plt.show()'''

# Building the neural net model
mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', solver='adam', max_iter=500)
mlp.fit(x_res, y_res)
predict_train = mlp.predict(x_res)
predict_test = mlp.predict(x_test)

'''scores = cross_val_score(mlp, x, y, cv=5)
print(scores)'''

# Now performance for testing data
print('Neural Net performance on testing data')
print(confusion_matrix(y_test, predict_test))
print(classification_report_imbalanced(y_test, predict_test))

# Naive Bayes
NB = GaussianNB()
predicted_y = NB.fit(x_res, y_res).predict(x_test)
print('Naive Bayes')
print((x_test.shape[0], (y_test != predicted_y).sum()))
print(classification_report_imbalanced(y_test, predicted_y))

# Logistic regression
LR = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto')
new_predicted_y = LR.fit(x_res, y_res, sample_weight=1).predict(x_test)
print('Logistic regression')
print(classification_report_imbalanced(y_test, new_predicted_y))

importance = LR.coef_[0]
for i, v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
pred_y = knn.fit(x_res, y_res).predict(x_test)
print('K Nearest Neighbors')
print(classification_report_imbalanced(y_test, pred_y))

results = permutation_importance(knn, x, y, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
for i, v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
