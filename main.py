# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, recall_score

# Read and load the data
data = pd.read_csv("dataset/indian_liver_patient.csv")
print(f"Total number of samples: {data.shape[0]}. Total number of features in each sample: {data.shape[1]}.")

# Get the first five data
print(data.head())
# To get the tail info of the dataset
print(data.tail())
# Help to know if there are missing data or not
print(data.info())

print(data.describe())
print(data.shape)


# Data Preprocessing
# Check to see if there are duplicates
''' Duplicates are removed as it's most likely these entries have been inputted twice. '''
data_duplicate = data[data.duplicated(keep=False)]
print(data_duplicate)

# Keep = False gives you all rows with duplicate entries
data = data[~data.duplicated(subset=None, keep='first')]
# Here, keep = 'first' ensures that only the first row is taken into the final dataset.
# The '~' sign tells pandas to keep all values except the 13 duplicate values
print(data.shape)

# Checking if there are any NULL values in our Dataset
print(data.isnull().values.any())
# Display number of null values by column
print(data.isnull().sum())

# We can see that the column 'Albumin_and_Globulin_Ratio' has 4 missing values
# One way to deal with them can be to just directly remove these 4 values
print("Length before removing NaN values: %d" % len(data))
data_2 = data[pd.notnull(data['Albumin_and_Globulin_Ratio'])]
print("Length after removing NaN values: %d" % len(data_2))

new_data = data.dropna(axis=0, how='any')
print(new_data.isnull().values.any())

print(data.info())


'''
The Albumin-Globulin Ratio feature has four missing values, as seen above.
Here, we are dropping those particular rows which have missing data.
We could, in fact, fill those place with values of our own, using options like:
A constant value that has meaning within the domain, such as 0, distinct from all other values.
A value from another randomly selected record, or the immediately next or previous record.
A mean, median or mode value for the column.
A value estimated by another predictive model.
But here, since a very small fraction of values are missing, we choose to drop those rows.
'''

# Transform our data
le = preprocessing.LabelEncoder()
le.fit(['Male', 'Female'])
data.loc[:, 'Gender'] = le.transform(data['Gender'])

# Remove rows with missing values
data = data.dropna(how='any', axis=0)

# Also transform Selector variable into usual conventions followed
data['Dataset'] = data['Dataset'].map({2: 0, 1: 1})

# Overview of data
print(data.head())
# Features characteristics to determine if feature scaling is necessary
print(data.describe())


# Split the data into test and train samples
X = data.drop(['Dataset'], axis=1)
y = data['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Exploratory Data Analysis
# Determining the healthy-affected split
print("Positive records:", data['Dataset'].value_counts().iloc[0])
print("Negative records:", data['Dataset'].value_counts().iloc[1])

# Determine statistics based on age
plt.figure(figsize=(12, 10))
plt.hist(data[data['Dataset'] == 1]['Age'], bins=16, align='mid', rwidth=0.5, color='black', alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Frequency-Age Distribution')
plt.grid(True)
plt.savefig('fig1.png')
plt.show()


# Correlation Matrix
plt.figure(figsize=(12, 10))
plt.title('Pearson Correlation of Features')
# Draw the heatmap using seaborn
sns.heatmap(data.corr(), linewidths=0.25, vmax=1.0, square=True, annot=True)
plt.savefig('fig2.png')
plt.show()


'''
The correlation matrix gives us the relationship between two features.
As seen above, the following pairs of features seem to be very closely related:
1. Total Bilirubin and Direct Bilirubin (0.87)
2. Sgpt Alamine Aminotransferase and Sgot Aspartate Aminotransferase (0.79)
3. Albumin and Total Proteins (0.78)
4. Albumin and Albumin-Globulin Ratio (0.69)
'''


'''
# Using Classification Algorithms
Let us now evaluate the performance of various classifiers on this dataset.
For the sake of understanding as to how feature scaling affects classifier performance,
we will train models using both scaled and unscaled data.
Since we are interested in capturing records of people who have been tested positive,
we will base our classifier evaluation metric on precision and recall instead of accuracy.
We could also use F1 score, since it takes into account both precision and recall.
'''

# Logistic Regression: Using normal data
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
print("Logistic Regression Classifier on unscaled test data:")
print("Accuracy:", logreg.score(X_test, y_test))
print("Precision:", precision_score(y_test, logreg.predict(X_test)))
print("Recall:", recall_score(y_test, logreg.predict(X_test)))
print("F-1 score:", f1_score(y_test, logreg.predict(X_test)))

# Using feature-scaled data
logreg_scaled = LogisticRegression(C=0.1).fit(X_train_scaled, y_train)
print("Logistic Regression Classifier on scaled test data:")
print("Accuracy:", logreg_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, logreg_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, logreg_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, logreg_scaled.predict(X_test_scaled)))


'''
Well! The performance has definitely improved by feature scaling, though not drastically,
as there was already very little scope of improvement.
Let us look at other classifiers and analyse how they react to scaling.
'''

# SVM Classifier with RBF kernel: Using normal data
svc_clf = SVC(C=0.1, kernel='rbf').fit(X_train, y_train)
print("SVM Classifier on unscaled test data:")
print("Accuracy:", svc_clf.score(X_test, y_test))
print("Precision:", precision_score(y_test, svc_clf.predict(X_test)))
print("Recall:", recall_score(y_test, svc_clf.predict(X_test)))
print("F-1 score:", f1_score(y_test, svc_clf.predict(X_test)))

# Using scaled data
svc_clf_scaled = SVC(C=0.1, kernel='rbf').fit(X_train_scaled, y_train)
print("SVM Classifier on scaled test data:")
print("Accuracy:", svc_clf_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, svc_clf_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, svc_clf_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, svc_clf_scaled.predict(X_test_scaled)))

# Random Forest Classifier: using normal data
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train, y_train)
print("Random Forest Classifier on unscaled test data:")
print("Accuracy:", rfc.score(X_test, y_test))
print("Precision:", precision_score(y_test, rfc.predict(X_test)))
print("Recall:", recall_score(y_test, rfc.predict(X_test)))
print("F-1 score:", f1_score(y_test, rfc.predict(X_test)))

# Using scaled data
rfc_scaled = RandomForestClassifier(n_estimators=20)
rfc_scaled.fit(X_train_scaled, y_train)
print("Random Forest Classifier on scaled test data:")
print("Accuracy:", rfc_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, rfc_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, rfc_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, rfc_scaled.predict(X_test_scaled)))
