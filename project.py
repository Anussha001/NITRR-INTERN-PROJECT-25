# #### Introduction
# The dataset provides a comprehensive view of user behavior on social media, capturing various attributes such as demographics, platform usage, productivity loss, and satisfaction. The analysis explores key factors influencing productivity loss, examining demographic, psychological, and behavioral attributes. By identifying critical features, we can develop targeted strategies to enhance user satisfaction and minimize negative impacts of social media use.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')
dt=pd.read_csv('C:/Users/Asus/Downloads/Time-Wasters on Social Media.csv')

dt.describe()
dt.head()
#first 5 rows
dt.tail()
#last 5 rows
dt.columns
#name of all columns
dt.shape
#(rows,columns)
dt.info()
dt.isnull()
dt.isnull().sum()
#check if somewhere in dataset some values are missing or not

# %%
plt.figure(figsize=(10,4))
sns.barplot(data=dt,x='Gender', y='ProductivityLoss',palette='Set1')
plt.title("Barplot for dataset")
plt.xlabel('Gender')
plt.ylabel('Productivity Loss')

# %%
plt.figure(figsize=(10,4))
sns.barplot(data=dt,x='Age', y='Total Time Spent',palette='Set3')
plt.title("Age v/s Total time spent")
plt.xlabel('Age')
plt.ylabel('Total Time Spent')

# %%
plt.figure(figsize=(10,4))
sns.barplot(data=dt,x='ProductivityLoss', y='Platform',palette='Set2')
plt.title("Productivity loss v/s Platforms used")
plt.xlabel('Productivity Loss')
plt.ylabel('Platform')

# %%
plt.figure(figsize=(10, 4))
sns.countplot(data=dt, x='Platform', palette='Set2')
plt.title('Frequency of Each Social Media Platform Used')
plt.xlabel('Platform')
plt.ylabel('Number of Users')

# %%
# Plot distribution of productivity loss
plt.figure(figsize=(10, 4))
sns.histplot(dt['ProductivityLoss'], kde=True, bins=35)
plt.title('Distribution of Productivity Loss')
plt.xlabel('Productivity Loss')
plt.ylabel('Frequency')
plt.show()

# * Distribution of Productivity Loss:
# The histogram shows how productivity loss is distributed among users. The distribution appears to be fairly uniform with slight peaks at certain levels.

# %%
numeric_features = ['Age', 'Income', 'Satisfaction', 'Self Control', 'Addiction Level', 'ProductivityLoss']

# %%
# Correlation matrix to identify relationships between features
plt.figure(figsize=(10, 4))
corr_matrix = dt[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# * Correlation Matrix:
# The heatmap reveals the relationships between various features. Strong correlations can help identify key factors influencing productivity loss.

# %% [markdown]
# #### Predictive Modeling

# %%
# Dropping columns with high cardinality or less relevance for simplicity
selected_features = ['Age', 'Gender', 'Income', 'Debt', 'Owns Property', 
                     'Profession', 'Demographics', 'Platform', 
                     'Watch Reason', 'DeviceType','Self Control','Addiction Level']

# Encode categorical variables
data_encoded = pd.get_dummies(dt[selected_features], drop_first=True)

# Define target variable
X = data_encoded
y = dt['ProductivityLoss']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=45)
model.fit(X_train, y_train)

# Predict on the test set
y_predict = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(mae,",",r2)

# Plot actual vs predicted productivity loss
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_predict, alpha=0.65)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Productivity Loss')
plt.ylabel('Predicted Productivity Loss')
plt.title('Actual v/s Predicted Productivity Loss')
plt.show()


# %% [markdown]
# * Model Evaluation:
# Mean Absolute Error (MAE): 0.067, R-squared (R2) Score: 0.99, The very high R-squared value indicates that the model explains almost all the variability in the productivity loss, while the low MAE suggests high accuracy in predictions.
# 
# * Actual vs Predicted Productivity Loss:The scatter plot shows a strong correlation between actual and predicted productivity loss values, indicating the model's effectiveness.

# %%
# Calculate feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Display the feature importance DataFrame
feature_importance_df

