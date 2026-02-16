# =========================
# PART 1: DATA MANIPULATION
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a) Read the provided CSV file ‘data.csv’.
df = pd.read_csv("data.csv")

# c) Show the basic statistical description about the data.
print("=== Basic Statistical Description ===")
print(df.describe())
print("\n=== Info ===")
print(df.info())

# d) Check if the data has null values.
print("\n=== Null Values Per Column (Before) ===")
print(df.isnull().sum())

# i. Replace the null values with the mean
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n=== Null Values Per Column (After Mean Fill) ===")
print(df.isnull().sum())

# e) Select at least two columns and aggregate the data using: min, max, count, mean.
print("\n=== Aggregation for Calories and Pulse ===")
agg_data = df[['Calories', 'Pulse']].agg(['min', 'max', 'count', 'mean'])
print(agg_data)

# f) Filter the dataframe to select the rows with calories values between 500 and 1000.
df_cal_500_1000 = df[(df['Calories'] >= 500) & (df['Calories'] <= 1000)]
print("\n=== Rows with Calories between 500 and 1000 ===")
print(df_cal_500_1000)

# g) Filter the dataframe to select the rows with calories values > 500 and pulse < 100.
df_cal_gt_500_pulse_lt_100 = df[(df['Calories'] > 500) & (df['Pulse'] < 100)]
print("\n=== Rows with Calories > 500 and Pulse < 100 ===")
print(df_cal_gt_500_pulse_lt_100)

# h) Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”.
df_modified = df.drop(columns=['Maxpulse'])
print("\n=== df_modified (without Maxpulse) - first 5 rows ===")
print(df_modified.head())

# i) Delete the “Maxpulse” column from the main df dataframe
df.drop(columns=['Maxpulse'], inplace=True)
print("\n=== df after deleting Maxpulse - first 5 rows ===")
print(df.head())

# j) Convert the datatype of Calories column to int datatype.
df['Calories'] = df['Calories'].astype(int)
print("\n=== Data Types After Converting Calories to int ===")
print(df.dtypes)

# k) Using pandas create a scatter plot for the two columns (Duration and Calories).
plt.scatter(df['Duration'], df['Calories'])
plt.xlabel("Duration")
plt.ylabel("Calories")
plt.title("Duration vs Calories")
plt.show()


# =========================
# PART 2: LINEAR REGRESSION
# =========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# a) Import the given “Salary_Data.csv”
salary_df = pd.read_csv("Salary_Data.csv")
print("\n=== Salary Data (first 5 rows) ===")
print(salary_df.head())

# b) Split the data in train_test partitions, such that 1/3 of the data is reserved as test subset.
X = salary_df[['YearsExperience']]
y = salary_df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0
)

# c) Train and predict the model.
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# d) Calculate the mean_squared error
mse = mean_squared_error(y_test, y_pred)
print("\n=== Mean Squared Error (MSE) ===")
print(mse)

# e) Visualize both train and test data using scatter plot.

# Train plot
plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train))
plt.title("Training Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Test plot
plt.scatter(X_test, y_test)
plt.plot(X_train, model.predict(X_train))
plt.title("Test Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
