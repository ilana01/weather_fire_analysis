import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# file paths
csv1_file_path = "~/Desktop/DataFiles/FRNSW_data_export_daily_20250513.csv"  
csv2_file_path = "~/Desktop/DataFiles/combined_attributes.csv" 

# expand the '~' to the full path
csv1_file_path = os.path.expanduser(csv1_file_path)
csv2_file_path = os.path.expanduser(csv2_file_path)

# reading in the files
csv1_data = pd.read_csv(csv1_file_path, encoding='latin1') 
csv2_data = pd.read_csv(csv2_file_path, encoding='latin1')
csv2_data = csv2_data.iloc[:, 1:]  # drops the first column

# Convert REPORT_DATE to datetime for consistent merging
csv1_data['REPORT_DATE'] = pd.to_datetime(csv1_data['REPORT_DATE'])
csv2_data['REPORT_DATE'] = pd.to_datetime(csv2_data['REPORT_DATE'])

# Merging the dataframes on the REPORT_DATE column
merged_data = pd.merge(csv1_data, csv2_data, on="REPORT_DATE", how="inner")

# Preview the merged data
print(merged_data.head())

# Define constants for column names
FIRE_INCIDENTS_COLUMN = 'FIRES_INCDS'
WIND_SPEED_COLUMN = '9am wind speed (km/h)'
TEMPERATURE_COLUMN = '9am Temperature'
HUMIDITY_COLUMN = '9am relative humidity (%)'
CLOUD_AMOUNT_COLUMN = '9am cloud amount (oktas)'
PRESSURE_COLUMN = '9am MSL pressure (hPa)'

# Replace 'Calm' with 0 (or another number) in wind speed columns
merged_data[WIND_SPEED_COLUMN] = pd.to_numeric(
    merged_data[WIND_SPEED_COLUMN].replace(r'(?i)^\s*calm\s*$', 0, regex=True),
    errors='coerce'
)

# Ensure all required columns are numeric
columns_to_check = [
    FIRE_INCIDENTS_COLUMN,
    TEMPERATURE_COLUMN, 
    HUMIDITY_COLUMN, 
    CLOUD_AMOUNT_COLUMN, 
    WIND_SPEED_COLUMN, 
    PRESSURE_COLUMN
]
merged_data[columns_to_check] = merged_data[columns_to_check].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values for these columns
cleaned_data = merged_data.dropna(subset=columns_to_check)

# Calculate the correlation matrix
correlation_matrix = cleaned_data[columns_to_check].corr()

# Display the correlation matrix
print(correlation_matrix)

# Heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

#summary statistics
print(merged_data.describe())

#distribution plots
sns.histplot(merged_data['FIRES_INCDS'], bins=20, kde=True)
plt.show()

#pairplot to spot relationship
sns.pairplot(merged_data[['FIRES_INCDS', '9am Temperature', '9am relative humidity (%)', '9am wind speed (km/h)']])
plt.show()



# Regression analysis
predictors = ["9am relative humidity (%)", "9am wind speed (km/h)", "9am Temperature"]  
target = "FIRES_INCDS"  

regression_data = merged_data.dropna(subset=predictors + [target])
X = regression_data[predictors]
y = regression_data[target]

model = LinearRegression()
model.fit(X, y)

# Print coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 score:", model.score(X, y))

# Plot actual vs predicted
y_pred = model.predict(X)
plt.scatter(y, y_pred)
plt.xlabel("Actual FIRES_INCDS")
plt.ylabel("Predicted FIRES_INCDS")
plt.title("Actual vs Predicted Fires")
plt.show()
