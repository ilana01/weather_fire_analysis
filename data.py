import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# file paths
csv1_file_path = "DataFiles/FRNSW_data_export_daily_20250513.csv"  
csv2_file_path = "DataFiles/combined_attributes.csv" 

# expand the '~' to the full path
csv1_file_path = os.path.expanduser(csv1_file_path)
csv2_file_path = os.path.expanduser(csv2_file_path)

# reading in the files
csv1_data = pd.read_csv(csv1_file_path, encoding='latin1') 
csv2_data = pd.read_csv(csv2_file_path, encoding='utf-8-sig')
csv2_data = csv2_data.iloc[:, 1:]  # drops the first column

# Convert REPORT_DATE to datetime for consistent merging
csv1_data['REPORT_DATE'] = pd.to_datetime(csv1_data['REPORT_DATE'])
csv2_data['Date'] = pd.to_datetime(csv2_data['Date'])

# Rename the 'Date' column to match
csv2_data.rename(columns={'Date': 'REPORT_DATE'}, inplace=True)
# Merging the dataframes on the REPORT_DATE column
merged_data = pd.merge(csv1_data, csv2_data, on="REPORT_DATE", how="inner")
# Preview the merged data
print(merged_data.head())

# Define constants for column names used in analysis of FIRES_INCDS
FIRE_INCIDENTS_COLUMN = 'FIRES_INCDS'
WIND_SPEED_COLUMN = '9am wind speed (km/h)'
TEMPERATURE_COLUMN = '9am Temperature (°C)'
HUMIDITY_COLUMN = '9am relative humidity (%)'
CLOUD_AMOUNT_COLUMN = '9am cloud amount (oktas)'
PRESSURE_COLUMN = '9am MSL pressure (hPa)'

# New Emergency Callout Types for additional analyses
STORM_INCIDENTS_COLUMN = 'STORM_RELATED_INCDS'
WIRES_DOWN_COLUMN       = 'WIRES_DOWN_INCDS'

# Replace 'Calm' with 0 in wind speed columns (handle case-insensitive matching)
merged_data[WIND_SPEED_COLUMN] = pd.to_numeric(
    merged_data[WIND_SPEED_COLUMN].replace(r'(?i)^\s*calm\s*$', 0, regex=True),
    errors='coerce'
)

# Ensure required columns (for FIRES_INCDS analysis) are numeric
columns_to_check = [
    FIRE_INCIDENTS_COLUMN,
    TEMPERATURE_COLUMN, 
    HUMIDITY_COLUMN, 
    CLOUD_AMOUNT_COLUMN, 
    WIND_SPEED_COLUMN, 
    PRESSURE_COLUMN
]
merged_data[columns_to_check] = merged_data[columns_to_check].apply(pd.to_numeric, errors='coerce')

# ================================================
# ANALYSIS 1: FIRES_INCDS vs Weather
# ================================================

# Drop rows with missing values in the key columns
cleaned_data = merged_data.dropna(subset=columns_to_check)

# Calculate correlation matrix for FIRES_INCDS and weather-related columns
correlation_matrix = cleaned_data[columns_to_check].corr()
print("\nCorrelation Matrix for FIRES_INCDS and Weather:")
print(correlation_matrix)

# Plot correlation heatmap for FIRES_INCDS
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Fires and Weather')
plt.show()

# Summary statistics and distribution plots
print("\nSummary Statistics:")
print(merged_data.describe())

sns.histplot(merged_data[FIRE_INCIDENTS_COLUMN], bins=20, kde=True)
plt.title("Distribution of FIRES_INCDS")
plt.show()

sns.pairplot(merged_data[[FIRE_INCIDENTS_COLUMN, TEMPERATURE_COLUMN, HUMIDITY_COLUMN, WIND_SPEED_COLUMN]])
plt.suptitle("Pairplot: FIRES_INCDS vs Weather", y=1.02)
plt.show()

# Regression analysis for FIRES_INCDS using three predictors
predictors = [HUMIDITY_COLUMN, WIND_SPEED_COLUMN, TEMPERATURE_COLUMN]
target = FIRE_INCIDENTS_COLUMN

regression_data = merged_data.dropna(subset=predictors + [target])
X = regression_data[predictors]
y = regression_data[target]

model = LinearRegression()
model.fit(X, y)

print("\nRegression Analysis for FIRES_INCDS:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 score:", model.score(X, y))

# Plot actual vs predicted for FIRES_INCDS
y_pred = model.predict(X)
plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual FIRES_INCDS")
plt.ylabel("Predicted FIRES_INCDS")
plt.title("FIRES_INCDS: Actual vs Predicted")
plt.show()

# ================================================
# ANALYSIS 2: STORM_RELATED_INCDS vs Weather
# ================================================

# Ensure the STORM_RELATED_INCDS column is numeric
merged_data[STORM_INCIDENTS_COLUMN] = pd.to_numeric(merged_data[STORM_INCIDENTS_COLUMN], errors='coerce')

# Define weather-related columns (same as before)
weather_columns = [TEMPERATURE_COLUMN, HUMIDITY_COLUMN, CLOUD_AMOUNT_COLUMN, WIND_SPEED_COLUMN, PRESSURE_COLUMN]

# For correlation, select STORM_RELATED_INCDS and weather columns, dropping rows with missing data
columns_storm = [STORM_INCIDENTS_COLUMN] + weather_columns
storm_data = merged_data.dropna(subset=columns_storm)

# Calculate and display correlation matrix
storm_correlation_matrix = storm_data[columns_storm].corr()
print("\nCorrelation Matrix for STORM_RELATED_INCDS and Weather:")
print(storm_correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(storm_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Storm Incidents and Weather')
plt.show()

# Regression analysis for STORM_RELATED_INCDS using selected weather predictors
storm_predictors = predictors  # reusing the same 3 predictors as before
storm_regression_data = merged_data.dropna(subset=storm_predictors + [STORM_INCIDENTS_COLUMN])
X_storm = storm_regression_data[storm_predictors]
y_storm = storm_regression_data[STORM_INCIDENTS_COLUMN]

storm_model = LinearRegression()
storm_model.fit(X_storm, y_storm)

print("\nRegression Analysis for STORM_RELATED_INCDS:")
print("Coefficients:", storm_model.coef_)
print("Intercept:", storm_model.intercept_)
print("R^2 score:", storm_model.score(X_storm, y_storm))

# Plot actual vs predicted for STORM_RELATED_INCDS
y_storm_pred = storm_model.predict(X_storm)
plt.figure()
plt.scatter(y_storm, y_storm_pred)
plt.xlabel("Actual STORM_RELATED_INCDS")
plt.ylabel("Predicted STORM_RELATED_INCDS")
plt.title("STORM_RELATED_INCDS: Actual vs Predicted")
plt.show()

# ================================================
# ANALYSIS 3: WIRES_DOWN_INCDS vs Weather
# ================================================

# Ensure the WIRES_DOWN_INCDS column is numeric
merged_data[WIRES_DOWN_COLUMN] = pd.to_numeric(merged_data[WIRES_DOWN_COLUMN], errors='coerce')

# Prepare data for correlation analysis between WIRES_DOWN_INCDS and weather variables
columns_wires = [WIRES_DOWN_COLUMN] + weather_columns
wires_data = merged_data.dropna(subset=columns_wires)

# Calculate and display the correlation matrix
wires_correlation_matrix = wires_data[columns_wires].corr()
print("\nCorrelation Matrix for WIRES_DOWN_INCDS and Weather:")
print(wires_correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(wires_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix: Wires Down Incidents and Weather')
plt.show()

# Regression analysis for WIRES_DOWN_INCDS using the same weather predictors
wires_regression_data = merged_data.dropna(subset=predictors + [WIRES_DOWN_COLUMN])
X_wires = wires_regression_data[predictors]
y_wires = wires_regression_data[WIRES_DOWN_COLUMN]

wires_model = LinearRegression()
wires_model.fit(X_wires, y_wires)

print("\nRegression Analysis for WIRES_DOWN_INCDS:")
print("Coefficients:", wires_model.coef_)
print("Intercept:", wires_model.intercept_)
print("R^2 score:", wires_model.score(X_wires, y_wires))

# Plot actual vs predicted for WIRES_DOWN_INCDS
y_wires_pred = wires_model.predict(X_wires)
plt.figure()
plt.scatter(y_wires, y_wires_pred)
plt.xlabel("Actual WIRES_DOWN_INCDS")
plt.ylabel("Predicted WIRES_DOWN_INCDS")
plt.title("WIRES_DOWN_INCDS: Actual vs Predicted")
plt.show()