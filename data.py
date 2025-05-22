import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define file paths for emergency incidents and weather data
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

# DATA CLEANING & PREPROCESSING

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


# ANALYSIS 1: FIRES_INCDS vs Weather

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


# ANALYSIS 2: Combined Analysis of STORM_RELATED_INCDS and WIRES_DOWN_INCDS



# Ensure the emergency call types are numeric
merged_data[STORM_INCIDENTS_COLUMN] = pd.to_numeric(merged_data[STORM_INCIDENTS_COLUMN], errors='coerce')
merged_data[WIRES_DOWN_COLUMN]   = pd.to_numeric(merged_data[WIRES_DOWN_COLUMN], errors='coerce')

# Create a dataset for the combined analysis by dropping missing values for predictors and both targets
combined_data = merged_data.dropna(subset=predictors + [STORM_INCIDENTS_COLUMN, WIRES_DOWN_COLUMN])
combined_data = combined_data.sort_values("REPORT_DATE")

# Prepare predictor variables (X) and multi-target responses (Y)
X_multi = combined_data[predictors]
Y_multi = combined_data[[STORM_INCIDENTS_COLUMN, WIRES_DOWN_COLUMN]]

# Perform multi-output regression using LinearRegression
multi_model = LinearRegression()
multi_model.fit(X_multi, Y_multi)

# Display the coefficients and intercept for each target.
# multi_model.coef_ shape: (2, n_features) --> first row for storm, second for wires down.
coefficients = pd.DataFrame(
    multi_model.coef_,
    index=[STORM_INCIDENTS_COLUMN, WIRES_DOWN_COLUMN],
    columns=predictors
)
intercepts = pd.Series(multi_model.intercept_, index=[STORM_INCIDENTS_COLUMN, WIRES_DOWN_COLUMN])

print("\nMulti-Output Regression Analysis for Storm and Wires Down Incidents:")
print("Coefficients:")
print(coefficients)
print("\nIntercepts:")
print(intercepts)
print("\nOverall R^2 score:", multi_model.score(X_multi, Y_multi))

# Obtain predictions for the multi-output regression model
Y_multi_pred = multi_model.predict(X_multi)

# Plot actual vs predicted values in subplots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(combined_data[STORM_INCIDENTS_COLUMN], Y_multi_pred[:, 0], color='tab:blue', alpha=0.7)
plt.xlabel("Actual Storm Incidents")
plt.ylabel("Predicted Storm Incidents")
plt.title("Storm Incidents: Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(combined_data[WIRES_DOWN_COLUMN], Y_multi_pred[:, 1], color='tab:orange', alpha=0.7)
plt.xlabel("Actual Wires Down Incidents")
plt.ylabel("Predicted Wires Down Incidents")
plt.title("Wires Down Incidents: Actual vs Predicted")

plt.tight_layout()
plt.show()

def load_weather_data(filepath, city):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except Exception as e:
        raise FileNotFoundError(f"Error loading {filepath}: {e}")
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Date' not in df.columns:
        raise KeyError(f"Expected a 'Date' column in file: {filepath}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['City'] = city
    return df

# Define file paths for the three weather datasets.
wollongong_fp = os.path.expanduser("DataFiles/combined_Wollongong.csv")
sydney_fp     = os.path.expanduser("DataFiles/combined_attributes.csv")
newcastle_fp  = os.path.expanduser("DataFiles/combined_Newcastle.csv")

# Load each weather dataset.
wollongong_weather = load_weather_data(wollongong_fp, "Wollongong")
sydney_weather     = load_weather_data(sydney_fp, "Sydney")
newcastle_weather  = load_weather_data(newcastle_fp, "Newcastle")

# Combine all weather data.
combined_weather = pd.concat([wollongong_weather, sydney_weather, newcastle_weather], ignore_index=True)
combined_weather.sort_values("Date", inplace=True)
print("Combined Weather Data Preview:")
print(combined_weather.head())

# Define key weather metrics.
weather_cols = ["9am Temperature (°C)", "9am relative humidity (%)", "9am wind speed (km/h)"]

# Replace non-numeric wind speed texts like "Calm" with 0.
combined_weather["9am wind speed (km/h)"] = combined_weather["9am wind speed (km/h)"].replace(
    r'(?i)^\s*calm\s*$', "0", regex=True
)
for col in weather_cols:
    combined_weather[col] = pd.to_numeric(combined_weather[col], errors='coerce')

# Aggregate weather data by City (averaging across all daily records).
weather_by_city = combined_weather.groupby('City')[weather_cols].mean().reset_index()
print("\nAggregated Weather Data by City:")
print(weather_by_city)


# LOAD, CLEAN, AND AGGREGATE LGA INCIDENT DATA (Yearly Breakdown)


# Load the LGA incidents file.
lga_filepath = os.path.expanduser("DataFiles/LGA_Incidents.csv")
try:
    lga_incidents = pd.read_csv(lga_filepath)
except Exception as e:
    raise FileNotFoundError(f"Error loading {lga_filepath}: {e}")

print("\nLGA Incidents Data Preview (Raw):")
print(lga_incidents.head())

# Clean Column Names
# Remove extra whitespace and any quotation marks from column headers.
lga_incidents.columns = lga_incidents.columns.str.strip().str.replace('"', '')

# Convert numeric columns (all columns except the first) by removing commas and converting to numeric.
for col in lga_incidents.columns[1:]:
    lga_incidents[col] = lga_incidents[col].astype(str).str.replace(",", "").str.strip()
    lga_incidents[col] = pd.to_numeric(lga_incidents[col], errors='coerce')

# Define a mapping from LGA names to City names.
lga_to_city = {
    'SYDNEY': 'Sydney',
    'INNER WEST': 'Sydney',
    'WAVERLEY': 'Sydney',
    'WILLLOUGHBY': 'Sydney',
    'CAMDEN': 'Sydney',
    'CANTERBURY-BANKSTOWN': 'Sydney',
    'BLACKTOWN': 'Sydney',
    'NEWCASTLE': 'Newcastle',
    'LAKE MACQUARIE': 'Newcastle',
    'WOLLONGONG': 'Wollongong',
    'SHELLHARBOUR': 'Wollongong'
    # Extend the mapping as needed.
}

# Ensure the incidents file contains the "Local Government Area" column.
if "Local Government Area" not in lga_incidents.columns:
    raise KeyError("The incidents file must contain a 'Local Government Area' column.")

# Create a "City" column using the mapping.
lga_incidents["City"] = lga_incidents["Local Government Area"].map(lga_to_city)
lga_incidents = lga_incidents.dropna(subset=["City"])
print("\nLGA Incidents with City Assignment (Preview):")
print(lga_incidents.head())

# Define the incident column names you want to analyze.
incident_cols = [
    "Fires & explosions", 
    "Storm, floods and other natural disasters & calls for assistance from other agencies",
    "Total primary incidents"
]

# Aggregate incident data by City by taking the average across records (if multiple years exist).
incidents_by_city = lga_incidents.groupby("City")[incident_cols].mean().reset_index()
print("\nAggregated Incident Data by City (Averages):")
print(incidents_by_city[["City", "Total primary incidents"]])


# VISUALISATION - GROUPED BAR CHART FOR INCIDENT TYPES BY CITY

# Reshape the incident data to long format for a grouped bar chart.
incidents_long = pd.melt(incidents_by_city,
                         id_vars=["City"],
                         value_vars=incident_cols,
                         var_name="Incident Type",
                         value_name="Average Count")
print("\nIncidents Data (Long Format):")
print(incidents_long)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="City", y="Average Count", hue="Incident Type", data=incidents_long)
ax.set_title("Average Incident Counts by City (Yearly Breakdown)")
ax.set_xlabel("City")
ax.set_ylabel("Average Incident Count")
plt.legend(title="Incident Type")
plt.tight_layout()
plt.show()

# MERGE & CORRELATION ANALYSIS (City-Level Averages)


# Merge weather and incident data by City.
merged_city = pd.merge(weather_by_city, incidents_by_city, on="City", how="inner")
print("\nMerged Data by City:")
print(merged_city)

# For correlation analysis, select relevant columns.
cols_for_corr = weather_cols + ["Total primary incidents"]
corr_matrix = merged_city[cols_for_corr].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (City-Level Averages)")
plt.tight_layout()
plt.show()