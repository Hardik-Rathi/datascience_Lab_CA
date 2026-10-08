import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
df = pd.read_csv('export.csv')

# ==========================================
# 1. Attribute Identification & Data Quality
# ==========================================
# Quick look at the data types and missing values
print(df.info())
print("\nDuplicates found:", df.duplicated().sum())

# Dropping duplicates right away
df = df.drop_duplicates()

# Let's drop columns that are completely empty (from info, these are all NaNs)
empty_cols = ['Geocoded_City (address)', 'Geocoded_City (state)', 'Geocoded_City (zip)']
df = df.drop(columns=empty_cols)

# ==========================================
# 2. String Cleaning (Crucial step for this dataset)
# ==========================================
# Several columns are loaded as 'object' because they contain '$' and ',' 
# We need to convert them to continuous numeric attributes before processing
cols_to_clean = ['cur_passengers', 'cur_fare', 'distance', 'ly_passengers', 'ly_fare', 'ly_distance']

for col in cols_to_clean:
    # Remove dollar signs and commas, then convert to float
    df[col] = df[col].astype(str).str.replace('$', '', regex=False)
    df[col] = df[col].str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ==========================================
# 3. Missing Data Handling
# ==========================================
print("\nMissing values before imputation:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Using Median imputation for continuous numeric features to be robust against outliers
numeric_cols_with_na = ['ly_passengers', 'ly_fare', 'ly_yield', 'ly_distance']
for col in numeric_cols_with_na:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# For the categorical 'Geocoded_City' columns, we'll use mode (most frequent)
cat_cols_with_na = ['Geocoded_City', 'Geocoded_City (city)']
for col in cat_cols_with_na:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

# ==========================================
# 4. Data Transformation & Normalization
# ==========================================
# I'll create a new dataframe for scaled features so we don't mess up the original
df_scaled = df.copy()

# A. Min-Max Normalization on passenger counts (scales to 0-1)
min_max = MinMaxScaler()
df_scaled['cur_passengers_minmax'] = min_max.fit_transform(df[['cur_passengers']])

# B. Z-score Standardization on fares (centers around mean 0, std dev 1)
z_scaler = StandardScaler()
df_scaled['cur_fare_zscore'] = z_scaler.fit_transform(df[['cur_fare']])

# C. Decimal Scaling on 'distance'
# Find the max absolute value to determine how many decimal places to shift
max_distance = df['distance'].abs().max()
j = len(str(int(max_distance)))
df_scaled['distance_decimal_scaled'] = df['distance'] / (10 ** j)

# ==========================================
# 5. Discretization & Binning
# ==========================================
# We'll categorize the 'distance' attribute into intervals

# A. Equal-width binning (e.g., Short-haul, Medium-haul, Long-haul)
df_scaled['distance_category_eq_width'] = pd.cut(
    df['distance'], 
    bins=3, 
    labels=['Short-Haul', 'Medium-Haul', 'Long-Haul']
)

# B. Equal-frequency binning (Quartiles: ~same number of flights in each bin)
df_scaled['distance_quartiles'] = pd.qcut(
    df['distance'], 
    q=4, 
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

# Optional: Histogram analysis to check our binning distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['distance'], bins=30, kde=True)
plt.title('Distribution of Flight Distances (For Binning Analysis)')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

# Save the preprocessed dataset for Phase 2
df_scaled.to_csv('export_phase1_cleaned.csv', index=False)
print("\nPhase 1 complete. Cleaned data saved to 'export_phase1_cleaned.csv'")