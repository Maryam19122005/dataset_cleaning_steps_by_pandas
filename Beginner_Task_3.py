import pandas as pd
import numpy as np

# ==================================================
# 1. Load Dataset (make a copy to avoid modifying original)
# ==================================================
file_path = "Netflix Dataset.csv"
df = pd.read_csv(file_path)
df_clean = df.copy()

# ==================================================
# 2. Handle Missing Values
# ==================================================
# Drop columns with too many missing values (threshold: 40%)
threshold = len(df_clean) * 0.4
df_clean = df_clean.dropna(thresh=threshold, axis=1)

# Fill categorical missing values with mode or placeholder
for col in ["Director", "Cast", "Country", "Rating"]:
    if col in df_clean.columns:
        mode_value = df_clean[col].mode()[0] if df_clean[col].mode().any() else "Unknown"
        df_clean[col] = df_clean[col].fillna(mode_value)

# Convert Release_Date to datetime and handle missing values
if "Release_Date" in df_clean.columns:
    df_clean["Release_Date"] = pd.to_datetime(df_clean["Release_Date"], errors="coerce")
    df_clean["Release_Date"] = df_clean["Release_Date"].fillna(df_clean["Release_Date"].mode()[0])

# ==================================================
# 3. Remove Duplicates
# ==================================================
df_clean = df_clean.drop_duplicates()

# ==================================================
# 4. Simple Data Transformations
# ==================================================
# Rename columns for clarity
df_clean = df_clean.rename(columns={
    "Show_Id": "show_id",
    "Category": "category",
    "Title": "title",
    "Director": "director",
    "Cast": "cast",
    "Country": "country",
    "Release_Date": "release_date",
    "Rating": "rating",
    "Duration": "duration",
    "Type": "type",
    "Description": "description"
})

# Extract year and month from release_date
df_clean["release_year"] = df_clean["release_date"].dt.year
df_clean["release_month"] = df_clean["release_date"].dt.month

# Handle Duration: split into number + unit
def split_duration(value):
    if pd.isna(value):
        return np.nan, np.nan
    parts = value.split(" ")
    if len(parts) == 2:
        return int(parts[0]), parts[1]
    return np.nan, np.nan

df_clean[["duration_value", "duration_unit"]] = df_clean["duration"].apply(lambda x: pd.Series(split_duration(str(x))))

# Standardize text columns
text_cols = ["title", "director", "cast", "country", "type", "description"]
for col in text_cols:
    df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()

# ==================================================
# 5. Save Cleaned Dataset
# ==================================================
cleaned_file_path = "Netflix_Dataset_Cleaned.csv"
df_clean.to_csv(cleaned_file_path, index=False)

print("✅ Data cleaning complete!")
print(f"Original Shape: {df.shape}, Cleaned Shape: {df_clean.shape}")
print(f"Cleaned dataset saved to: {cleaned_file_path}")



import seaborn as sns
import matplotlib.pyplot as plt

# Check missing values visually
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Netflix Dataset")
plt.show()
