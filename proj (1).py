import pandas as pd 
import re 
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Important functions 
def match_Other(value): 
    if not isinstance(value, str): return value
    if re.match(r'^Other', value.strip(), re.IGNORECASE): 
        return "Other"
    else: 
        return value

def insert_multilable_encoding(df, encoded_df, column_name): 
    # Logic improved to actually join the encoded columns
    final_df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)
    return final_df

def match_Other_list(array): 
    new_array = []
    for item in array: 
        item = item.strip() 
        if item != "": 
            new_array.append(match_Other(item))
    array_without_repeated = list(dict.fromkeys(new_array))
    return array_without_repeated 

def apply_multilable_encoding(df, column_name, seperator=";", defaultNanValue="None of the above"): 
    if column_name in df.columns:
        df[column_name] = df[column_name].fillna(defaultNanValue)
        series = df[column_name].apply(
            lambda x: [] if x == defaultNanValue else match_Other_list(str(x).split(seperator))
        )
        mlb = MultiLabelBinarizer() 
        encoded_array = mlb.fit_transform(series)
        new_cols_names = [f"{column_name}_{class_name}" for class_name in mlb.classes_]
        encoded_df = pd.DataFrame(encoded_array, columns=new_cols_names, index=df.index)
        return insert_multilable_encoding(df, encoded_df, column_name)
    else: 
        raise Exception(f"the column {column_name} is not in the dataframe")

def apply_label_encoding(df, column_name, defaultNanValue="None of the above"): 
    df[column_name] = df[column_name].fillna(defaultNanValue)
    df[column_name] = df[column_name].apply(match_Other)
    uniqueLabels = df[column_name].unique()
    mapping = {label: i for i, label in enumerate(uniqueLabels)}
    df[column_name] = df[column_name].map(mapping)
    return df 

def classify_features(df, features, threshold=150):
    one_choice_array = []
    multiple_choice_array = []
    text_writing_array = []
    
    for col in features:
        if col not in df.columns: continue
        n_unique = df[col].nunique()
        sample_values = df[col].dropna().astype(str)
        if sample_values.empty:
            one_choice_array.append(col)
            continue
        has_semicolon = sample_values.str.contains(';', regex=False).any()
        if n_unique < threshold and not has_semicolon:
            one_choice_array.append(col)
        elif has_semicolon:
            multiple_choice_array.append(col)
        else:
            text_writing_array.append(col)
    return one_choice_array, multiple_choice_array, text_writing_array

# --- MAIN PROCESS ---

df = pd.read_csv('train_data.csv')

# Initial cleaning
df = df.drop(columns=["ResponseId"], errors='ignore')
if "AIExplain" in df.columns: df = df.drop(columns=["AIExplain"], axis=1)

# Drop columns with too many NaNs
df = df.dropna(subset=["JobSat"], axis=0)
for col in df.columns: 
    if df[col].isnull().sum() > 3000: 
        df = df.drop(col, axis=1)

# Convert JobSat to numeric (Ordinal Encoding)
jobsat_mapping = {
    'Very satisfied': 5,
    'Slightly satisfied': 4,
    'Neither satisfied nor dissatisfied': 3,
    'Slightly dissatisfied': 2,
    'Very dissatisfied': 1
}
df['JobSat'] = df['JobSat'].map(jobsat_mapping).fillna(3)

# Process categorical columns
dt_obj_features = df.select_dtypes(include=['object']).columns
one_choice, multi_choice, text_cols = classify_features(df, dt_obj_features)

for col in one_choice: 
    df = apply_label_encoding(df, col)
for col in multi_choice: 
    df = apply_multilable_encoding(df, col)
for col in text_cols: 
    df = df.drop(col, axis=1)

# Handle remaining floats/missing
float_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in float_cols:
    df[col] = df[col].fillna(df[col].median())

# --- MODELING ---

X = df.drop(columns=['JobSat'])
y = df['JobSat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset shape after preprocessing: {df.shape}")

# 1. Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Linear Regression R² Score: {r2_score(y_test, y_pred):.4f}")

# 2. Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
print(f"Ridge Regression R² Score: {r2_score(y_test, ridge_model.predict(X_test)):.4f}")

# 3. XGBoost
try:
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train)
    print(f"XGBoost R² Score: {r2_score(y_test, xgb_model.predict(X_test)):.4f}")
except Exception as e:
    print(f"XGBoost failed: {e}")
