import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\nambu\Downloads\traffic volume.csv")

# If there's a date column, parse it
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # ✅ Updated line
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

# Define features and target
features = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day']
target = 'traffic_volume'

# Check for missing columns
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing or invalid values
df = df.dropna(subset=features + [target])

# Handle non-numeric values (like 'None', 'Clear', etc.)
label_encoders = {}
for col in ['holiday', 'weather']:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Now prepare input and output
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save scaler and model
with open('scale.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save encoders if needed during prediction
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Model, scaler, and encoders saved successfully.")
