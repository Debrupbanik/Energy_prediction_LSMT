import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import io

# Load the trained model
model = tf.keras.models.load_model("lstm_model.h5")

# Load and preprocess the data
file_path = "events.csv"
df = pd.read_csv(file_path)
df['Start time UTC'] = pd.to_datetime(df['Start time UTC'])
df.set_index('Start time UTC', inplace=True)

# Feature Engineering
df['Hour'] = df.index.hour
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Weekday'] = df.index.weekday

# Define seasons
def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3

df['Season'] = df['Month'].apply(get_season)
df['Lag_1'] = df['Electricity consumption in Finland'].shift(1)
df['Lag_24'] = df['Electricity consumption in Finland'].shift(24)
df.fillna(method='bfill', inplace=True)

features = ['Electricity consumption in Finland', 'Hour', 'Day', 'Month', 'Weekday', 'Season', 'Lag_1', 'Lag_24']
df = df[features]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Function to create sequences
def create_dataset(dataset, time_step=60):
    X = []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step])
    return np.array(X)

time_step = 60
X_test = create_dataset(scaled_data, time_step)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(
    np.concatenate((predictions, np.zeros((predictions.shape[0], df.shape[1] - 1))), axis=1)
)[:, 0]

# Streamlit UI
st.title("⚡ Smart Energy Usage Dashboard")
st.write("This interactive dashboard allows you to analyze and visualize electricity consumption trends and predictions.")

# Sidebar Inputs
st.sidebar.header("User Controls")
time_range = st.sidebar.slider("Select Time Range", 1, len(df)-time_step, 500)

# Show data preview
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df.head())

# Actual vs. Predicted Plot
st.subheader("📈 Actual vs Predicted Electricity Consumption")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[time_step:time_range+time_step], df['Electricity consumption in Finland'][time_step:time_range+time_step], label="Actual", color="blue")
ax.plot(df.index[time_step:time_range+time_step], predictions[:time_range], label="Predicted", linestyle="dashed", color="red")
ax.set_xlabel("Time")
ax.set_ylabel("Electricity Consumption")
ax.legend()
st.pyplot(fig)

# Feature Importance
st.subheader("🔥 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Residual Plot
st.subheader("📊 Prediction Errors Over Time")
residuals = df['Electricity consumption in Finland'][time_step:time_range+time_step] - predictions[:time_range]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(residuals, color="purple", label="Residuals")
ax.axhline(y=0, color="black", linestyle="dashed")
ax.set_xlabel("Time")
ax.set_ylabel("Error")
ax.legend()
st.pyplot(fig)

# Histogram of Residuals
st.subheader("📌 Distribution of Prediction Errors")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="purple", ax=ax)
st.pyplot(fig)

# Feature Importance Bar Chart
st.subheader("📌 Feature Importance Based on Correlation")
feature_importance = df.corr()['Electricity consumption in Finland'].drop('Electricity consumption in Finland')
fig, ax = plt.subplots(figsize=(10, 5))
feature_importance.sort_values().plot(kind='barh', color='skyblue', ax=ax)
ax.set_xlabel("Correlation with Electricity Consumption")
st.pyplot(fig)

# Download Report
st.sidebar.subheader("📥 Download Report")
if st.sidebar.button("Generate Report"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Data")
        pd.DataFrame({"Actual": df['Electricity consumption in Finland'][time_step:time_range+time_step].values,
                      "Predicted": predictions[:time_range]}).to_excel(writer, sheet_name="Predictions")
        writer.close()
    st.sidebar.download_button("Download Excel Report", buffer, file_name="energy_report.xlsx", mime="application/vnd.ms-excel")

st.success("✅ Dashboard is up-to-date!")