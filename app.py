# app.py
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
df = pd.read_csv("Student Mental Health Analysis During Online Learning.csv")

# 2. Drop irrelevant or redundant columns
df.drop(columns=["Timestamp"], inplace=True)

# 3. Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# 4. Split features and target
X = df.drop("Mental health affected during online classes", axis=1)
y = df["Mental health affected during online classes"]

# 5. Scale features (optional for RandomForest, but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 7. Define Gradio input interface
input_features = list(X.columns)
input_components = []

for feature in input_features:
    if feature in label_encoders:
        choices = list(label_encoders[feature].classes_)
        input_components.append(gr.Dropdown(choices=choices, label=feature))
    else:
        input_components.append(gr.Slider(minimum=float(df[feature].min()), 
                                          maximum=float(df[feature].max()), 
                                          step=1, label=feature))

# 8. Prediction function
def predict_mental_health(*inputs):
    input_dict = {}
    for i, feature in enumerate(input_features):
        val = inputs[i]
        if feature in label_encoders:
            val = label_encoders[feature].transform([val])[0]
        input_dict[feature] = val

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    result = "Yes" if pred == 1 else "No"
    return f"Mental Health Affected: {result}"

# 9. Gradio app
gr.Interface(
    fn=predict_mental_health,
    inputs=input_components,
    outputs="text",
    title="Student Mental Health Prediction",
    description="Predict whether a student's mental health is affected during online learning based on various factors."
).launch()
