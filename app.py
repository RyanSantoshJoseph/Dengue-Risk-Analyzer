import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load("dengue_rf_model.pkl")

# Define the prediction function
def predict_dengue(temperature, humidity, rainfall, month):
    try:
        features = np.array([[temperature, humidity, rainfall, month]])
        prediction = model.predict(features)[0]
        return "High Risk" if prediction == 1 else "Low Risk"
    except Exception as e:
        return f"❌ Prediction Error: {e}"

# Create Gradio Interface
demo = gr.Interface(
    fn=predict_dengue,
    inputs=[
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Rainfall (mm)"),
        gr.Dropdown(choices=list(range(1, 13)), label="Month")  # 1 to 12 for months
    ],
    outputs=gr.Textbox(label="Outbreak Risk"),
    title="Dengue Outbreak Prediction",
    description="Enter weather parameters to predict dengue outbreak risk."
)

demo.launch()
