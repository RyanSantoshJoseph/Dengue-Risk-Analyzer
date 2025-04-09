import gradio as gr
import joblib
import numpy as np

# Load dengue prediction model
try:
    model = joblib.load("dengue_rf_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Predict dengue risk
def predict_dengue(temperature, humidity, rainfall, month):
    if None in [temperature, humidity, rainfall, month]:
        return "❗ Please fill all the input fields."

    try:
        month_number = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month) + 1
        features = np.array([[temperature, humidity, rainfall, month_number]])
        prediction = model.predict(features)[0]
        return "🦠 High Risk of Dengue" if prediction == 1 else "✅ Low Risk of Dengue"
    except Exception as e:
        return f"Prediction failed: {str(e)}"

# Months list
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🦟 Dengue Risk Predictor")
    gr.Markdown("Enter environmental values to predict the risk of dengue.")

    with gr.Row():
        with gr.Column():
            temperature = gr.Number(label="Temperature (°C)")
            humidity = gr.Number(label="Humidity (%)")
            rainfall = gr.Number(label="Rainfall (mm)")
            month = gr.Dropdown(choices=month_names, label="Month")

            submit_btn = gr.Button("Predict Risk")
            output = gr.Textbox(label="Prediction")

            submit_btn.click(
                fn=predict_dengue,
                inputs=[temperature, humidity, rainfall, month],
                outputs=output
            )

# Run app
if __name__ == "__main__":
    demo.launch()
