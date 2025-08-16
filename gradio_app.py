import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crops.joblib")  # Ensure crops.joblib exists

# Prediction function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    try:
        inputs = np.array([[float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall)]])
        prediction = model.predict(inputs)[0]
        return f"🌾 Recommended Crop: {prediction.capitalize()}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🌱 Crop Recommendation System")

    with gr.Row():
        with gr.Column():
            N           = gr.Number(label="Nitrogen (N)")
            P           = gr.Number(label="Phosphorous (P)")
            K           = gr.Number(label="Potassium (K)")
            temperature = gr.Number(label="Temperature (°C)")
        with gr.Column():
            humidity = gr.Number(label="Humidity (%)")
            ph       = gr.Number(label="pH")
            rainfall = gr.Number(label="Rainfall (mm)")

    submit_btn = gr.Button("🚀 Recommend Crop")
    output_text = gr.Textbox(label="Prediction Result")

    submit_btn.click(fn=recommend_crop,
                     inputs=[N, P, K, temperature, humidity, ph, rainfall],
                     outputs=output_text)

demo.launch(share=True)
