from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import wikipedia
import io

app = FastAPI()

# Allow your Lovable/Bolt frontend to communicate with this Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI Model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Process the image for the AI
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Get predictions
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    # Format the results
    results = []
    for _, label, score in decoded_predictions:
        results.append({"label": label.replace('_', ' ').title(), "confidence": float(score)})

    # Fetch Wikipedia info for the top prediction
    top_label = results[0]["label"]
    try:
        wiki_summary = wikipedia.summary(top_label, sentences=3)
        wiki_url = f"https://en.wikipedia.org/wiki/{top_label.replace(' ', '_')}"
    except Exception:
        wiki_summary = "Wikipedia information is not available for this specific item."
        wiki_url = ""

    return {
        "predictions": results,
        "wiki_summary": wiki_summary,
        "wiki_url": wiki_url
    }