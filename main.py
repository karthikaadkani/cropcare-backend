from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = tf.keras.models.load_model(
    "cropcare_model.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False
)

# IMPORTANT: Exact training class order
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


@app.get("/")
def home():
    return {
        "message": "CropCare AI Backend is Running!"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Open image
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert to array
    image_array = np.array(image).astype(np.float32)

# Apply MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

# Add batch dimension
    image_array = np.expand_dims(
    image_array,
    axis=0
)

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Print all predictions
    print("\nALL PREDICTIONS:")

    for i, probability in enumerate(predictions[0]):
        print(
            i,
            CLASS_NAMES[i],
            round(float(probability) * 100, 2),
            "%"
        )

    # Find highest prediction
    predicted_index = int(
        np.argmax(predictions[0])
    )

    # Get confidence
    confidence = float(
        np.max(predictions[0]) * 100
    )

    # Debug information
    print("FINAL INDEX:", predicted_index)
    print(
        "FINAL CLASS:",
        CLASS_NAMES[predicted_index]
    )

    # Send result to website
    return {
        "disease": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 2)
    }