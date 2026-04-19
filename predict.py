import numpy as np
from PIL import Image
import tensorflow as tf
import sys

# Load model
model = tf.keras.models.load_model("glaucoma_model.h5")

# Get image path from terminal
img_path = sys.argv[1]

# Load and preprocess image
image = Image.open(img_path).convert("RGB")
image = image.resize((224, 224))

img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Output result
if prediction > 0.5:
    print(f"⚠️ Glaucoma Detected ({prediction:.2f})")
else:
    print(f"✅ Normal Eye ({1 - prediction:.2f})")