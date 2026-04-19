import sys
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("glaucoma_model.h5")

img_path = sys.argv[1]

img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0][0]

if pred > 0.5:
    print(f"GLAUCOMA:{pred}")
else:
    print(f"NORMAL:{1-pred}")