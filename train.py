import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ======================
# DATA PATHS
# ======================

train_dir = "dataset_new/Training"
test_dir  = "dataset_new/Testing"

# ======================
# DATA GENERATORS
# ======================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

print("Class indices:", train_gen.class_indices)

# ======================
# MODEL — TRANSFER LEARNING
# ======================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze most layers first
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Custom head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# CLASS WEIGHTS (IMBALANCE FIX)
# ======================

class_weight = {
    0: 1.0,   # glaucoma
    1: 4.0    # normal boosted
}

# ======================
# TRAIN
# ======================

print("🚀 Starting training...")

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=20,
    class_weight=class_weight
)

# ======================
# SAVE MODEL
# ======================

model.save("glaucoma_model.h5")

print("✅ Training complete. Model saved.")