import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import os

# =========================================================
# SETTINGS
# =========================================================

DATASET_DIR = r"C:\AIML\food-recognition-calorie-estimation\data\food10"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

CLASS_NAMES = [
    "apple_pie",
    "baklava",
    "bread_pudding",
    "cheesecake",
    "chocolate_cake",
    "cup_cakes",
    "donuts",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad"
]

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("FOOD RECOGNITION MODEL TRAINING")
print("=" * 60)

print("\nClasses:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nDataset loaded successfully.")

print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())


# =========================================================
# PERFORMANCE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# =========================================================
# MOBILE NET V2
# =========================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially
base_model.trainable = False


# =========================================================
# BUILD MODEL
# =========================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/food_recognition_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]


# =========================================================
# TRAIN
# =========================================================

print("\nStarting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================================================
# SAVE CLASS NAMES
# =========================================================

with open(
    "models/class_names.json",
    "w"
) as f:

    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )


# =========================================================
# FINAL RESULTS
# =========================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "\nModel saved to:"
)

print(
    "models/food_recognition_model.keras"
)

print(
    "\nClass names saved to:"
)

print(
    "models/class_names.json"
)

print("=" * 60)