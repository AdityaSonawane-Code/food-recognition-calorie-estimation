import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from calorie_data import CALORIE_DATA


MODEL_PATH = r"C:\AIML\food-recognition-calorie-estimation\models\food_recognition_model.keras"
CLASS_PATH = r"C:\AIML\food-recognition-calorie-estimation\models\class_names.json"


print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")


def predict_food(image_path):

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Same preprocessing used during MobileNetV2 training
    img = image.load_img(
    image_path,
    target_size=(224, 224)
)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
    img_array,
    axis=0
)
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    index = np.argmax(predictions)

    food = class_names[index]

    confidence = float(predictions[index]) * 100

    calorie_info = CALORIE_DATA.get(
        food,
        {
            "calories": 0,
            "serving": "unknown"
        }
    )

    print("\n" + "=" * 55)
    print("FOOD RECOGNITION RESULT")
    print("=" * 55)

    print(
        f"Food       : {food.replace('_', ' ').title()}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

    print(
        f"Calories   : ~{calorie_info['calories']} kcal"
    )

    print(
        f"Serving    : {calorie_info['serving']}"
    )

    print("=" * 55)


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            r'python src\predict.py "image_path"'
        )

        sys.exit()

    image_path = sys.argv[1]

    predict_food(image_path)