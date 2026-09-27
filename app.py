import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing import image

from src.calorie_data import CALORIE_DATA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Food Recognition & Calorie Estimator",
    page_icon="🍽️",
    layout="centered"
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = r"C:\AIML\food-recognition-calorie-estimation\models\food_recognition_model.keras"
CLASS_PATH = r"C:\AIML\food-recognition-calorie-estimation\models\class_names.json"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_data
def load_class_names():

    with open(CLASS_PATH, "r") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()


# =========================================================
# PREDICTION
# =========================================================

def predict_food(uploaded_image):

    # Convert uploaded image to RGB
    img = uploaded_image.convert("RGB")

    # EXACT SAME SIZE AS predict.py
    img = img.resize((224, 224))

    # Convert to NumPy array
    img_array = np.array(img)

    # EXACT SAME AS predict.py
    img_array = img_array.astype("float32")

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    index = np.argmax(predictions)

    food = class_names[index]

    confidence = float(
        predictions[index]
    ) * 100

    return food, confidence


# =========================================================
# FORMAT FOOD NAME
# =========================================================

def format_food_name(food):

    return food.replace(
        "_",
        " "
    ).title()


# =========================================================
# USER INTERFACE
# =========================================================

st.title(
    "🍽️ Food Recognition & Calorie Estimator"
)

st.write(
    "Upload a food image and the AI model will "
    "recognize the food and provide an approximate "
    "calorie estimate."
)

st.divider()


uploaded_file = st.file_uploader(
    "📷 Upload Food Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # Open uploaded image
    uploaded_image = Image.open(
        uploaded_file
    )

    # Display image
    st.image(
        uploaded_image,
        caption="Uploaded Food Image",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Recognize Food",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing food image..."
        ):

            food, confidence = predict_food(
                uploaded_image
            )

        food_display = format_food_name(
            food
        )

        calorie_info = CALORIE_DATA.get(
            food,
            {
                "calories": 0,
                "serving": "Unknown"
            }
        )

        calories = calorie_info[
            "calories"
        ]

        serving = calorie_info[
            "serving"
        ]


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader(
            "🤖 Recognition Result"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Detected Food",
                food_display
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        st.progress(
            min(
                int(confidence),
                100
            )
        )


        # =================================================
        # CALORIE INFORMATION
        # =================================================

        st.subheader(
            "🔥 Calorie Information"
        )


        col3, col4 = st.columns(2)


        with col3:

            st.metric(
                "Approx. Calories",
                f"{calories} kcal"
            )


        with col4:

            st.metric(
                "Serving",
                serving
            )


        st.info(
            "⚠️ Calorie values are approximate reference "
            "values. Actual calories depend on portion size, "
            "ingredients, and preparation method."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Food Recognition & Calorie Estimation | "
    "MobileNetV2 + TensorFlow + Streamlit"
)