"""
Modul transformasi data menggunakan TensorFlow Transform (TFT).
"""

import tensorflow as tf
import tensorflow_transform as tft

# Daftar nama fitur dari dataset Pima Indians Diabetes
NUMERIC_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

LABEL_KEY = "Outcome"


def transformed_name(key):
    """Menambahkan suffix '_xf' pada fitur yang telah ditransformasi."""
    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Fungsi preprocessing utama untuk TFT.
    Menerima dictionary input dan mengembalikan dictionary output.
    """
    outputs = {}

    # Normalisasi semua fitur numerik menggunakan z-score
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])

    # Melewatkan label (target) dengan melakukan casting ke int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
