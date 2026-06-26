"""
Modul Tuner untuk mencari hyperparameter terbaik menggunakan KerasTuner.
"""

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.tuner.component import TunerFnResult

# Sesuaikan dengan fitur di transform.py
NUMERIC_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
LABEL_KEY = "Outcome"

def transformed_name(key):
    """Menambahkan suffix '_xf' pada fitur yang telah ditransformasi."""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Membaca file TFRecord yang dikompresi gzip."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """Fungsi untuk memuat dan mem-batch dataset."""
    transform_feature_spec = tf_transform_output.transformed_feature_spec()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset

def model_builder(hp):
    """Membangun arsitektur model Keras untuk di-tuning."""
    inputs = {}
    for feature in NUMERIC_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.float32
        )

    # Menggabungkan semua input menjadi satu tensor
    concatenated_inputs = tf.keras.layers.Concatenate()(list(inputs.values()))

    # Pencarian hyperparameter untuk jumlah unit di layer pertama
    x = tf.keras.layers.Dense(
        units=hp.Int('units_1', min_value=16, max_value=128, step=16),
        activation='relu'
    )(concatenated_inputs)

    # Pencarian hyperparameter untuk nilai dropout
    x = tf.keras.layers.Dropout(
        rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    )(x)

    # Pencarian hyperparameter untuk jumlah unit di layer kedua
    x = tf.keras.layers.Dense(
        units=hp.Int('units_2', min_value=8, max_value=64, step=8),
        activation='relu'
    )(x)

    # Output layer untuk klasifikasi biner
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Pencarian hyperparameter untuk learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def tuner_fn(fn_args):
    """Fungsi utama yang dipanggil oleh komponen Tuner TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Memuat dataset training dan evaluasi
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=5)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5)

    # MENGGUNAKAN RANDOM SEARCH UNTUK MENGHINDARI BUG MAX_TRIALS TFX
    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective='val_accuracy',
        max_trials=5,  # Nilai absolut agar TFX tidak error
        directory=fn_args.working_dir,
        project_name='kt_random_search'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )