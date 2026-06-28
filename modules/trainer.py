"""
Modul Trainer untuk melatih model klasifikasi Pima Indians Diabetes.
"""

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

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

def model_builder(hyperparameters):
    """Membangun arsitektur model Keras menggunakan hyperparameter terbaik."""
    inputs = {}
    for feature in NUMERIC_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.float32
        )

    concatenated_inputs = tf.keras.layers.Concatenate()(list(inputs.values()))

    # Mengambil nilai hyperparameter dari Tuner, atau pakai default kalau tidak ada
    units_1 = hyperparameters.get('units_1') if hyperparameters else 64
    dropout_1 = hyperparameters.get('dropout_1') if hyperparameters else 0.2
    units_2 = hyperparameters.get('units_2') if hyperparameters else 32
    learning_rate = hyperparameters.get('learning_rate') if hyperparameters else 1e-3

    x = tf.keras.layers.Dense(units=units_1, activation='relu')(concatenated_inputs)
    x = tf.keras.layers.Dropout(rate=dropout_1)(x)
    x = tf.keras.layers.Dense(units=units_2, activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Fungsi untuk melayani (serving) model dengan data raw."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    """Fungsi utama yang dipanggil oleh komponen Trainer TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Load best hyperparameters
    if fn_args.hyperparameters:
        hparams = fn_args.hyperparameters.get('values')
    else:
        hparams = None

    model = model_builder(hparams)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch'
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy',
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping_callback],
        epochs=10
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    # FIX UNTUK KERAS 3: Pakai tf.saved_model.save
    # FIX UNTUK KERAS 3: Ikat fungsi ke dalam object model agar variable tidak terhapus
    model.serve_tf_examples_fn = _get_serve_tf_examples_fn(
        model, tf_transform_output
    ).get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    )

    signatures = {
        'serving_default': model.serve_tf_examples_fn
    }

    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)