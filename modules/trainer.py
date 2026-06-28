"""
Modul Trainer untuk melatih model klasifikasi Pima Indians Diabetes.
(Dilengkapi fix tf.Module dan DictWrapper copy untuk bug Keras 3)
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
    return model


# --- KELAS SAKTI UNTUK NGE-BYPASS BUG KERAS 3 ---
class ServeModelWrapper(tf.Module):
    def __init__(self, model, tf_transform_output):
        super().__init__()
        self.model = model
        self.tft_layer = tf_transform_output.transform_features_layer()
        
        # FIX: Ambil dict-nya, copy, hapus label, BARU dimasukkan ke self
        raw_spec = tf_transform_output.raw_feature_spec().copy()
        raw_spec.pop(LABEL_KEY) 
        self.raw_feature_spec = raw_spec

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_fn(self, serialized_tf_examples):
        parsed_features = tf.io.parse_example(serialized_tf_examples, self.raw_feature_spec)
        transformed_features = self.tft_layer(parsed_features)
        outputs = self.model(transformed_features)
        return {'outputs': outputs}
# ------------------------------------------------


def run_fn(fn_args: FnArgs):
    """Fungsi utama yang dipanggil oleh komponen Trainer TFX."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

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

    # Eksekusi Wrapper dan Save
    serve_wrapper = ServeModelWrapper(model, tf_transform_output)
    signatures = {'serving_default': serve_wrapper.serve_fn}
    
    tf.saved_model.save(serve_wrapper, fn_args.serving_model_dir, signatures=signatures)