# Gunakan image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Set environment variable
ENV MODEL_NAME=diabetes-model
ENV PORT=8080

# COPY langsung dari output Trainer (Trial 30) ke folder versi '1' standar TF Serving
COPY ./output/aliffauzan-pipeline/Trainer/model/30/Format-Serving /models/diabetes-model/1

# Reset ENTRYPOINT bawaan image supaya port dinamis Cloud/Railway tidak bentrok
ENTRYPOINT []
CMD ["sh", "-c", "tensorflow_model_server --port=8500 --rest_api_port=${PORT} --model_name=${MODEL_NAME} --model_base_path=/models/${MODEL_NAME}"]