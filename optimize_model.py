import tensorflow as tf
import os

# Carregar modelo treinado
model = tf.keras.models.load_model("model.h5")

# Dynamic Range Quantization
converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(model)
converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_dynamic = converter_dynamic.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_dynamic)

print("Modelo Dynamic Range salvo.")

# Float16 Quantization
converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_float16.target_spec.supported_types = [tf.float16]

tflite_float16 = converter_float16.convert()

with open("model_float16.tflite", "wb") as f:
    f.write(tflite_float16)

print("Modelo Float16 salvo.")

# Comparação de tamanho
size_dynamic = os.path.getsize("model.tflite") / 1024
size_float16 = os.path.getsize("model_float16.tflite") / 1024

print("\n=== COMPARAÇÃO DE TAMANHO ===")
print(f"Dynamic Range: {size_dynamic:.2f} KB")
print(f"Float16:       {size_float16:.2f} KB")