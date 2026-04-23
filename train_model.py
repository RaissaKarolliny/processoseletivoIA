import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

#carregando os dados
mnist = keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Ajustando as imagens
# Deixando os pixels entre 0 e 1 
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0 
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

#Criando o modelo
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Treinamento
history = model.fit(x_train, y_train, epochs=3, batch_size=16, validation_split=0.1)

#Avaliando os resultados
# Pegando as métricas finais de treino e validação
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

# Calculando Precision, Recall e F1-score
# Fazendo as predições do teste
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = tf.math.confusion_matrix(y_test, y_pred).numpy()

#Métricas individuais por classe
diag = np.diag(cm)
precision_per_class = np.divide(diag, np.sum(cm, axis=0), out=np.zeros_like(diag, dtype=float), where=np.sum(cm, axis=0)!=0)
recall_per_class = np.divide(diag, np.sum(cm, axis=1), out=np.zeros_like(diag, dtype=float), where=np.sum(cm, axis=1)!=0)
f1_per_class = np.divide(2 * precision_per_class * recall_per_class, precision_per_class + recall_per_class, 
                         out=np.zeros_like(diag, dtype=float), where=(precision_per_class + recall_per_class)!=0)

#Tirando a média (Macro-average) de todas as classes
precision = np.mean(precision_per_class)
recall = np.mean(recall_per_class)
f1 = np.mean(f1_per_class)

print("\n" + "="*20)
print("RELATORIO DE DESEMPENHO")
print("="*20)
print(f"Acuracia de Treino:    {train_acc:.4f}")
print(f"Acuracia de Validacao: {val_acc:.4f}")
print(f"Acuracia de Teste:     {test_acc:.4f}")
print(f"Precision (Macro):     {precision:.4f}")
print(f"Recall (Macro):        {recall:.4f}")
print(f"F1-score (Macro):      {f1:.4f}")
print(f"Perda:                 {loss:.4f}")
print("="*30)

#salvando
#Guardando o modelo no formato h5 pra usar depois na otimização
model.save('model.h5')