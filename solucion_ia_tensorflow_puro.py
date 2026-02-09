import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*60)
print("SOLUCIÓN IA - TensorFlow PURO")
print("="*60)

# 1. CARGAR DATOS
print("\n1. Cargando datos...")
df = pd.read_csv('datos_casas.csv')
X = df[['area', 'habitaciones', 'antiguedad', 'distancia_centro']].values
y = df['precio'].values.reshape(-1, 1)

# 2. DIVIDIR DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. NORMALIZAR
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler_X.transform(X_test).astype(np.float32)
y_train_scaled = scaler_y.fit_transform(y_train).astype(np.float32)
y_test_scaled = scaler_y.transform(y_test).astype(np.float32)

# 4. DEFINIR HIPERPARÁMETROS
n_input = 4      # Características de entrada
n_hidden1 = 64   # Neuronas capa 1
n_hidden2 = 32   # Neuronas capa 2
n_hidden3 = 16   # Neuronas capa 3
n_output = 1     # Salida (precio)
learning_rate = 0.001
epochs = 100
batch_size = 32

# 5. INICIALIZAR PESOS Y SESGOS (Variables de TensorFlow)
print("\n2. Inicializando red neuronal con TensorFlow puro...")

# Inicialización Xavier/Glorot para estabilidad
def xavier_init(shape):
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.Variable(
        tf.random.uniform(shape, -limit, limit, dtype=tf.float32),
        trainable=True
    )

# Pesos (weights) y sesgos (biases) de cada capa
W1 = xavier_init([n_input, n_hidden1])      # Capa 1
b1 = tf.Variable(tf.zeros([n_hidden1], dtype=tf.float32), trainable=True)

W2 = xavier_init([n_hidden1, n_hidden2])    # Capa 2
b2 = tf.Variable(tf.zeros([n_hidden2], dtype=tf.float32), trainable=True)

W3 = xavier_init([n_hidden2, n_hidden3])    # Capa 3
b3 = tf.Variable(tf.zeros([n_hidden3], dtype=tf.float32), trainable=True)

W4 = xavier_init([n_hidden3, n_output])     # Capa de salida
b4 = tf.Variable(tf.zeros([n_output], dtype=tf.float32), trainable=True)

print(f"   Capa 1: {n_input} → {n_hidden1} neuronas")
print(f"   Capa 2: {n_hidden1} → {n_hidden2} neuronas")
print(f"   Capa 3: {n_hidden2} → {n_hidden3} neuronas")
print(f"   Capa 4: {n_hidden3} → {n_output} neurona(s)")

# 6. DEFINIR RED NEURONAL (Forward Pass)
def neural_network(x):
    """
    Propagación hacia adelante (forward propagation)
    usando operaciones puras de TensorFlow
    """
    # Capa 1: X * W1 + b1 → ReLU
    layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    
    # Capa 2: layer1 * W2 + b2 → ReLU
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
    
    # Capa 3: layer2 * W3 + b3 → ReLU
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
    
    # Capa de salida: layer3 * W4 + b4 (sin activación)
    output = tf.matmul(layer3, W4) + b4
    
    return output

# 7. FUNCIÓN DE PÉRDIDA (Mean Squared Error)
def loss_function(y_true, y_pred):
    """Error cuadrático medio"""
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 8. OPTIMIZADOR (Adam)
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# 9. ENTRENAMIENTO
print("\n3. Entrenando red neuronal...")
print(f"   Épocas: {epochs}")
print(f"   Tamaño de batch: {batch_size}")
print(f"   Tasa de aprendizaje: {learning_rate}\n")

n_batches = len(X_train_scaled) // batch_size
variables = [W1, b1, W2, b2, W3, b3, W4, b4]

for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Mezclar datos al inicio de cada época
    indices = np.random.permutation(len(X_train_scaled))
    X_shuffled = X_train_scaled[indices]
    y_shuffled = y_train_scaled[indices]
    
    # Entrenar por mini-batches
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        
        # Calcular gradientes y actualizar pesos
        with tf.GradientTape() as tape:
            predictions = neural_network(X_batch)
            loss = loss_function(y_batch, predictions)
        
        # Backpropagation
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        epoch_loss += loss.numpy()
    
    # Mostrar progreso cada 10 épocas
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / n_batches
        print(f"Época {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

# 10. EVALUACIÓN
print("\n4. Evaluando modelo...")
predictions_test = neural_network(X_test_scaled)
test_loss = loss_function(y_test_scaled, predictions_test)

# Convertir a escala original
predictions_original = scaler_y.inverse_transform(predictions_test.numpy())
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Calcular error absoluto medio
mae = np.mean(np.abs(y_test_original - predictions_original))
print(f"   Error Absoluto Medio: ${mae:,.0f}")

# 11. PREDICCIONES DE EJEMPLO
print("\n5. Realizando predicciones...\n")

ejemplos = np.array([
    [100, 3, 10, 5],
    [150, 4, 5, 10],
    [80, 2, 20, 15],
], dtype=np.float32)

ejemplos_scaled = scaler_X.transform(ejemplos)
predicciones = neural_network(ejemplos_scaled)
predicciones_original = scaler_y.inverse_transform(predicciones.numpy())

nombres = ['area', 'habitaciones', 'antiguedad', 'distancia_centro']
for i, ejemplo in enumerate(ejemplos, 1):
    print(f"Casa {i}:")
    for j, nombre in enumerate(nombres):
        print(f"  {nombre}: {ejemplo[j]}")
    print(f"  → Precio predicho: ${predicciones_original[i-1][0]:,.0f}")
    print()

# 12. GUARDAR PESOS
print("6. Guardando pesos del modelo...")
np.savez('modelo_tensorflow_puro.npz',
         W1=W1.numpy(), b1=b1.numpy(),
         W2=W2.numpy(), b2=b2.numpy(),
         W3=W3.numpy(), b3=b3.numpy(),
         W4=W4.numpy(), b4=b4.numpy())
print("   ✓ Pesos guardados: modelo_tensorflow_puro.npz")

print("\n" + "="*60)
print("¡ENTRENAMIENTO COMPLETADO CON TENSORFLOW PURO!")
print("="*60)
