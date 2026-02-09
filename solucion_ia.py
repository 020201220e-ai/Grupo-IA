import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*60)
print("SOLUCIÓN IA - Red Neuronal con TensorFlow")
print("="*60)

# 1. CARGAR Y PREPARAR DATOS
print("\n1. Cargando datos...")
df = pd.read_csv('datos_casas.csv')

X = df[['area', 'habitaciones', 'antiguedad', 'distancia_centro']].values
y = df['precio'].values

print(f"   Dataset: {X.shape[0]} muestras, {X.shape[1]} características")

# 2. DIVIDIR DATOS
print("\n2. Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Entrenamiento: {X_train.shape[0]} muestras")
print(f"   Prueba: {X_test.shape[0]} muestras")

# 3. NORMALIZAR DATOS
print("\n3. Normalizando datos...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 4. CREAR RED NEURONAL
print("\n4. Creando red neuronal...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu', name='capa_oculta_1'),
    tf.keras.layers.Dense(32, activation='relu', name='capa_oculta_2'),
    tf.keras.layers.Dense(16, activation='relu', name='capa_oculta_3'),
    tf.keras.layers.Dense(1, name='capa_salida')
])

print("\nArquitectura de la red:")
model.summary()

# 5. COMPILAR MODELO
print("\n5. Compilando modelo...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 6. ENTRENAR
print("\n6. Entrenando red neuronal...")
print("   (Esto puede tomar un momento...)\n")

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 7. EVALUAR
print("\n7. Evaluando modelo...")
loss, mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
mae_real = mae * scaler_y.scale_[0]  # Convertir a escala original

print(f"\n   Error Absoluto Medio: ${mae_real:,.0f}")

# 8. PREDICCIONES
print("\n8. Realizando predicciones de ejemplo...\n")

ejemplos = np.array([
    [100, 3, 10, 5],   # Casa 1
    [150, 4, 5, 10],   # Casa 2
    [80, 2, 20, 15],   # Casa 3
])

ejemplos_scaled = scaler_X.transform(ejemplos)
predicciones_scaled = model.predict(ejemplos_scaled, verbose=0)
predicciones = scaler_y.inverse_transform(predicciones_scaled)

nombres = ['area', 'habitaciones', 'antiguedad', 'distancia_centro']
for i, ejemplo in enumerate(ejemplos, 1):
    print(f"Casa {i}:")
    for j, nombre in enumerate(nombres):
        print(f"  {nombre}: {ejemplo[j]}")
    print(f"  → Precio predicho: ${predicciones[i-1][0]:,.0f}")
    print()

# 9. GUARDAR MODELO
print("9. Guardando modelo...")
model.save('modelo_precios_casas.keras')
print("   ✓ Modelo guardado: modelo_precios_casas.keras")

# 10. GRAFICAR RESULTADOS
print("\n10. Generando gráficas...")

plt.figure(figsize=(12, 4))

# Gráfica 1: Loss durante entrenamiento
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Gráfica 2: Predicciones vs Reales
plt.subplot(1, 2, 2)
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Precio Real ($)')
plt.ylabel('Precio Predicho ($)')
plt.grid(True)

plt.tight_layout()
plt.savefig('resultados.png', dpi=100)
print("   ✓ Gráfica guardada: resultados.png")

print("\n" + "="*60)
print("¡ENTRENAMIENTO COMPLETADO!")
print("="*60)
