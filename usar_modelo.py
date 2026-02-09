import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("="*60)
print("🏠 PREDICTOR DE PRECIOS DE CASAS")
print("   Usando Red Neuronal Entrenada")
print("="*60)

# 1. CARGAR MODELO
print("\n1. Cargando modelo entrenado...")
model = tf.keras.models.load_model('modelo_precios_casas.keras')
print("   ✓ Modelo cargado exitosamente")

# 2. CARGAR DATOS ORIGINALES PARA EL SCALER
df = pd.read_csv('datos_casas.csv')
X = df[['area', 'habitaciones', 'antiguedad', 'distancia_centro']].values
y = df['precio'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X)
scaler_y.fit(y.reshape(-1, 1))

# 3. FUNCIÓN DE PREDICCIÓN
def predecir_precio(area, habitaciones, antiguedad, distancia):
    """Predice el precio de una casa"""
    entrada = np.array([[area, habitaciones, antiguedad, distancia]])
    entrada_scaled = scaler_X.transform(entrada)
    prediccion_scaled = model.predict(entrada_scaled, verbose=0)
    precio = scaler_y.inverse_transform(prediccion_scaled)
    return precio[0][0]

# 4. MENÚ INTERACTIVO
print("\n" + "="*60)
print("OPCIONES:")
print("="*60)
print("\n[1] Predecir precio de una casa personalizada")
print("[2] Ver ejemplos predefinidos")
print("[3] Comparar con método tradicional")

opcion = input("\nElige una opción (1-3): ")

if opcion == "1":
    print("\n" + "="*60)
    print("Ingresa las características de la casa:")
    print("="*60)
    
    area = float(input("Área (m²): "))
    habitaciones = int(input("Número de habitaciones: "))
    antiguedad = float(input("Antigüedad (años): "))
    distancia = float(input("Distancia al centro (km): "))
    
    precio = predecir_precio(area, habitaciones, antiguedad, distancia)
    
    print("\n" + "="*60)
    print("RESULTADO:")
    print("="*60)
    print(f"💰 Precio estimado: ${precio:,.0f}")
    print("="*60)

elif opcion == "2":
    print("\n" + "="*60)
    print("EJEMPLOS DE PREDICCIONES:")
    print("="*60)
    
    ejemplos = [
        {"nombre": "Casa económica", "area": 80, "hab": 2, "ant": 20, "dist": 15},
        {"nombre": "Casa media", "area": 120, "hab": 3, "ant": 10, "dist": 8},
        {"nombre": "Casa premium", "area": 200, "hab": 5, "ant": 2, "dist": 3},
        {"nombre": "Apartamento céntrico", "area": 70, "hab": 2, "ant": 5, "dist": 2},
        {"nombre": "Casa de campo", "area": 150, "hab": 4, "ant": 15, "dist": 25},
    ]
    
    for ej in ejemplos:
        precio = predecir_precio(ej["area"], ej["hab"], ej["ant"], ej["dist"])
        print(f"\n{ej['nombre']}:")
        print(f"  📐 Área: {ej['area']} m²")
        print(f"  🚪 Habitaciones: {ej['hab']}")
        print(f"  📅 Antigüedad: {ej['ant']} años")
        print(f"  📍 Distancia: {ej['dist']} km")
        print(f"  💰 Precio: ${precio:,.0f}")

elif opcion == "3":
    print("\n" + "="*60)
    print("COMPARACIÓN: RED NEURONAL vs MÉTODO TRADICIONAL")
    print("="*60)
    
    # Función tradicional
    def precio_tradicional(area, hab, ant, dist):
        return max(100000 + area*1500 + hab*20000 - ant*1000 - dist*2000, 50000)
    
    ejemplos = [
        [100, 3, 10, 5],
        [150, 4, 5, 10],
        [80, 2, 20, 15],
    ]
    
    for i, ej in enumerate(ejemplos, 1):
        precio_ia = predecir_precio(ej[0], ej[1], ej[2], ej[3])
        precio_trad = precio_tradicional(ej[0], ej[1], ej[2], ej[3])
        diferencia = abs(precio_ia - precio_trad)
        
        print(f"\nCasa {i}: {ej[0]}m², {ej[1]} hab, {ej[2]} años, {ej[3]}km")
        print(f"  🧠 Red Neuronal: ${precio_ia:,.0f}")
        print(f"  📝 Tradicional:  ${precio_trad:,.0f}")
        print(f"  📊 Diferencia:   ${diferencia:,.0f}")

print("\n" + "="*60)
print("✓ Predicción completada")
print("="*60)
