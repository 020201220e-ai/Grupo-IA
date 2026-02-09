import pandas as pd
import numpy as np

print("="*60)
print("SOLUCIÓN TRADICIONAL - Fórmula Programada")
print("="*60)

# Cargar datos
df = pd.read_csv('datos_casas.csv')

# Reglas de negocio programadas manualmente
def calcular_precio_tradicional(area, habitaciones, antiguedad, distancia):
    """
    Fórmula fija basada en reglas de negocio
    """
    precio_base = 0
    precio_por_m2 = 1500
    precio_por_habitacion = 20000
    descuento_antiguedad = 1000
    descuento_distancia = 2000
    
    precio = (
        precio_base +
        area * precio_por_m2 +
        habitaciones * precio_por_habitacion -
        antiguedad * descuento_antiguedad -
        distancia * descuento_distancia
    )
    
    return max(precio, 50000)  # Precio mínimo

# Probar con ejemplos
print("\nEjemplos de predicción:\n")

ejemplos = [
    {'area': 100, 'habitaciones': 3, 'antiguedad': 10, 'distancia': 5},
    {'area': 150, 'habitaciones': 4, 'antiguedad': 5, 'distancia': 10},
    {'area': 80, 'habitaciones': 2, 'antiguedad': 20, 'distancia': 15},
]

for i, ejemplo in enumerate(ejemplos, 1):
    precio_pred = calcular_precio_tradicional(
        ejemplo['area'],
        ejemplo['habitaciones'],
        ejemplo['antiguedad'],
        ejemplo['distancia']
    )
    print(f"Casa {i}:")
    print(f"  Área: {ejemplo['area']} m²")
    print(f"  Habitaciones: {ejemplo['habitaciones']}")
    print(f"  Antigüedad: {ejemplo['antiguedad']} años")
    print(f"  Distancia: {ejemplo['distancia']} km")
    print(f"  → Precio predicho: ${precio_pred:,.0f}")
    print()

# Evaluar en todo el dataset
predicciones = []
for _, row in df.iterrows():
    pred = calcular_precio_tradicional(
        row['area'],
        row['habitaciones'],
        row['antiguedad'],
        row['distancia_centro']
    )
    predicciones.append(pred)

predicciones = np.array(predicciones)
errores = np.abs(df['precio'].values - predicciones)

print("Evaluación en dataset completo:")
print(f"  Error promedio: ${errores.mean():,.0f}")
print(f"  Error máximo: ${errores.max():,.0f}")
print(f"  Error mínimo: ${errores.min():,.0f}")
