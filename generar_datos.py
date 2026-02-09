import numpy as np
import pandas as pd

# Generar datos sintéticos de casas
np.random.seed(42)
n_samples = 1000

# Features (características)
area = np.random.uniform(50, 300, n_samples)
habitaciones = np.random.randint(1, 6, n_samples)
antiguedad = np.random.uniform(0, 50, n_samples)
distancia_centro = np.random.uniform(1, 30, n_samples)

# Target (precio) - fórmula simulada
precio = (
    area * 1500 +
    habitaciones * 20000 +
    -antiguedad * 1000 +
    -distancia_centro * 2000 +
    np.random.normal(0, 15000, n_samples)
)

# Crear DataFrame
df = pd.DataFrame({
    'area': area,
    'habitaciones': habitaciones,
    'antiguedad': antiguedad,
    'distancia_centro': distancia_centro,
    'precio': precio
})

# Guardar a CSV
df.to_csv('datos_casas.csv', index=False)
print("✓ Dataset generado: datos_casas.csv")
print(f"  Total de muestras: {n_samples}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nEstadísticas:")
print(df.describe())
