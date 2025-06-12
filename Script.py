# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import datetime as dt

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS. CASO PREDICTIVO DE VENTAS*")

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
# Cargar el dataset (Para mermas es posible transformar a CSV o modificar script para conectarse a la base de datos. Queda a elecccion del grupo)
data = pd.read_excel("C:/Users/scist/Downloads/U/inteligencia de negocios/scripts/mermas_actividad_unidad_2.xlsx")


# Convertir fechas a formato datetime con formato día/mes/año
data['fecha'] = pd.to_datetime(data['fecha'])
data['mes'] = data['fecha'].dt.month
data['año'] = data['fecha'].dt.year
data = data[(data['merma_unidad_p'] > -2) & (data['merma_unidad_p'] < 2)]

# Convertir solo los valores decimales de merma_unidad a enteros
data['merma_unidad'] = data['merma_unidad'].apply(lambda x: int(x) if not float(x).is_integer() else x)


# Crear nuevas características para las fechas
#data['Order Year'] = data['Order Date'].dt.year
#data['Order Month'] = data['Order Date'].dt.month

# PASO 3: SELECCIÓN DE CARACTERÍSTICAS
# Características elegidas desde el dataset de mermas reales
features = [
    'codigo_producto', 'negocio', 'seccion', 'linea',
    'categoria', 'abastecimiento', 'comuna', 'region',
    'tienda', 'zonal', 'mes', 'año'
]

X = data[features]
y = data['merma_unidad_p']  # O cambia a 'merma_monto' si quieres predecir pérdida en dinero


# PASO 4: DIVISIÓN DE DATOS
# 80% entrenamiento, 20% prueba. Este porcentaje es el habitual en la literatura para este tipo de modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 5: PREPROCESAMIENTO
# Definir qué variables son categóricas y numéricas
categorical_features = [
    'negocio', 'seccion', 'linea', 'categoria', 'abastecimiento',
    'comuna', 'region', 'tienda', 'zonal'
]

numeric_features = ['mes', 'año']

# Crear preprocesador para manejar ambos tipos de variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# PASO 6: IMPLEMENTACIÓN DE MODELOS
# Modelo 1: Regresión Lineal. Este modelo es el habitual para este tipo de problemas debido a su simplicidad y interpretabilidad.
# En caso de mermas, es posible utilizar este modelo pero pueden explorar otros modelos mas eficientes.
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Modelo 2: Random Forest
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Modelo 3: XGBoost
pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    ))
])


# PASO 7: ENTRENAMIENTO DE MODELOS
# Entrenamos ambos modelos
print("Entrenando Regresión Lineal...")
pipeline_lr.fit(X_train, y_train)

print("Entrenando Random Forest...")
pipeline_rf.fit(X_train, y_train)

print("Entrenando XGBoost...")
pipeline_xgb.fit(X_train, y_train)

y_pred_xgb = pipeline_xgb.predict(X_test)

print("Modelos entrenados correctamente")

# -------------------------------------------------
# EVALUACIÓN DE LOS MODELOS
# -------------------------------------------------

print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")

# PASO 8: REALIZAR PREDICCIONES CON LOS MODELOS ENTRENADOS
y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_xgb = pipeline_xgb.predict(X_test)

# PASO 9: CALCULAR MÚLTIPLES MÉTRICAS DE EVALUACIÓN
# Error Cuadrático Medio (MSE)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb) 

# Raíz del Error Cuadrático Medio (RMSE)
rmse_lr = np.sqrt(mse_lr)
rmse_rf = np.sqrt(mse_rf)
rmse_xgb = np.sqrt(mse_xgb)

# Error Absoluto Medio (MAE)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# Coeficiente de Determinación (R²)
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb) 


# NUEVO PASO: GUARDAR RESULTADOS DE PREDICCIÓN EN ARCHIVOS MARKDOWN
# Crear un DataFrame con las predicciones y valores reales
results_df = pd.DataFrame({
    'Valor_Real': y_test,
    'Prediccion_LR': y_pred_lr,
    'Prediccion_RF': y_pred_rf,
    'Error_LR': y_test - y_pred_lr,
    'Error_RF': y_test - y_pred_rf,
    'Error_Porcentual_LR': ((y_test - y_pred_lr) / y_test) * 100,
    'Error_Porcentual_RF': ((y_test - y_pred_rf) / y_test) * 100
})

# Reiniciar el índice para añadir información de las características
results_df = results_df.reset_index(drop=True)

# Añadir algunas columnas con información de las características para mayor contexto
X_test_reset = X_test.reset_index(drop=True)
for feature in X_test.columns:
    results_df[feature] = X_test_reset[feature]

# Ordenar por valor real para facilitar la comparación
results_df = results_df.sort_values('Valor_Real', ascending=False)

# Guardar resultado para Regresión Lineal
with open('prediccion_lr.md', 'w') as f:
    f.write('# Resultados de Predicción: Regresión Lineal\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_lr:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_lr:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_lr:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Regresión Lineal explica aproximadamente el {r2_lr*100:.1f}% de la variabilidad en las ventas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_lr:.2f} unidades.\n\n')
    
    # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Región |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_LR']:.2f} | {row['Error_LR']:.2f} | {row['Error_Porcentual_LR']:.1f}% | {row['categoria']} | {row['region']} |\n")

    
    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_LR"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_LR"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_LR"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_LR"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

# Guardar resultado para Random Forest
with open('prediccion_rf.md', 'w') as f:
    f.write('# Resultados de Predicción: Random Forest\n\n')
    
    # Añadir resumen de métricas
    f.write('## Resumen de Métricas\n\n')
    f.write(f'- **R²**: {r2_rf:.4f} (Proporción de varianza explicada por el modelo)\n')
    f.write(f'- **RMSE**: {rmse_rf:.2f} (Error cuadrático medio, en unidades de la variable objetivo)\n')
    f.write(f'- **MAE**: {mae_rf:.2f} (Error absoluto medio, en unidades de la variable objetivo)\n\n')
    
    # Añadir interpretación
    f.write('## Interpretación\n\n')
    f.write(f'El modelo de Random Forest explica aproximadamente el {r2_rf*100:.1f}% de la variabilidad en las ventas. ')
    f.write(f'En promedio, las predicciones difieren de los valores reales en ±{rmse_rf:.2f} unidades.\n\n')
    
    # Mostrar muestra de predicciones (top 10)
    f.write('## Muestra de Predicciones (Top 10)\n\n')
    f.write('| # | Valor Real | Predicción | Error | Error % | Categoría | Región |\n')
    f.write('|---|------------|------------|-------|---------|-----------|--------|\n')
    
    for i, row in results_df.head(10).iterrows():
        f.write(f"| {i} | {row['Valor_Real']:.2f} | {row['Prediccion_RF']:.2f} | {row['Error_RF']:.2f} | {row['Error_Porcentual_RF']:.1f}% | {row['categoria']} | {row['region']} |\n")

    # Estadísticas de error
    f.write('\n## Distribución del Error\n\n')
    f.write(f'- **Error Mínimo**: {results_df["Error_RF"].min():.2f}\n')
    f.write(f'- **Error Máximo**: {results_df["Error_RF"].max():.2f}\n')
    f.write(f'- **Error Promedio**: {results_df["Error_RF"].mean():.2f}\n')
    f.write(f'- **Desviación Estándar del Error**: {results_df["Error_RF"].std():.2f}\n\n')
    
    f.write('*Nota: Un error negativo indica que el modelo sobrestimó el valor real, mientras que un error positivo indica una subestimación.*\n')

print("Archivos de predicción generados: prediccion_lr.md y prediccion_rf.md")

# PASO 10: PRESENTAR RESULTADOS DE LAS MÉTRICAS EN FORMATO TABULAR
metrics_df = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Random Forest', 'XGBoost'],
    'MSE': [mse_lr, mse_rf, mse_xgb],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'MAE': [mae_lr, mae_rf, mae_xgb],
    'R²': [r2_lr, r2_rf, r2_xgb]
})

print("\nComparación de métricas entre modelos:")
print(metrics_df)

# VISUALIZACIÓN DEL MODELO XGBOOST
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones XGBoost')
plt.title('XGBoost: Predicciones vs Valores Reales')
plt.savefig('xgboost_predicciones_vs_reales.png')
plt.show()

# PASO 11: VISUALIZACIÓN DE PREDICCIONES VS VALORES REALES
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Random Forest: Predicciones vs Valores Reales')
plt.savefig('predicciones_vs_reales.png')
print("\nGráfico guardado: predicciones_vs_reales.png")

# PASO 12: VISUALIZACIÓN DE RESIDUOS PARA EVALUAR CALIDAD DEL MODELO
residuals = y_test - y_pred_rf
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - Random Forest')
plt.savefig('analisis_residuos.png')
print("Gráfico guardado: analisis_residuos.png")

# PASO 13: DISTRIBUCIÓN DE ERRORES
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - Random Forest')
plt.xlabel('Error')
plt.savefig('distribucion_errores.png')
print("Gráfico guardado: distribucion_errores.png")

# PREDICCIONES VS VALORES REALES - XGBoost
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('XGBoost: Predicciones vs Valores Reales')
plt.grid(True)
plt.savefig('xgb_predicciones_vs_reales.png')

# RESIDUOS - XGBoost
residuals_xgb = y_test - y_pred_xgb
plt.figure(figsize=(12, 6))
plt.scatter(y_pred_xgb, residuals_xgb, alpha=0.5, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos - XGBoost')
plt.grid(True)
plt.savefig('xgb_analisis_residuos.png')

# DISTRIBUCIÓN DE ERRORES - XGBoost
plt.figure(figsize=(10, 6))
sns.histplot(residuals_xgb, kde=True, color='blue')
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribución de Errores - XGBoost')
plt.xlabel('Error')
plt.grid(True)
plt.savefig('xgb_distribucion_errores.png')

# -------------------------------------------------
# DOCUMENTACIÓN DEL PROCESO
# -------------------------------------------------

print("\n=== DOCUMENTACIÓN DEL PROCESO ===")

# PASO 14: DOCUMENTAR LA EXPLORACIÓN INICIAL DE DATOS
print(f"Dimensiones del dataset: {data.shape[0]} filas x {data.shape[1]} columnas")
print(f"Período de tiempo analizado: de {data['fecha'].min().date()} a {data['fecha'].max().date()}")
print(f"Tipos de datos en las columnas principales:")
#aca se puedn agregar las variable segun los datos que quier obtener
print(data[features + ['merma_unidad_p', 'merma_monto', 'motivo']].dtypes)


# PASO 15: DOCUMENTAR EL PREPROCESAMIENTO
print("\n--- PREPROCESAMIENTO APLICADO ---")
print(f"Variables numéricas: {numeric_features}")
print(f"Variables categóricas: {categorical_features}")
print("Transformaciones aplicadas:")
print("- Variables numéricas: Estandarización")
print("- Variables categóricas: One-Hot Encoding")

# PASO 16: DOCUMENTAR LA DIVISIÓN DE DATOS
print("\n--- DIVISIÓN DE DATOS ---")
print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/data.shape[0]:.1%} del total)")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/data.shape[0]:.1%} del total)")
print(f"Método de división: Aleatoria con random_state=42")

# PASO 17: DOCUMENTAR LOS MODELOS EVALUADOS
print("\n--- MODELOS IMPLEMENTADOS ---")
print("1. Regresión Lineal:")
print("   - Ventajas: Simple, interpretable")
print("   - Limitaciones: Asume relación lineal entre variables")

print("\n2. Random Forest Regressor:")
print("   - Hiperparámetros: n_estimators=100, random_state=42")
print("   - Ventajas: Maneja relaciones no lineales, menor riesgo de overfitting")
print("   - Limitaciones: Menos interpretable, mayor costo computacional")

# PASO 18: DOCUMENTAR LA VALIDACIÓN DEL MODELO
print("\n--- VALIDACIÓN DEL MODELO ---")
print("Método de validación: Evaluación en conjunto de prueba separado")
print("Métricas utilizadas: MSE, RMSE, MAE, R²")

# PASO 19: VISUALIZAR IMPORTANCIA DE CARACTERÍSTICAS
if hasattr(pipeline_rf['regressor'], 'feature_importances_'):
    print("\n--- IMPORTANCIA DE CARACTERÍSTICAS ---")
    # Obtener nombres de características después de one-hot encoding
    preprocessor = pipeline_rf.named_steps['preprocessor']
    cat_cols = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, cat_cols])
    
    # Obtener importancias
    importances = pipeline_rf['regressor'].feature_importances_
    
    # Crear un DataFrame para visualización
    if len(feature_names) == len(importances):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Mostrar las 10 características más importantes
        print(feature_importance.head(10))
        
        # Visualizar
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Características Más Importantes')
        plt.savefig('importancia_caracteristicas.png')
        print("Gráfico guardado: importancia_caracteristicas.png")
    else:
        print("No se pudo visualizar la importancia de características debido a diferencias en la dimensionalidad")

# PASO 20: CONCLUSIÓN
print("\n=== CONCLUSIÓN ===")
print(f"El mejor modelo según R² es: {'Random Forest' if r2_rf > r2_lr else 'Regresión Lineal'}")
print(f"R² del mejor modelo: {max(r2_rf, r2_lr):.4f}")
print(f"RMSE del mejor modelo: {rmse_rf if r2_rf > r2_lr else rmse_lr:.2f}")

# Explicaciones adicionales para facilitar la interpretación
print("\n--- INTERPRETACIÓN DE RESULTADOS ---")
print(f"• R² (Coeficiente de determinación): Valor entre 0 y 1 que indica qué proporción de la variabilidad")
print(f"  en las mermas/ventas es explicada por el modelo. Un valor de {max(r2_rf, r2_lr):.4f} significa que")
print(f"  aproximadamente el {max(r2_rf, r2_lr)*100:.1f}% de la variación puede ser explicada por las variables utilizadas.")

print(f"\n• RMSE (Error cuadrático medio): Representa el error promedio de predicción en las mismas unidades")
print(f"  que la variable objetivo. Un RMSE de {rmse_rf if r2_rf > r2_lr else rmse_lr:.2f} significa que, en promedio,")
print(f"  las predicciones difieren de los valores reales en ±{rmse_rf if r2_rf > r2_lr else rmse_lr:.2f} unidades.")

print(f"\n• {'Random Forest' if r2_rf > r2_lr else 'Regresión Lineal'} es el mejor modelo porque:")
if r2_rf > r2_lr:
    print("  - Captura mejor las relaciones no lineales entre las variables")
    print("  - Tiene mayor capacidad predictiva (R² más alto)")
    print("  - Menor error de predicción (RMSE más bajo)")
else:
    print("  - Ofrece un buen equilibrio entre simplicidad y capacidad predictiva")
    print("  - Es más interpretable que modelos complejos")
    print("  - Presenta un mejor ajuste a los datos en este caso específico")

print("\nEl análisis predictivo ha sido completado exitosamente.")