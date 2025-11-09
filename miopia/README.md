# Clasificación Jerárquica de Miopía - Documentación Completa

## Índice
- [Descripción General](#descripción-general)
- [Requisitos](#requisitos)
- [Estructura del Notebook](#estructura-del-notebook)
- [Parte 1: Entrenamiento de Modelos](#parte-1-entrenamiento-de-modelos)
- [Parte 2: Estrategias Jerárquicas](#parte-2-estrategias-jerárquicas)
- [Funciones Principales](#funciones-principales)
- [Archivos Generados](#archivos-generados)
- [Resultados y Conclusiones](#resultados-y-conclusiones)

---

## Descripción General

Este notebook implementa y compara **tres estrategias diferentes** para clasificación jerárquica de miopía:

1. **Top-Down**: Predice directamente las 4 clases (C, M1, M2, MM) y mapea a la jerarquía
2. **Bottom-Up**: Clasificación en cascada (M → MM → M1/M2)
3. **Hybrid**: Combina Top-Down con validación Bottom-Up

### Jerarquía de Clases:
```
DCombo (4 clases)
├── C  (Control - sin miopía)
├── M1 (Miopía tipo 1)
├── M2 (Miopía tipo 2)
└── MM (Miopía Magna)

Mapeo Jerárquico:
- DCombo → Combo (3 clases: C, M, MM)
- DCombo → M (2 clases: NO, SI)
- DCombo → MM (2 clases: NO, SI)
```

---

## Requisitos

### Software
- Python 3.8+
- Jupyter Notebook / VS Code

### Librerías
```python
pandas          # Manipulación de datos
numpy           # Operaciones numéricas
scikit-learn    # Modelos ML y evaluación
xgboost         # Gradient Boosting (opcional)
```

### Archivos de Entrada
- `X_train.csv` - Features de entrenamiento (132 muestras)
- `Y_train.csv` - Labels jerárquicos (M, MM, Combo, DCombo)

---

## Estructura del Notebook

El notebook está dividido en **2 partes principales** y **46 celdas**:

### **PARTE 1: Entrenamiento de Modelos** (Secciones 1-10)
Entrenamiento y selección del mejor modelo para DCombo

### **PARTE 2: Estrategias Jerárquicas** (Sección 11)
Comparación de 3 estrategias de clasificación jerárquica

---

# PARTE 1: Entrenamiento de Modelos

## Sección 1: Imports and Setup

### Propósito
Importar todas las librerías necesarias y configurar el entorno.

### Componentes Clave
```python
RANDOM_STATE = 42  # Reproducibilidad
XGBOOST_AVAILABLE  # Flag para XGBoost
```

### Librerías Importadas
- **Pandas/Numpy**: Manipulación de datos
- **Sklearn**: Pipeline, transformers, modelos, métricas
- **XGBoost**: Gradient boosting (si disponible)

---

## Sección 2: Load Data

### Propósito
Cargar los datos de entrenamiento y preparar el target.

### Proceso
1. Carga `X_train.csv` y `Y_train.csv`
2. **NO elimina** 'fecha' ni columnas con NaN (se hace en `build_preprocessor`)
3. Define `y_train = DCombo` como target principal

### Salida
```
✓ Data loaded: 132 samples, 47 features (antes de limpieza)
✓ Target classes: ['C', 'M1', 'M2', 'MM']
```

---

## Sección 3: Helper Functions

### Función 1: `build_preprocessor(X, nan_threshold=0.20)`

#### Propósito
Construir un preprocesador que limpia datos y transforma features.

#### Parámetros
- `X`: DataFrame de features
- `nan_threshold`: Umbral máximo de NaN permitido (default 20%)

#### Proceso Paso a Paso

**1. Limpieza de Datos**
```python
# Eliminar columna 'fecha' si existe
if 'fecha' in X_clean.columns:
    X_clean = X_clean.drop(columns=['fecha'])

# Eliminar columnas con >20% NaN
nan_percentage = X_clean.isna().sum() / len(X_clean)
cols_to_drop = nan_percentage[nan_percentage > nan_threshold]
```

**2. Detección Automática de Tipos**
```python
numeric_features = X.select_dtypes(include=['int64', 'float64'])
categorical_features = X.select_dtypes(include=['object', 'category'])
```

**3. Construcción de Transformadores**

- **Numéricas**: 
  - `SimpleImputer(strategy='median')` → Rellena NaN con mediana
  - `StandardScaler(with_mean=False)` → Normaliza sin centrar (compatible con sparse)

- **Categóricas**:
  - `SimpleImputer(strategy='most_frequent')` → Rellena NaN con moda
  - `OneHotEncoder(handle_unknown='ignore')` → Codificación one-hot

**4. ColumnTransformer**
```python
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')
```

#### Retorna
- `(preprocessor, X_clean)` → Tuple con preprocesador y datos limpios

---

### Función 2: `build_models(preprocessor)`

#### Propósito
Crear diccionario de modelos con sus pipelines y grids de hiperparámetros.

#### Modelos Configurados

**1. Random Forest**
```python
rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}
```

**2. XGBoost** (si disponible)
```python
xgb_param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0]
}
```
 **Requiere label encoding** (strings → integers)

**3. Multinomial Naive Bayes**
```python
nb_param_grid = {
    'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
}
```

#### Retorna
```python
{
    'model_name': (pipeline, param_grid, needs_label_encoding),
    ...
}
```

---

### Función 3: `evaluate_with_cv(model, X, y, cv, scoring)`

#### Propósito
Evaluar modelo usando cross-validation con múltiples métricas.

#### Proceso
```python
cv_results = cross_validate(
    model, X, y,
    cv=cv,                    # StratifiedKFold
    scoring=scoring,          # Dict con 4 métricas
    return_train_score=False,
    n_jobs=-1                 # Paralelización
)
```

#### Métricas Calculadas
- `accuracy`: Exactitud general
- `f1_macro`: F1 promedio de todas las clases
- `precision_macro`: Precisión promedio
- `recall_macro`: Recall promedio

---

### Función 4: `fit_and_report(model_name, pipeline, param_grid, X, y, cv, scoring, needs_label_encoding)`

#### Propósito
Entrenar modelo con búsqueda de hiperparámetros y generar reportes.

#### Proceso Detallado

**1. Label Encoding (si necesario)**
```python
if needs_label_encoding:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
```

**2. Búsqueda de Hiperparámetros**
```python
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=20,              # 20 combinaciones aleatorias
    cv=cv,
    scoring='f1_macro',     # Métrica de optimización
    n_jobs=-1,
    random_state=42
)
search.fit(X, y_encoded)
```

**3. Evaluación con Cross-Validation**
```python
cv_results = evaluate_with_cv(best_pipeline, X, y_encoded, cv, scoring)
```

**4. Generación de DataFrame de Resultados**
```python
cv_results_df = pd.DataFrame({
    'fold': [1, 2, 3, 4, 5],
    'accuracy': [...],
    'f1_macro': [...],
    'precision_macro': [...],
    'recall_macro': [...]
})
```

**5. Guardar Resultados**
```python
cv_results_df.to_csv(f'cv_results_{model_name}.csv', index=False)
```

#### Retorna
```python
(best_pipeline, cv_results_df, mean_f1, label_encoder)
```

---

### Función 5: `create_labels_mapping(y)`

#### Propósito
Crear tabla de mapeo de clases con soporte.

#### Proceso
```python
labels_df = pd.DataFrame({
    'label_index': [0, 1, 2, 3],
    'label_name': ['C', 'M1', 'M2', 'MM'],
    'support': [44, 39, 31, 18]  # Conteo de muestras
})
labels_df.to_csv('labels_mapping.csv')
```

---

### Función 6: `generate_best_model_reports(best_pipeline, model_name, X, y, cv, label_encoder)`

#### Propósito
Generar reportes detallados del mejor modelo.

#### Reportes Generados

**1. Confusion Matrix**
```python
y_pred = cross_val_predict(best_pipeline, X, y, cv=cv)
cm = confusion_matrix(y, y_pred)
# Guardado en confusion_matrix.csv
```

**2. Classification Report**
```python
report = classification_report(y, y_pred, output_dict=True)
# Guardado en classification_report.csv
```

**3. Feature Importances** (si aplicable)
```python
if hasattr(classifier, 'feature_importances_'):
    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    # Guardado en feature_importances.csv
```

---

## Sección 4: Build Preprocessor

### Propósito
Construir el preprocesador y limpiar `X_train`.

### Proceso
```python
preprocessor, X_train = build_preprocessor(X_train, nan_threshold=0.20)
```

### Salida Esperada
```
  → Eliminando columna 'fecha' (no aporta información predictiva)
  → Eliminando X columnas con >20% NaN:
      • columna1: 25.0% NaN
      • columna2: 30.5% NaN
  Features finales: 45 columnas
    - Numéricas: 30
    - Categóricas: 15
✓ Features después de limpieza: 45 columnas
```

---

## Sección 5: Define Cross-Validation Strategy

### Propósito
Configurar estrategia de validación cruzada y métricas.

### Configuración
```python
cv = StratifiedKFold(
    n_splits=5,           # 5 folds
    shuffle=True,         # Mezclar datos
    random_state=42       # Reproducibilidad
)

scoring = {
    'accuracy': accuracy_score,
    'f1_macro': f1_score (average='macro'),
    'precision_macro': precision_score (average='macro'),
    'recall_macro': recall_score (average='macro')
}
```

### ¿Por qué StratifiedKFold?
Mantiene la **proporción de clases** en cada fold:
```
Fold 1: 20% C, 30% M1, 23% M2, 14% MM
Fold 2: 20% C, 30% M1, 23% M2, 14% MM
...
```

---

## Sección 6: Build and Train Models

### Celda 1: Build Models
```python
models = build_models(preprocessor)
# Retorna: {'random_forest': (...), 'xgb': (...), 'nb': (...)}
```

### Celda 2: Train All Models
```python
results = {}
for model_name, (pipeline, param_grid, needs_encoding) in models.items():
    best_pipe, cv_df, mean_f1, label_enc = fit_and_report(...)
    results[model_name] = {
        'pipeline': best_pipe,
        'cv_results': cv_df,
        'mean_f1_macro': mean_f1,
        'label_encoder': label_enc
    }
```

### Proceso por Modelo
1. **RandomizedSearchCV** → Busca 20 combinaciones de hiperparámetros
2. **Cross-validation** → Evalúa mejor modelo en 5 folds
3. **Guarda resultados** → `cv_results_{model}.csv`

---

## Sección 7: Compare Models and Select Best

### Propósito
Comparar todos los modelos y seleccionar el mejor según F1-macro.

### Tabla de Comparación
```
Model           Accuracy         F1-Macro         Precision        Recall
random_forest   0.7200 ± 0.05   0.6800 ± 0.06   0.7000 ± 0.05   0.6950 ± 0.04
xgb             0.7100 ± 0.06   0.6700 ± 0.07   0.6900 ± 0.06   0.6850 ± 0.05
nb              0.6500 ± 0.08   0.6000 ± 0.09   0.6200 ± 0.08   0.6100 ± 0.07
```

### Selección del Mejor
```python
best_model_name = max(results.items(), 
                     key=lambda x: x[1]['mean_f1_macro'])[0]
# Típicamente: 'random_forest'
```

---

## Sección 8: Generate Labels Mapping

### Propósito
Crear y guardar mapeo de etiquetas.

### Archivo Generado: `labels_mapping.csv`
```csv
label_index,label_name,support
0,C,44
1,M1,39
2,M2,31
3,MM,18
```

---

## Sección 9: Generate Reports for Best Model

### Archivos Generados

**1. confusion_matrix.csv**
```
     C   M1  M2  MM
C    40   2   1   1
M1    5  30   3   1
M2    2   4  23   2
MM    1   1   2  14
```

**2. classification_report.csv**
```
          precision  recall  f1-score  support
C         0.83      0.91    0.87      44
M1        0.81      0.77    0.79      39
M2        0.79      0.74    0.77      31
MM        0.78      0.78    0.78      18
```

**3. feature_importances.csv** (si Random Forest)
```
feature                     importance
num__edad                   0.0850
num__refraccion_esferica    0.0720
cat__sexo_M                 0.0650
...
```

---

##  Sección 10: Summary

### Archivos Generados (Parte 1)
```
✓ labels_mapping.csv
✓ cv_results_random_forest.csv
✓ cv_results_xgb.csv
✓ cv_results_nb.csv
✓ confusion_matrix.csv
✓ classification_report.csv
✓ feature_importances.csv
```

---

# Clasificación Jerárquica de Miopía - Pipeline Completo con 3 Estrategias

## Sección 11: Comparación de Estrategias

### Subsección 11.1: Preparación de Datos

#### Propósito
Preparar datos completos con todas las columnas jerárquicas.

#### Diccionario de Datos
```python
y_dict_full = {
    'M': ['NO', 'SI', 'SI', ...],      # 132 valores
    'MM': ['NO', 'NO', 'SI', ...],      # 132 valores
    'Combo': ['C', 'M', 'MM', ...],     # 132 valores
    'DCombo': ['C', 'M1', 'M2', 'MM']   # 132 valores
}
```

#### Distribución de Clases
```
DCombo: C=44, M1=39, M2=31, MM=18
Combo:  C=44, M=70, MM=18
M:      NO=44, SI=88
MM:     NO=122, SI=10
```

---

### Funciones Auxiliares

#### Función: `dcombo_to_hierarchical(dcombo_values)`

**Propósito**: Mapear DCombo a las otras columnas jerárquicas.

**Reglas de Mapeo**:
```python
C  → M=NO,  MM=NO,  Combo=C
M1 → M=SI,  MM=NO,  Combo=M
M2 → M=SI,  MM=NO,  Combo=M
MM → M=SI,  MM=SI,  Combo=MM
```

**Implementación**:
```python
M = np.where(dcombo == 'C', 'NO', 'SI')
MM = np.where(dcombo == 'MM', 'SI', 'NO')
Combo = np.where(dcombo == 'C', 'C',
                np.where(dcombo == 'MM', 'MM', 'M'))
```

**Retorna**: `(M, MM, Combo)`

---

#### Función: `evaluate_hierarchical_predictions(y_true_dict, y_pred_dict, strategy_name)`

**Propósito**: Evaluar predicciones en todas las columnas jerárquicas.

**Proceso**:
```python
for col in ['DCombo', 'Combo', 'M', 'MM']:
    y_true = y_true_dict[col]
    y_pred = y_pred_dict[col]
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    results[col] = {'accuracy': acc, 'f1_macro': f1_macro}
    
    # Imprime classification_report completo
```

**Retorna**: 
```python
{
    'DCombo': {'accuracy': 0.60, 'f1_macro': 0.64},
    'Combo': {'accuracy': 0.68, 'f1_macro': 0.71},
    'M': {'accuracy': 0.72, 'f1_macro': 0.69},
    'MM': {'accuracy': 0.90, 'f1_macro': 0.73}
}
```

---

### Subsección 11.2: Estrategia 1 - TOP-DOWN

#### Concepto
**"Predecir lo complejo primero, derivar lo simple"**

#### Proceso

**1. Predicción de DCombo**
```python
y_pred_dcombo = cross_val_predict(
    best_pipeline,      # Modelo entrenado en secciones 1-10
    X_train,
    y_train_dcombo,
    cv=cv,              # 5-fold stratified
    method='predict'
)
```

**IMPORTANTE**: Usa `cross_val_predict` → predicciones **out-of-fold** (sin overfitting)

**2. Mapeo Jerárquico**
```python
M_pred, MM_pred, Combo_pred = dcombo_to_hierarchical(y_pred_dcombo)
```

**3. Resultado**
```python
y_pred_topdown = {
    'DCombo': ['C', 'M1', 'M2', 'MM', ...],
    'Combo': ['C', 'M', 'M', 'MM', ...],
    'M': ['NO', 'SI', 'SI', 'SI', ...],
    'MM': ['NO', 'NO', 'NO', 'SI', ...]
}
```

#### Ventajas
- Simple (1 solo modelo)  
- Coherencia jerárquica garantizada  
- Rápido de entrenar e implementar

#### Desventajas
- Problema de 4 clases (más difícil)  
- No se especializa por nivel

---

### Subsección 11.2.1: Guardar Modelo Top-Down

#### Propósito
Guardar el mejor modelo de la Parte 1 y su configuración para uso posterior.

#### Archivos Guardados

```python
import joblib
from pathlib import Path

models_dir = Path('models')

# Guardar el mejor pipeline (DCombo directo)
joblib.dump(best_pipeline, models_dir / 'topdown_dcombo.pkl')

# Guardar label encoder si existe
if best_label_encoder is not None:
    joblib.dump(best_label_encoder, models_dir / 'topdown_label_encoder.pkl')

# Guardar información del modelo
topdown_info = {
    'model_name': best_model_name,
    'best_f1_macro': best_f1,
    'uses_label_encoding': best_label_encoder is not None,
    'classes': sorted(np.unique(y_train))
}
joblib.dump(topdown_info, models_dir / 'topdown_info.pkl')
```

#### Cargar Modelo Posteriormente

```python
import joblib

# Cargar modelo
model_topdown = joblib.load('models/topdown_dcombo.pkl')

# Cargar info
topdown_info = joblib.load('models/topdown_info.pkl')
print(f"Modelo: {topdown_info['model_name']}")
print(f"F1-Macro: {topdown_info['best_f1_macro']:.4f}")

# Cargar label encoder si existe
if topdown_info['uses_label_encoding']:
    label_encoder = joblib.load('models/topdown_label_encoder.pkl')
```

---

### Subsección 11.3: Estrategia 2 - BOTTOM-UP OPTIMIZADA (Cascada con GridSearchCV)

#### Concepto
**"Dividir y conquistar: decisiones simples en cascada con optimización independiente"**

#### Optimización con GridSearchCV
Cada nivel de la cascada optimiza sus hiperparámetros **independientemente** usando GridSearchCV, con especial enfoque en mejorar la clasificación M1 vs M2.

#### Arquitectura de 3 Niveles

```
┌─────────────────────────────────────────────┐
│ NIVEL 1: Clasificador M (NO/SI)            │
│ GridSearchCV → 160 combinaciones            │
│ Best: n_estimators=150, max_depth=12       │
│ F1-Macro: 0.7618                           │
└──────────┬──────────────────────────────────┘
           │
           ├─ Si M=NO → DCombo='C'
           │
           └─ Si M=SI ↓
              ┌───────────────────────────────────────┐
              │ NIVEL 2: Clasificador MM (NO/SI)     │
              │ GridSearchCV → 160 combinaciones      │
              │ Best: n_estimators=100, max_depth=5  │
              │ F1-Macro: 0.4699                     │
              └──────────┬────────────────────────────┘
                         │
                         ├─ Si MM=SI → DCombo='MM'
                         │
                         └─ Si MM=NO ↓
                            ┌─────────────────────────────────────────┐
                            │ NIVEL 3: Clasificador M1 vs M2 ⚡      │
                            │ GridSearchCV INTENSIVO → ~3360 combos  │
                            │ Best: n_estimators=100, max_depth=5    │
                            │       class_weight='balanced_subsample'│
                            │       max_features=0.7                 │
                            │ F1-Macro: 0.7601                       │
                            └──────────┬──────────────────────────────┘
                                       │
                                       ├─ DCombo='M1'
                                       └─ DCombo='M2'
```

#### Proceso Detallado

**NIVEL 1: Clasificador M (NO/SI) - Optimización con GridSearchCV**

**Grid de Hiperparámetros**:
```python
param_grid_m = {
    'classifier__n_estimators': [100, 150, 200, 300],
    'classifier__max_depth': [8, 10, 12, 15, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}
# Total: 160 combinaciones
```

**Entrenamiento con GridSearchCV**:
```python
preprocessor_m, _ = build_preprocessor(X_train, nan_threshold=0.20)
pipeline_m = Pipeline([
    ('preprocessor', preprocessor_m),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

grid_m = GridSearchCV(
    pipeline_m,
    param_grid_m,
    cv=cv,              # 5-fold stratified
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)
grid_m.fit(X_train, y_m)
pipeline_m = grid_m.best_estimator_
```

**Mejores Hiperparámetros**:
```python
n_estimators: 150
max_depth: 12
min_samples_split: 2
min_samples_leaf: 1
max_features: 'sqrt'
Best F1-Macro: 0.7618
```

**Predicción con CV**:
```python
m_pred = cross_val_predict(pipeline_m, X_train, y_m, cv=cv)
# Resultado: ['NO', 'SI', 'SI', 'NO', ...]
```

---

**NIVEL 2: Clasificador MM (Solo casos M=SI) - Optimización con GridSearchCV**

**Subset de Datos**:
```python
mask_m_si = (y_m == 'SI')
X_m_si = X_train.loc[mask_m_si]      # 88 muestras
y_mm_si = y_mm[mask_m_si]             # Solo labels de M=SI
```

**Grid de Hiperparámetros**:
```python
param_grid_mm = {
    'classifier__n_estimators': [100, 150, 200, 300],
    'classifier__max_depth': [5, 8, 10, 12, None],  # Más profundidades
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}
```

**Entrenamiento con GridSearchCV**:
```python
preprocessor_mm, _ = build_preprocessor(X_m_si, nan_threshold=0.20)
pipeline_mm = Pipeline([
    ('preprocessor', preprocessor_mm),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

grid_mm = GridSearchCV(
    pipeline_mm,
    param_grid_mm,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=0
)
grid_mm.fit(X_m_si, y_mm_si)
pipeline_mm = grid_mm.best_estimator_
```

**Mejores Hiperparámetros**:
```python
n_estimators: 100
max_depth: 5
min_samples_split: 10
min_samples_leaf: 2
max_features: 'sqrt'
Best F1-Macro: 0.4699
```

**Predicción con CV**:
```python
mask_m_si_pred = (m_pred == 'SI')
X_m_si_pred = X_train.loc[mask_m_si_pred]
y_mm_si_pred = y_mm[mask_m_si_pred]

mm_pred_subset = cross_val_predict(pipeline_mm, X_m_si_pred, y_mm_si_pred, cv=cv)
mm_pred[mask_m_si_pred] = mm_pred_subset
```

---

**NIVEL 3: Clasificador M1 vs M2 - OPTIMIZACIÓN INTENSIVA**

**¿Por qué optimización intensiva?**
Este es el nivel más crítico donde el modelo original tenía peor desempeño. Se implementa una búsqueda exhaustiva con:
- Más opciones de hiperparámetros
- Incluye `class_weight` para balancear clases
- Fracciones de features además de 'sqrt' y 'log2'

**Subset de Datos**:
```python
mask_m1m2 = ((y_dcombo == 'M1') | (y_dcombo == 'M2'))
X_m1m2 = X_train.loc[mask_m1m2]      # 70 muestras
y_dcombo_m1m2 = y_dcombo[mask_m1m2]   # ['M1', 'M2', ...]
# M1: 39 muestras, M2: 31 muestras
```

**Grid EXTENDIDO de Hiperparámetros**:
```python
param_grid_m1m2 = {
    'classifier__n_estimators': [100, 150, 200, 300, 500],  # ← +500
    'classifier__max_depth': [5, 8, 10, 12, 15, 20, None],  # ← +15, 20
    'classifier__min_samples_split': [2, 5, 10, 15],        # ← +15
    'classifier__min_samples_leaf': [1, 2, 4, 8],           # ← +8
    'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7], # ← Fracciones
    'classifier__class_weight': [None, 'balanced', 'balanced_subsample']  # ← ¡NUEVO!
}
# Total: ~3,360 combinaciones
```

**Entrenamiento con GridSearchCV**:
```python
preprocessor_m1m2, _ = build_preprocessor(X_m1m2, nan_threshold=0.20)
pipeline_m1m2 = Pipeline([
    ('preprocessor', preprocessor_m1m2),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

grid_m1m2 = GridSearchCV(
    pipeline_m1m2,
    param_grid_m1m2,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1  # ← Mostrar progreso
)
grid_m1m2.fit(X_m1m2, y_dcombo_m1m2)
pipeline_m1m2 = grid_m1m2.best_estimator_
```

**Mejores Hiperparámetros**:
```python
n_estimators: 100
max_depth: 5
min_samples_split: 5
min_samples_leaf: 2
max_features: 0.7  # ← 70% de features
class_weight: 'balanced_subsample'  # ← Balance de clases
Best F1-Macro: 0.7601
```

**Predicción con CV**:
```python
mask_m1m2_pred = mask_m_si_pred & (mm_pred == 'NO')
X_m1m2_pred = X_train.loc[mask_m1m2_pred]
y_m1m2_pred = y_dcombo[mask_m1m2_pred]

m1m2_pred_subset = cross_val_predict(pipeline_m1m2, X_m1m2_pred, y_m1m2_pred, cv=cv)
dcombo_pred[mask_m1m2_pred] = m1m2_pred_subset
```

---

**Construcción de DCombo Final**:
```python
# Inicializar con 'C' (todos)
dcombo_pred = np.array(['C'] * n_samples, dtype=object)

# Casos M=SI y MM=SI → 'MM'
mask_mm_si = mask_m_si_pred & (mm_pred == 'SI')
dcombo_pred[mask_mm_si] = 'MM'

# Casos M=SI y MM=NO → 'M1' o 'M2' (ya asignado arriba)
# Los casos M=NO quedan como 'C'
```

**Derivar Combo**:
```python
combo_pred = np.where(dcombo_pred == 'C', 'C',
                     np.where(dcombo_pred == 'MM', 'MM', 'M'))
```

#### Ventajas
- Especialización por nivel  
- Divide problemas complejos  
- Mejor para datos desbalanceados  
- Cada modelo se enfoca en una decisión  
- **GridSearchCV optimiza cada nivel independientemente**  
- **Balance de clases en nivel M1/M2**  

#### Desventajas
- 3 modelos que entrenar y mantener  
- Errores se acumulan en cascada  
- Mayor complejidad de implementación  
- **Búsqueda de hiperparámetros tarda varios minutos**  

---

### Subsección 11.3.1: Guardar Modelos Bottom-Up

#### Propósito
Guardar los 3 modelos entrenados y sus hiperparámetros para uso posterior.

#### Archivos Guardados

```python
import joblib
from pathlib import Path

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# Guardar modelos
joblib.dump(pipeline_m, models_dir / 'bottomup_level1_M.pkl')
joblib.dump(pipeline_mm, models_dir / 'bottomup_level2_MM.pkl')
joblib.dump(pipeline_m1m2, models_dir / 'bottomup_level3_M1M2.pkl')

# Guardar hiperparámetros
hyperparams = {
    'level1_M': grid_m.best_params_,
    'level2_MM': grid_mm.best_params_,
    'level3_M1M2': grid_m1m2.best_params_
}
joblib.dump(hyperparams, models_dir / 'bottomup_best_params.pkl')

# Guardar scores de validación
scores_bottomup = {
    'level1_M_f1_macro': grid_m.best_score_,
    'level2_MM_f1_macro': grid_mm.best_score_,
    'level3_M1M2_f1_macro': grid_m1m2.best_score_
}
joblib.dump(scores_bottomup, models_dir / 'bottomup_validation_scores.pkl')
```

#### Cargar Modelos Posteriormente

```python
import joblib

# Cargar modelos
model_m = joblib.load('models/bottomup_level1_M.pkl')
model_mm = joblib.load('models/bottomup_level2_MM.pkl')
model_m1m2 = joblib.load('models/bottomup_level3_M1M2.pkl')

# Cargar hiperparámetros
hyperparams = joblib.load('models/bottomup_best_params.pkl')
print(f"Nivel 1 (M): {hyperparams['level1_M']}")

# Cargar scores
scores = joblib.load('models/bottomup_validation_scores.pkl')
print(f"F1-Macro Nivel 3: {scores['level3_M1M2_f1_macro']:.4f}")
```

---

### Subsección 11.4: Estrategia 3 - HYBRID (Híbrido)

#### Concepto
**"Confía en lo simple, valida con lo especializado"**

#### Arquitectura
```
┌──────────────────────┐
│  Top-Down (Base)     │  ← Predicción principal
│  Confianza: 0.0-1.0  │
└──────────┬───────────┘
           │
           ↓ Comparar con
┌──────────────────────┐
│  Bottom-Up (Validador)│  ← Validación especializada
└──────────┬───────────┘
           │
           ↓
  ¿Hay conflicto? → SI
  ¿Confianza < 0.70? → SI
           │
           ↓
   Usar Bottom-Up para este caso
```

#### Proceso Detallado

**1. Calcular Confianza de Top-Down**
```python
proba_topdown = best_pipeline.predict_proba(X_train)
# Ejemplo: [[0.65, 0.15, 0.10, 0.10],  # Clase C con 65%
#           [0.05, 0.80, 0.10, 0.05]]  # Clase M1 con 80%

confidence_topdown = proba_topdown.max(axis=1)
# [0.65, 0.80, ...]
```

**2. Inicializar con Top-Down**
```python
y_pred_hybrid = {
    'DCombo': y_pred_topdown['DCombo'].copy(),
    'Combo': y_pred_topdown['Combo'].copy(),
    'M': y_pred_topdown['M'].copy(),
    'MM': y_pred_topdown['MM'].copy()
}
```

**3. Validación Cruzada**
```python
CONFIDENCE_THRESHOLD = 0.70
conflicts_resolved = 0

for i in range(n_samples):
    # Detectar conflictos
    conflict_m = (y_pred_topdown['M'][i] != y_pred_bottomup['M'][i])
    conflict_mm = (y_pred_topdown['MM'][i] != y_pred_bottomup['MM'][i])
    
    # Resolver si hay conflicto Y baja confianza
    if (conflict_m or conflict_mm) and confidence_topdown[i] < CONFIDENCE_THRESHOLD:
        conflicts_resolved += 1
        # Usar Bottom-Up para este caso
        y_pred_hybrid['M'][i] = y_pred_bottomup['M'][i]
        y_pred_hybrid['MM'][i] = y_pred_bottomup['MM'][i]
        y_pred_hybrid['DCombo'][i] = y_pred_bottomup['DCombo'][i]
        y_pred_hybrid['Combo'][i] = y_pred_bottomup['Combo'][i]
```

**Ejemplo de Conflicto Resuelto**:
```
Muestra #42:
  Top-Down:    M=SI,  MM=NO,  DCombo=M1  (Confianza: 0.58)
  Bottom-Up:   M=SI,  MM=SI,  DCombo=MM
  Conflicto:   MM diferente (NO vs SI)
  Confianza:   0.58 < 0.70
  Acción:      Usar Bottom-Up → DCombo=MM
```

#### Parámetros Clave
- `CONFIDENCE_THRESHOLD = 0.70` (70%)
- Casos con confianza ≥70% → Mantener Top-Down
- Casos con confianza <70% + conflicto → Usar Bottom-Up

#### Ventajas
- Combina lo mejor de ambos  
- Más robusto ante incertidumbre  
- Validación cruzada automática  
- Adaptativo según confianza

#### Desventajas
- Mayor complejidad computacional  
- Requiere ambos modelos entrenados  
- Umbral de confianza es un hiperparámetro

---

### Subsección 11.5: Comparación Final

#### Tabla Comparativa - Accuracy

```
Columna  Bottom-Up  Hybrid  Top-Down
DCombo     0.5682   0.6212    0.6061
Combo      0.7348   0.7348    0.6894
M          0.7727   0.7879    0.7273
MM         0.9242   0.9167    0.9015
```

#### Tabla Comparativa - F1-Macro

```
Columna  Bottom-Up  Hybrid  Top-Down
DCombo     0.5902   0.6550    0.6437
Combo      0.7457   0.7511    0.7191
M          0.7084   0.7489    0.6932
MM         0.4803   0.7570    0.7318
```

#### Análisis de Rendimiento Promedio

```
Estrategia   Accuracy   F1-Macro
Top-Down      0.7519     0.7225
Bottom-Up     0.7633     0.6563
Hybrid        0.7860     0.7529  ← MEJOR
```

---

### Subsección 11.6: Guardar Configuración y Resultados Finales

#### Propósito
Guardar toda la configuración de las 3 estrategias y crear documentación completa.

#### Archivos Guardados

**1. Configuración Hybrid**
```python
hybrid_config = {
    'confidence_threshold': 0.70,
    'conflicts_resolved': 12,
    'total_samples': 132,
    'use_bottomup_percentage': 9.09
}
joblib.dump(hybrid_config, models_dir / 'hybrid_config.pkl')
```

**2. Resultados de las 3 Estrategias**
```python
all_results = {
    'topdown': results_topdown,
    'bottomup': results_bottomup,
    'hybrid': results_hybrid
}
joblib.dump(all_results, models_dir / 'strategy_results.pkl')
```

**3. Configuración de Cross-Validation**
```python
cv_config = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}
joblib.dump(cv_config, models_dir / 'cv_config.pkl')
```

**4. Preprocessor**
```python
joblib.dump(preprocessor, models_dir / 'preprocessor.pkl')
```

**5. README de Modelos**
Se generó un README.md en `models/` con:
- Fecha de entrenamiento
- Configuración general (random state, CV, umbral NaN)
- Features después de limpieza
- Muestras de entrenamiento
- Hiperparámetros de cada modelo
- Scores de validación
- Instrucciones de uso
- Tabla de rendimiento comparativo



### Subsección 11.7: Conclusiones y Recomendaciones

#### Análisis Comparativo

**1. Top-Down vs Bottom-Up**
```
Bottom-Up es MEJOR por 1.89%
→ La especialización ayuda levemente
```

**2. Hybrid vs Top-Down**
```
Hybrid es MEJOR por 3.41%
→ La validación cruzada mejora significativamente
```

**3. Hybrid vs Bottom-Up**
```
Hybrid es MEJOR por 1.52%
→ La combinación supera a ambos individuales
```

#### Estrategia Recomendada: HYBRID

**Accuracy promedio: 76.52%**

**Ventajas**:
✓ Combina lo mejor de ambos enfoques  
✓ Validación cruzada entre estrategias  
✓ Más robusto ante casos dudosos  
✓ Usa Bottom-Up solo cuando hay incertidumbre

---

## Archivos Generados

### Parte 1 (Entrenamiento)
```
labels_mapping.csv           - Mapeo de clases
cv_results_random_forest.csv - Resultados CV Random Forest
cv_results_xgb.csv           - Resultados CV XGBoost
cv_results_nb.csv            - Resultados CV Naive Bayes
confusion_matrix.csv         - Matriz de confusión
classification_report.csv    - Reporte de clasificación
feature_importances.csv      - Importancia de features
```

### Parte 2 (Estrategias)
```
strategy_comparison.csv      - Comparación de 3 estrategias
```

### Modelos Guardados (carpeta `models/`)

**Top-Down:**
```
topdown_dcombo.pkl           - Pipeline completo del mejor modelo
topdown_label_encoder.pkl    - Label encoder (si XGBoost)
topdown_info.pkl             - Metadata del modelo
```

**Bottom-Up (3 niveles):**
```
bottomup_level1_M.pkl        - Clasificador M (NO/SI)
bottomup_level2_MM.pkl       - Clasificador MM (NO/SI)
bottomup_level3_M1M2.pkl     - Clasificador M1 vs M2
bottomup_best_params.pkl     - Mejores hiperparámetros de cada nivel
bottomup_validation_scores.pkl - Scores de validación cruzada
```

**Hybrid:**
```
hybrid_config.pkl            - Configuración de estrategia Hybrid
```

**Configuración General:**
```
preprocessor.pkl             - ColumnTransformer para preprocesamiento
cv_config.pkl               - Configuración de cross-validation
strategy_results.pkl        - Resultados completos de las 3 estrategias
README.md                   - Documentación de modelos guardados
```

### Notebook de Inferencia

```
inference_pipeline.ipynb     - Pipeline limpio para hacer predicciones
                              (carga modelos sin entrenar)
```

---

## Métricas y Evaluación

### Métricas Utilizadas

**1. Accuracy (Exactitud)**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Porcentaje de predicciones correctas.

**2. F1-Score Macro**
```
F1-Macro = mean([F1_clase1, F1_clase2, F1_clase3, F1_clase4])
```
Promedio no ponderado de F1 de todas las clases.  
**Útil para datos desbalanceados**.

**3. Precision Macro**
```
Precision = TP / (TP + FP)
```
¿De lo que predijimos positivo, cuánto era realmente positivo?

**4. Recall Macro**
```
Recall = TP / (TP + FN)
```
¿De lo que era positivo, cuánto detectamos?

---


## Repo github:
[Repositorio en GitHub — dataset_ml_af_mb](https://github.com/moisesbritez92/dataset_ml_af_mb)

## Referencias:
- Github copilot
- Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow - Aurélien Géron




