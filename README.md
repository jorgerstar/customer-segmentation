# 🛍️ Laboratorio 2 — Modelos de Aprendizaje No Supervisado
### Segmentación de Clientes en una Plataforma de Servicios Digitales

> **Materia:** Aprendizaje Automático
> **Grupo:** 2 — Semana 3

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Estructura del Repositorio](#-estructura-del-repositorio)
3. [Dataset](#-dataset)
4. [Requisitos del Entorno](#-requisitos-del-entorno)
5. [Cómo Ejecutar el Notebook](#-cómo-ejecutar-el-notebook)
6. [Metodología](#-metodología)
7. [Resultados Principales](#-resultados-principales)
8. [Perfiles de Cliente Identificados](#-perfiles-de-cliente-identificados)
9. [Comparativa de Modelos](#-comparativa-de-modelos)
10. [Limitaciones y Trabajo Futuro](#-limitaciones-y-trabajo-futuro)

---

## 📌 Descripción del Proyecto

Una plataforma digital de servicios personalizados necesita **segmentar a sus usuarios** con base en patrones de comportamiento y consumo, para adaptar estrategias de marketing y mejorar la experiencia del cliente.

Este laboratorio implementa y compara cuatro técnicas de aprendizaje no supervisado:

| Técnica | Propósito |
|---|---|
| **K-Means** | Segmentación en K grupos por minimización de distancias |
| **DBSCAN** | Clustering por densidad con detección automática de outliers |
| **PCA** | Reducción dimensional lineal para visualización |
| **t-SNE** | Reducción dimensional no lineal para exploración visual |

Adicionalmente, se aplican **Isolation Forest** y **LOF** para detección de anomalías.

---

## 📁 Estructura del Repositorio

```
📦 lab2-aprendizaje-no-supervisado/
│
├── 📓 Lab2_NoSupervisado_FINAL.ipynb   ← Notebook principal (ejecutar este)
├── 📊 Mall_Customers.csv               ← Dataset de entrada
└── 📄 README.md                        ← Este archivo
```

---

## 📊 Dataset

**Fuente:** Mall Customer Segmentation Data  
**Registros:** 200 clientes | **Variables:** 5

| Variable | Tipo | Descripción | Uso en modelos |
|---|---|---|---|
| `CustomerID` | Numérico | Identificador único del cliente | ❌ Eliminada (sin valor estadístico) |
| `Genre` | Categórico | Género del cliente (Male/Female) | ❌ Excluida (requiere encoding) |
| `Age` | Numérico | Edad en años (18–70) | ✅ Feature del modelo |
| `Annual Income (k$)` | Numérico | Ingreso anual en miles de dólares (15–137) | ✅ Feature del modelo |
| `Spending Score (1-100)` | Numérico | Score de comportamiento de compra asignado por el mall (1–99) | ✅ Feature del modelo |

**Distribución de género:** 112 mujeres (56%) / 88 hombres (44%)  
**Sin valores nulos** — dataset limpio y completo.

> `CustomerID` y `Genre` se excluyen explícitamente en la Parte 3 del notebook, con su justificación técnica documentada.

---

## ⚙️ Requisitos del Entorno

### Versiones utilizadas

| Librería | Versión recomendada |
|---|---|
| Python | ≥ 3.8 |
| pandas | ≥ 1.3 |
| numpy | ≥ 1.21 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |
| scikit-learn | ≥ 1.0 |

### Instalación

Si trabajas en un entorno local, instala las dependencias con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Si usas Google Colab o Jupyter con entorno limpio, descomenta la primera celda del notebook:

```python
# !pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ Cómo Ejecutar el Notebook

### Opción 1 — Google Colab (recomendado)

1. Abre [Google Colab](https://colab.research.google.com)
2. Ve a **Archivo → Subir notebook** y sube `Lab2_NoSupervisado_FINAL.ipynb`
3. Sube también `Mall_Customers.csv` al entorno de Colab:
   - Panel izquierdo → ícono de carpeta → botón de subir archivo
4. Ejecuta todas las celdas: **Entorno de ejecución → Ejecutar todo** (`Ctrl+F9`)

### Opción 2 — Jupyter Notebook / JupyterLab (local)

```bash
# Clona el repositorio
git clone https://github.com/<tu-usuario>/<nombre-repo>.git
cd <nombre-repo>

# Inicia Jupyter
jupyter notebook
# o
jupyter lab
```

Luego abre `Lab2_NoSupervisado_FINAL.ipynb` desde la interfaz y ejecuta todas las celdas en orden (**Kernel → Restart & Run All**).

### Opción 3 — VS Code con extensión Jupyter

1. Abre la carpeta del proyecto en VS Code
2. Instala la extensión **Jupyter** de Microsoft
3. Abre el archivo `.ipynb` y usa el botón **▶ Run All**

> ⚠️ **Importante:** el archivo `Mall_Customers.csv` debe estar en la **misma carpeta** que el notebook para que `pd.read_csv('Mall_Customers.csv')` funcione correctamente.

---

## 🔬 Metodología

El notebook está organizado en 8 partes secuenciales. Cada parte debe ejecutarse en orden ya que las variables (como `X_scaled`, `df`, `pca`, etc.) se reutilizan entre secciones.

### Parte 1 — Preparación del entorno
Importación de librerías y configuración de semilla global (`SEED = 42`) para reproducibilidad de todos los modelos estocásticos.

### Parte 2 — Análisis Exploratorio (EDA)
- Carga del dataset y revisión de dimensiones, tipos y nulos
- Estadísticas descriptivas (`describe()`)
- Histogramas con KDE por variable
- Boxplots por género
- Heatmap de correlaciones
- Pairplot de relaciones cruzadas

### Parte 3 — Limpieza y Preprocesamiento
- Eliminación de `CustomerID` y `Genre` con justificación técnica
- Escalado con `StandardScaler` → media=0, std=1 por variable
- Verificación del resultado del escalado

### Parte 4 — Implementación de Modelos

**K-Means:**
- Loop de k=2 a k=10 calculando WCSS y Silhouette Score
- Visualización combinada: Método del Codo + Silhouette
- Selección de **k=5** (justificada numéricamente)
- Entrenamiento del modelo final con `init='k-means++'`, `n_init=10`

**DBSCAN:**
- K-Distance Graph con `NearestNeighbors(n_neighbors=5)` para selección de `eps`
- Selección de `eps=0.5`, `min_samples=5`
- Informe de clusters detectados y puntos de ruido
- Silhouette Score excluyendo ruido

**Comparativa visual K-Means vs DBSCAN:**
- Gráfico lado a lado en el plano Ingreso vs Gasto escalados

### Parte 5 — Reducción Dimensional
- **PCA:** 2 componentes, varianza explicada, loadings por variable, gráficas de varianza individual y acumulada
- **t-SNE:** perplexity=30, learning_rate=200, n_iter=1000
- **Visualización comparativa** PCA vs t-SNE, ambos coloreados por cluster K-Means

### Parte 6 — Perfilamiento de Segmentos
- Tabla resumen de características medias por cluster (con tamaño y % del total)
- Asignación de nombre descriptivo a cada perfil
- Heatmap normalizado de perfiles
- Barplot comparativo Ingreso vs Spending Score por cluster

### Parte 7 — Detección de Anomalías
- **Isolation Forest** (`contamination=0.05`)
- **LOF** (`n_neighbors=20`, `contamination=0.05`)
- Análisis de consenso entre ambos métodos
- Visualización comparativa de outliers detectados

### Parte 8 — Análisis Comparativo y Conclusiones
- Tabla comparativa técnica de los 4 modelos
- Resumen numérico de métricas (Silhouette de cada modelo)
- Respuesta estructurada a las 3 preguntas de reflexión de la tarea
- Tabla de limitaciones con propuestas de mejora

---

## 📈 Resultados Principales

### K-Means

| Parámetro | Valor |
|---|---|
| K óptimo | **5** (justificado por Elbow + Silhouette) |
| Inicialización | `k-means++` |
| Silhouette Score | **0.4166** |
| WCSS | 168.25 |

El método del codo muestra un quiebre pronunciado en k=5. El Silhouette Score de 0.4166 indica estructura de cluster moderada-buena, coherente con un dataset de solo 3 variables y 200 registros.

### DBSCAN

| Parámetro | Valor |
|---|---|
| eps | **0.5** (seleccionado con K-Distance Graph) |
| min_samples | **5** (regla: dimensiones + 2) |
| Clusters encontrados | 6 |
| Puntos de ruido | ~60 (30%) |
| Silhouette Score (sin ruido) | **0.4817** |

> El Silhouette Score de DBSCAN es ligeramente superior al de K-Means porque evalúa solo los puntos densos, excluyendo los outliers que K-Means forzaría dentro de un cluster.

### PCA

| Componente | Varianza explicada |
|---|---|
| PC1 | ~43% |
| PC2 | ~28% |
| **Total 2D** | **~55%** |

Con 2 componentes se conserva el 55% de la información, suficiente para visualización orientativa.

### t-SNE

Configuración: `perplexity=30`, `learning_rate=200`, `n_iter=1000`, `random_state=42`.  
Confirma visualmente los mismos agrupamientos detectados por K-Means, con clusters más compactos y separados que en PCA.

---

## 👤 Perfiles de Cliente Identificados

Con K-Means (k=5), se identificaron cinco segmentos con características de negocio claramente diferenciadas:

| Cluster | Perfil | Edad media | Ingreso medio | Gasto medio | Estrategia recomendada |
|---|---|---|---|---|---|
| **0** | 📉 Ahorradores de Bajo Ingreso | ~46 años | ~$27k | 18/100 | Fidelización con descuentos y beneficios económicos |
| **1** | 🛍️ Jóvenes Impulsivos | ~25 años | ~$41k | 62/100 | Opciones de financiamiento responsable; retención temprana |
| **2** | 💎 Premium Activos | ~33 años | ~$86k | 82/100 | Productos premium, programas de lealtad, atención VIP |
| **3** | 💼 Alto Ingreso Conservador | ~40 años | ~$86k | 19/100 | Activación mediante incentivos personalizados y exclusivos |
| **4** | ⚖️ Perfil Estándar / Moderado | ~56 años | ~$54k | 49/100 | Cross-selling y upselling de productos complementarios |

> Los clusters **2** (Premium Activos) y **3** (Alto Ingreso Conservador) son los más estratégicamente importantes: mismo nivel de ingreso, comportamiento de gasto opuesto → oportunidad de activación de Cluster 3 hacia el patrón de Cluster 2.

---

## ⚖️ Comparativa de Modelos

| Criterio | K-Means | DBSCAN | PCA | t-SNE |
|---|---|---|---|---|
| Tipo de técnica | Partición | Densidad | Lineal | No lineal |
| Nro. de clusters | Definido por usuario (5) | Automático (6) | N/A | N/A |
| Manejo de outliers | No — los asigna forzosamente | Sí — los marca como ruido (-1) | No | No |
| Forma de clusters | Asume esférica | Arbitraria | N/A | N/A |
| Interpretabilidad | Alta | Media | Alta (loadings) | Baja (ejes sin significado) |
| Silhouette Score | 0.4166 | 0.4817 (sin ruido) | N/A | N/A |
| Uso principal en este proyecto | Segmentación final de clientes | Validación y detección de outliers | Visualización 2D con varianza conservada | Exploración de estructura local |

**Diferencia clave observada:** K-Means asignó todos los 200 clientes a uno de 5 grupos. DBSCAN identificó ~60 clientes con comportamiento realmente atípico (ruido) que K-Means enmascaraba en el Cluster 4 (moderado). Esto tiene valor operativo: esos ~60 clientes merecen análisis individual antes de incluirlos en campañas masivas.

---

## ⚠️ Limitaciones y Trabajo Futuro

| Limitación encontrada | Propuesta de mejora |
|---|---|
| K-Means asume clusters esféricos de igual varianza | Reemplazar por **GMM** (Gaussian Mixture Models) que permite clusters elípticos con distinta covarianza |
| DBSCAN es muy sensible al parámetro `eps` | Usar **HDBSCAN**, que selecciona eps automáticamente según la densidad local variable |
| t-SNE no preserva distancias globales y es estocástico | Complementar con **UMAP**, que preserva mejor la estructura global y es determinista |
| Dataset pequeño (200 registros) | Validar con datos reales de producción de la plataforma antes de implementar en producción |
| Solo 3 variables de comportamiento | Enriquecer con datos de sesiones web, clics, historial de compras y métricas RFM (Recency, Frequency, Monetary) |

---

## 👥 Autores

Laboratorio 2 — Grupo 2 | Materia: Aprendizaje Automático

---

## 📚 Referencias y Recursos

- [Scikit-learn — Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn — DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Scikit-learn — PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Scikit-learn — t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [Dataset — Mall Customer Segmentation Data (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
