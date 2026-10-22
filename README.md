# Segmentación de clientes con K-Means – Mall Customers

Taller final de Minería de Datos. Segmentación de los 200 clientes del
[Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
con K-Means, evaluada con el método del codo y el coeficiente de silueta (K óptimo = 5, silueta ≈ 0.555).

## Contenido

| Archivo | Descripción |
|---|---|
| `Taller_Final_KMeans_Mall_Customers.ipynb` | Notebook con las partes 1–7 y las respuestas a todas las preguntas |
| `Taller_Final_KMeans_Mall_Customers.html` | El mismo notebook exportado a HTML |
| `app.py` | Aplicación Streamlit: mapa de segmentos, clasificación de un cliente nuevo, codo y silueta, estrategias y exploración de los datos |
| `Mall_Customers.csv` | Dataset |
| `modelo_kmeans.pkl` | Escalador + modelo K-Means (K = 5) guardados por el notebook; la app los usa para K = 5 |
| `requirements.txt` | Dependencias |
| `.streamlit/config.toml` | Tema de la aplicación |

## Perfiles obtenidos

| Segmento | Perfil | Clientes | Ingreso medio | Gasto medio |
|---|---|---|---|---|
| A | Alto valor | 39 | ~87 k$ | ~82 |
| B | Con potencial (ingreso alto, gasto bajo) | 35 | ~88 k$ | ~17 |
| C | Promedio | 81 | ~55 k$ | ~50 |
| D | Con potencial (jóvenes gastadores) | 22 | ~26 k$ | ~79 |
| E | Bajo valor | 23 | ~26 k$ | ~21 |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

## Desplegar en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio de GitHub (incluida la carpeta `.streamlit/`).
2. Entrar a [share.streamlit.io](https://share.streamlit.io), elegir **Create app** y seleccionar el repositorio,
   la rama y `app.py` como archivo principal.
3. Desplegar. Streamlit Cloud instala `requirements.txt` y aplica el tema de `.streamlit/config.toml`.

Si la versión de scikit-learn del servidor no puede leer `modelo_kmeans.pkl`, la app vuelve a entrenar el
modelo con los mismos parámetros (`k-means++`, `n_init=10`, `random_state=42`) y obtiene los mismos segmentos.
