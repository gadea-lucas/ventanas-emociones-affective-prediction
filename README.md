# Ventanas y emociones: hacia predicción afectiva en tiempo real

Repositorio asociado al paper **“Ventanas y emociones: hacia predicción afectiva en tiempo real”**.

El proyecto estudia la clasificación de estados afectivos a partir de señales fisiológicas registradas con **EmotiBit** en la zona **volar** de la muñeca. El análisis se centra en el efecto del tamaño de ventana temporal, el solapamiento de ventanas, el balanceo de clases y la selección de características para aproximar un escenario de inferencia afectiva en tiempo real.

## Datos

Los datos utilizados en este trabajo proceden del dataset publicado en Zenodo:

> **Physiological signals from three wearable devices recorded in real-world conditions**  
> DOI: [10.5281/zenodo.17985866](https://doi.org/10.5281/zenodo.17985866)

Los datos no se incluyen en el repositorio. Para reproducir los experimentos, debe descargarse el dataset original y colocar localmente en `data/` el subconjunto correspondiente a **EmotiBit volar**, procesado previamente como se describe en el paper.

La estructura esperada por los notebooks es:

```text
data/
  Stamps/
    1.txt
    ...
    10.txt
  emotibit_volar/
    1/
      Gsr.csv
      Hr.csv
    ...
    10/
      Gsr.csv
      Hr.csv
```

Los archivos esperados son:

- `Gsr.csv`: actividad electrodérmica, referida en el paper como EDA/GSR.
- `Hr.csv`: frecuencia cardíaca derivada de la señal PPG.
- `data/Stamps/`: marcas temporales de los eventos experimentales.

A partir de estos datos locales, los notebooks generan las ventanas temporales, extraen características con `tsfresh`, evalúan los modelos y exportan las figuras.

## Estructura del repositorio

- `data/`: carpeta local para colocar el subconjunto EmotiBit volar y las marcas temporales. Solo se versionan `data/README.md` y `data/.gitkeep`.
- `NB_01_dataset_extractor.ipynb`: extracción de características y evaluación por tamaño de ventana.
- `NB_02_feature_selection.ipynb`: selección de características mediante importancia Gini.
- `utils.py`: funciones auxiliares de carga, validación cruzada y balanceo.
- `01_export_window_smote_figures.py`: exporta figuras de rendimiento por ventana y balanceo.
- `02_export_confusion_matrices.py`: exporta matrices de confusión en PDF.
- `03_export_features_figures.py`: exporta figuras de selección de características.
- `features_con_solapamiento/` y `features_sin_solapamiento/`: carpetas de salida para características generadas.
- `figures/`: carpeta de salida para figuras generadas.

## Instalación

Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducción

1. Descargar el dataset original desde Zenodo y preparar localmente la estructura indicada en `data/README.md`.
2. Ejecutar `NB_01_dataset_extractor.ipynb` para generar las características y resultados por ventana.
3. Ejecutar `NB_02_feature_selection.ipynb` para realizar la selección de características.
4. Exportar las figuras con:

```bash
python 01_export_window_smote_figures.py
python 02_export_confusion_matrices.py
python 03_export_features_figures.py
```

Las características intermedias se generan en `features_con_solapamiento/` y `features_sin_solapamiento/`. Las figuras se generan en `figures/`.

## Nota sobre reproducibilidad

La evaluación emplea validación cruzada estratificada con separación por usuario (`StratifiedGroupKFold`) para evitar que ventanas de un mismo participante aparezcan simultáneamente en entrenamiento y prueba. Las estrategias de balanceo se aplican únicamente sobre el conjunto de entrenamiento en cada partición.
