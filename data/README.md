# Datos

Esta carpeta está reservada para los datos locales utilizados en el paper **“Ventanas y emociones: hacia predicción afectiva en tiempo real”**.

Los archivos de datos no se incluyen en el repositorio. Deben descargarse desde la fuente original y colocarse localmente con la estructura indicada más abajo.

Fuente original:

> **Physiological signals from three wearable devices recorded in real-world conditions**  
> DOI: [10.5281/zenodo.17985866](https://doi.org/10.5281/zenodo.17985866)

De la fuente original se selecciona únicamente el subconjunto correspondiente a **EmotiBit volar**. Los datos deben estar procesados previamente como se describe en el paper antes de ejecutar los notebooks.

Estructura esperada:

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

`Gsr.csv` contiene la señal de actividad electrodérmica, referida en el paper como EDA/GSR. `Hr.csv` contiene la frecuencia cardíaca derivada de PPG. Los archivos de `Stamps/` contienen las marcas temporales de los eventos experimentales utilizadas para segmentar las señales.
