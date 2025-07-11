## Descripción del Proyecto
Este proyecto ha sido elaborado como trabajo final de la asignatura *Inteligencia Artificial y Estadistica* de la Universidad de Sevilla por **Mario Pantoja Castro**.

El principal objetivo del mismo fue construir modelos basados en **redes convolucionales** para la clasificación de **imágenes diagnósticas** de ultrasonido de **tejidos mamarios**.
A su vez, la clasificación tiene como objetivo diagnosticar **tumores cancerosos**, **tumores benignos** o confirmar **ausencia** de estos. 
Durante el proceso se han establecido **tres objetivos** diferentes: detectar la presencia de tumores, diferenciar entre tumores benignos y malignos y clasificar
el tejido mamario entre ausente de tumores, presente de tumor benigno o presente de tumor maligno. Información más detallada sobre el proyecto y sus objetivos puede encontrarse en []().

Por último, el conjunto de datos utilizado, 780 imágenes de ultrasonido, se obtuvo de [Breast-Cancer-Ultrasound-Images-Dataset](https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset).
Más sobre el conjunto de datos puede encontrarse en []().

## Funcionamiento del Proyecto
Para este proyecto se ha utilizado principalente **Python**, en concreto su librería `PyTorch`, y **R**, en específico su librería `flexdashboard`.

En el directorio `data` se puede encontrar el conjunto de datos utilizado durante el proyecto con la estructura necesaria para cada uno de los objetivos. 
En `EDA` encontramos el **Análisis Descriptivo de los Datos**, a partir de *scripts* de R. Por otro lado, en `scripts` encontramos el núcleo del proyecto.
Aquí tenemos tres módulos de Python cruciales: `scripts/processingData`, `scripts/training` y `scripts/evaluating`, para el **preprocesamiento y transformación 
de los datos**, para el **entranamiento de los modelos** y para su **evaluación**. Cada uno contiene una función que cumple con estas tareas.

Aparte, en este mismo directorio encontramos tres **Jupyter Notebooks**. Cada uno trata de alcanzar un objetivo de los anteriormente mencionados. Así, en estos
archivos encontramos la **creación, entrenamiento y validación de modelos**. En concreto, en cada uno encontramos dos modelos, apuntando al mismo objetivo, pero 
con complejidad diferente. Aquí también se pueden ver los resultados de la evaluación de cada uno. Además, los parámetros de cada modelo creado en estos archivos 
fueron guardados en formato `pth` en el directorio `models`. Aquí se pueden encontrar modelos `_base`, más sencillos, y modelos `_tuned`, de mayor complejidad.

Por último, podemos encontrar el archivo `dashboard.Rmd` donde se usa `flexdashboard` para crear una página web (`dashboard.html`) donde se presentan los 
principales resultados del proyecto. Aquí también se puede encontrar información más detallada sobre cada uno de los pasos del proyecto. Este archivo usa el directorio
`images` para insertar ciertas imágenes en la web. 
