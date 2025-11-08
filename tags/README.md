## Datasets description

### English
One of the most studied identification technologies is chipless RFID (Radio Frequency Identifier). A chipless RFID tag is made of a metallic shape deposited on a subtract. The shape is designed so that the resulting tag should have specific resonance frequencies. Those tags are cheaper than the traditional RFID tag and can reach longer reading distances than the well-known barcode. Those advantages make them suitable for several identification needs. 
The downside is that the signal backscattered by chipless RFID tags is very unstable, making the reading difficult. Some studies proposed methods based on AI algorithms to resolve signal processing problems. 
Here two datasets are presented. They are made of measurements of distinct 16 chipless RFID tags. 
Each measurement is composed of 1601 samples (features), corresponding to the magnitude values of the electromagnetic wave reflected by the tag at different frequencies (the frequency range is 1.5 GHz - 5 GHz). 
The frequency axis is divided in five bands to code the tags: 1.5 – 2.3 GHz, 2.3 – 2.75 GHz, 2.75 – 3.4 GHz, 3.4 – 4.4 GHz, and 4.4 – 5 GHz. If a tag has a resonance frequency (a notch in the signal) in a given band, it is coded as “1”, and if not, as “0”.
It is also worthy to note that a resonator can be repeated to make a larger tag. A 2x2 tag means that the resonator is repeated four times and organized as a 2 by 2 matrix. The label “tag_10101_4x4” means that the resonator is repeated 16 times in a 4x4 way; and has three resonance frequencies: one in the first band, one in the third band and one in the last band. The labels are in a column named “Label”.
The last column of each dataset (“Range”) corresponds to the range at which the measurement was made. 
The dataset is made of measurements in the range 50-140 centimeters, divided in subintervals of 50-80; 80-110; 110-130; and 110-140. 
The aim is to train a model able to predict the labels (tag identifier) based on the measurement (i.e., the 1601 samples/features). You are provided with a training set and a test set.
Note that the labels (tag IDs) for the test set are not provided. The predicted tag ID should be provided for each of the measurements in the test set.

### Español
Una de las tecnologías de identificación más estudiadas es la RFID sin chip (Identificador por Radiofrecuencia). Una etiqueta RFID sin chip está hecha de una forma metálica depositada sobre un sustrato. La forma está diseñada para que la etiqueta resultante tenga frecuencias de resonancia específicas. Estas etiquetas son más baratas que las etiquetas RFID tradicionales y pueden alcanzar distancias de lectura más largas que el conocido código de barras. Estas ventajas las hacen adecuadas para varias necesidades de identificación.
La desventaja es que la señal retrodispersada por las etiquetas RFID sin chip es muy inestable, lo que dificulta la lectura. Algunos estudios han propuesto métodos basados en algoritmos de IA para resolver problemas de procesamiento de señales.
Aquí se presentan dos conjuntos de datos. Están compuestos por mediciones de 16 etiquetas RFID sin chip distintas.
Cada medición se compone de 1601 muestras (características), que corresponden a los valores de magnitud de la onda electromagnética reflejada por la etiqueta a diferentes frecuencias (el rango de frecuencia es de 1,5 GHz a 5 GHz).
El eje de frecuencia se divide en cinco bandas para codificar las etiquetas: 1,5    – 2,3 GHz, 2,3 – 2,75 GHz, 2,75 – 3,4 GHz, 3,4 – 4,4 GHz y 4,4 – 5 GHz. Si una etiqueta tiene una frecuencia de resonancia (una muesca en la señal) en una banda determinada, se codifica como "1", y si no, como "0".
También es importante tener en cuenta que un resonador puede repetirse para hacer una etiqueta más grande. Una etiqueta 2x2 significa que el resonador se repite cuatro veces y se organiza como una matriz de 2 por 2. La etiqueta "tag_10101_4x4" significa que el resonador se repite 16 veces de manera 4x4; y tiene tres frecuencias de resonancia: una en la primera banda, una en la tercera banda y una en la última banda. Las etiquetas están en una columna llamada "Label".
La última columna de cada conjunto de datos ("Range") corresponde al rango en el que se realizó la medición.
El conjunto de datos está compuesto por mediciones en el rango de 50-140 centímetros, divididas en subintervalos de 50-80; 80-110; 110-130; y 110-140.
El objetivo es entrenar un modelo capaz de predecir las etiquetas (identificador de la etiqueta) en función de la medición (es decir, las 1601 muestras/características). Se le proporciona un conjunto de entrenamiento y un conjunto de prueba.
Tenga en cuenta que las etiquetas (ID de etiqueta) para el conjunto de prueba no se proporcionan. Se debe proporcionar el ID de etiqueta predicho para cada una de las mediciones en el conjunto de prueba.


