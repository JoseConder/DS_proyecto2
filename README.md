# Clasificación de Pokémon Legendarios con ExtraTrees y Streamlit

Este proyecto es una aplicación web construida con **Streamlit** que predice si un Pokémon es legendario o no en función de sus estadísticas. La app utiliza el modelo **ExtraTrees Classifier**, una variante de Random Forest, entrenado con el dataset de Pokémon. También incluye visualizaciones interactivas de datos y muestra a los Pokémon que destacan en diversas estadísticas.
Link aqui [Proyecto](https://ds299506proyecto2.streamlit.app/)
## Características
- **Predicción de Pokémon Legendarios**: Introduce las estadísticas de un nuevo Pokémon y predice si es legendario utilizando un modelo ExtraTrees ya entrenado.
- **Visualizaciones**: Explora visualizaciones interactivas relacionadas con los Pokémon legendarios, como gráficos de distribución y mapas de calor de la matriz de confusión.
- **Estadísticas de Pokémon**: Muestra una tabla de todos los Pokémon legendarios con sus estadísticas.
- **Mejores Pokémon por Estadística**: Resalta los Pokémon que sobresalen en cada categoría estadística y muestra si son legendarios o no.

## Dataset
El dataset utilizado en este proyecto se obtiene de [Kaggle - Pokémon Dataset](https://www.kaggle.com/abcsds/pokemon). Contiene información detallada sobre los Pokémon, incluyendo sus estadísticas base como HP, Ataque, Defensa, etc., y si son legendarios o no.

### Características utilizadas:
- `HP`: Puntos de vida
- `Attack`: Fuerza de ataque físico
- `Defense`: Resistencia a ataques físicos
- `Sp. Atk`: Fuerza de ataque especial
- `Sp. Def`: Resistencia a ataques especiales
- `Speed`: Velocidad del Pokémon

## Instalación

### Dependecias usadas
- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Numpy


