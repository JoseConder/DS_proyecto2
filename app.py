import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import GridSearchCV

# Cargar el dataset
df = pd.read_csv('./Pokemon.csv')  # Cambia esto a la ubicación de tu dataset

# Seleccionar las características y la variable objetivo
X = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
y = df['Legendary']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo ExtraTrees
model = ExtraTreeClassifier(random_state=42)

# GridSearch
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Entrenar el mejor modelo
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Título de la aplicación
st.title('Predicción de Pokémon Legendarios')

# Mostrar los datos del dataset
st.subheader('Datos del Dataset')
st.write(df)

# Filtrar los Pokémon legendarios y no legendarios
legendary_pokemon = df[df['Legendary'] == True]
non_legendary_pokemon = df[df['Legendary'] == False]

# Visualización 1: Distribución de las características
st.subheader('Distribución de Características')

features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

for i, feature in enumerate(features):
    sns.kdeplot(legendary_pokemon[feature], label='Legendario', fill=True, color='blue', alpha=0.5, ax=axs[i])
    sns.kdeplot(non_legendary_pokemon[feature], label='No Legendario', fill=True, color='orange', alpha=0.5, ax=axs[i])
    axs[i].set_title(f'Distribución de {feature}')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Densidad')
    axs[i].legend()

plt.tight_layout()
st.pyplot(fig)

# Mostrar lista de Pokémon legendarios con sus estadísticas
st.subheader('Pokémon Legendarios')
st.write(legendary_pokemon[['Name', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])

# Mostrar Pokémon que destacan en cada estadística
st.subheader('Pokémon que destacan en cada estadística')
max_stats = {}
for feature in features:
    max_value = df[feature].max()
    best_pokemon = df[df[feature] == max_value][['Name', 'Legendary']]
    max_stats[feature] = best_pokemon

for feature, best in max_stats.items():
    st.write(f"**{feature}**: {best.iloc[0]['Name']} (Legendario: {best.iloc[0]['Legendary']})")


# Interfaz para las entradas del usuario
st.subheader('Ingrese las estadísticas del Pokémon')
hp = st.number_input('HP', min_value=0)
attack = st.number_input('Attack', min_value=0)
defense = st.number_input('Defense', min_value=0)
sp_atk = st.number_input('Sp. Atk', min_value=0)
sp_def = st.number_input('Sp. Def', min_value=0)
speed = st.number_input('Speed', min_value=0)

# Botón para realizar la predicción
if st.button('Predecir'):
    new_pokemon = np.array([[hp, attack, defense, sp_atk, sp_def, speed]])
    prediction = best_model.predict(new_pokemon)

    if prediction[0] == True:
        st.success("El Pokémon es legendario.")
    else:
        st.error("El pokemon no es legendario.")

