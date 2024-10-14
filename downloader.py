import os
import zipfile

# Descargar el dataset de Kaggle
os.system('kaggle datasets download -d abcsds/pokemon')

# Descomprimir el archivo zip
with zipfile.ZipFile('pokemon.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Listar los archivos descomprimidos
print(os.listdir())
