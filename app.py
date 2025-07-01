from flask import Flask, request, jsonify
import pandas as pd
import os
import kagglehub

app = Flask(__name__)

# Télécharge les données de Kaggle
def download_data():
    path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")
    return path

@app.route('/')
def index():
    return "Bienvenue sur l'API de prédiction du cancer du poumon"

@app.route('/predict', methods=['POST'])
def predict():
    # Exemple de prédiction, on va charger le dataset et effectuer une prédiction simple
    dataset_path = download_data()
    data = pd.read_csv(os.path.join(dataset_path, 'data.csv'))

    # Pour l'exemple, nous utilisons juste la taille du dataset comme une prédiction
    # En pratique, il faudrait charger ton modèle ML et faire la prédiction avec celui-ci
    prediction = {"number_of_rows": len(data)}
    
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
