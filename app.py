from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from datetime import datetime
import mlflow
import mlflow.sklearn



app = Flask(__name__)

# Télécharge les données de Kaggle
def download_data():
    path = kagglehub.dataset_download("khwaishsaxena/lung-cancer-dataset")
    return path

# Charger et préparer les données
def load_and_prepare_data():
    # Chemin relatif vers LungCancer.csv dans le même dossier que app.py
    dataset_path = os.path.join(os.getcwd(), 'LungCancer.csv')
    
    # Vérifie si le fichier existe à ce chemin
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Le fichier 'LungCancer.csv' n'a pas été trouvé à l'emplacement {dataset_path}")

    # Charger les données
    data = pd.read_csv(dataset_path)

    # Afficher les colonnes disponibles pour vérification
    print("Colonnes disponibles : ", data.columns)

    # Remplacer 'survived' par le nom de la colonne cible
    target_column = 'survived'  # 'survécu' est la colonne cible ici

    # Encoder les variables catégorielles
    categorical_columns = ['gender', 'smoking_status', 'family_history', 'hypertension', 'asthma', 
                           'cirrhosis', 'other_cancer', 'treatment_type', 'country', 'cancer_stage']

    # Appliquer l'encodage sur les colonnes catégorielles (OneHotEncoder)
    data = pd.get_dummies(data, columns=categorical_columns)

    # Convertir la colonne 'diagnosis_date' en nombre de jours depuis une date de référence
    data['diagnosis_date'] = pd.to_datetime(data['diagnosis_date'], errors='coerce')  # Convertir en datetime
    reference_date = datetime(2000, 1, 1)  # Date de référence
    data['diagnosis_date'] = (data['diagnosis_date'] - reference_date).dt.days  # Convertir en jours

    # Convertir la colonne 'end_treatment_date' en nombre de jours depuis la même date de référence
    data['end_treatment_date'] = pd.to_datetime(data['end_treatment_date'], errors='coerce')  # Convertir en datetime
    data['end_treatment_date'] = (data['end_treatment_date'] - reference_date).dt.days  # Convertir en jours

    # Séparer les caractéristiques (X) et la cible (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Diviser les données en un ensemble d'entraînement (80%) et un ensemble de test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normaliser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_test, y_test, scaler, X.columns.tolist()  # Retourne aussi les noms de colonnes

# Entraîner le modèle
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params({"n_estimators": 100, "random_state": 42})

    return model

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Enregistrer le modèle dans MLFlow
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")  # Enregistrer le modèle
        mlflow.log_params({"n_estimators": 100, "random_state": 42})  # Enregistrer des hyperparamètres
        # Ajouter d'autres métadonnées si nécessaire
    return model


# Sauvegarder le modèle pour un usage futur
def save_model(model, scaler, column_names):
    joblib.dump(model, 'lung_cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')  # Sauvegarde également le scaler
    joblib.dump(column_names, 'columns.pkl')  # Sauvegarde des noms de colonnes

# Charger le modèle
def load_model():
    model = joblib.load('lung_cancer_model.pkl')
    scaler = joblib.load('scaler.pkl')  # Charger le scaler
    return model, scaler

# Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

@app.route('/')
def index():
    # Rendre le fichier HTML du formulaire
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    try:
        # Charger et préparer les données
        X_train, X_test, y_train, y_test, X_test_final, y_test_final, scaler, column_names = load_and_prepare_data()

        # Entraîner le modèle
        model = train_model(X_train, y_train)

        # Sauvegarder le modèle et le scaler
        save_model(model, scaler, column_names)

        # Évaluer le modèle
        evaluation = evaluate_model(model, X_test_final, y_test_final)

        return jsonify({"message": "Modèle entraîné avec succès", "evaluation": evaluation})

    except Exception as e:
        # En cas d'erreur, retourner l'erreur dans la réponse
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Charger le modèle, le scaler et les noms des colonnes
        model, scaler = load_model()

        # Vérifie si le fichier `columns.pkl` existe
        if os.path.exists('columns.pkl'):
            column_names = joblib.load('columns.pkl')  # Charger les noms de colonnes
        else:
            raise FileNotFoundError("Le fichier 'columns.pkl' n'existe pas. Veuillez entraîner le modèle d'abord.")

        # Obtenir les données d'entrée envoyées par l'utilisateur
        input_data = request.get_json()

        # Convertir les données d'entrée en un DataFrame pandas
        input_df = pd.DataFrame([input_data])  # Placer dans une liste pour créer une ligne dans le DataFrame

        # Supprimer la colonne 'id' si elle existe
        if 'id' in input_df.columns:
            input_df = input_df.drop('id', axis=1)

        # Appliquer le même encodage sur les colonnes catégorielles
        categorical_columns = ['gender', 'smoking_status', 'family_history', 'hypertension', 'asthma', 
                               'cirrhosis', 'other_cancer', 'treatment_type', 'country', 'cancer_stage']
        
        # Appliquer OneHotEncoder dans le predict aussi
        input_df = pd.get_dummies(input_df, columns=categorical_columns)

        # Aligner les colonnes d'entrée avec celles utilisées lors de l'entraînement
        missing_cols = set(column_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Ajouter les colonnes manquantes avec 0
        input_df = input_df[column_names]  # Réorganiser les colonnes pour correspondre à l'ordre d'entraînement

        # Convertir la colonne 'diagnosis_date' en nombre de jours depuis une date de référence
        input_df['diagnosis_date'] = pd.to_datetime(input_df['diagnosis_date'], errors='coerce')  # Convertir en datetime
        reference_date = datetime(2000, 1, 1)  # Date de référence
        input_df['diagnosis_date'] = (input_df['diagnosis_date'] - reference_date).dt.days  # Convertir en jours

        # Convertir la colonne 'end_treatment_date' en nombre de jours depuis la même date de référence
        input_df['end_treatment_date'] = pd.to_datetime(input_df['end_treatment_date'], errors='coerce')  # Convertir en datetime
        input_df['end_treatment_date'] = (input_df['end_treatment_date'] - reference_date).dt.days  # Convertir en jours

        # Assurer que les mêmes étapes de prétraitement sont appliquées (normalisation)
        input_scaled = scaler.transform(input_df)  # Utilise le scaler chargé

        # Faire la prédiction
        prediction = model.predict(input_scaled)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        # En cas d'erreur, retourner l'erreur dans la réponse
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
