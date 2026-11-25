import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(csv_path="Churn_Modelling.csv"):
    """Charge le CSV, prépare les variables et sépare les données."""
    df = pd.read_csv(csv_path)

    # Le notebook transforme Female/Male en valeurs numériques.
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # On sépare la cible des variables explicatives.
    X = df.drop(columns=["Exited", "Surname", "Geography", "RowNumber", "CustomerId"])
    y = df["Exited"]

    # On reprend les paramètres de séparation du notebook.
    return train_test_split(X, y, test_size=0.2, random_state=1)


def train_model(X_train, y_train):
    """Entraîne le modèle Random Forest utilisé dans le notebook."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcule et affiche l'accuracy et la matrice de confusion."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("Matrice de confusion :")
    print(matrix)

    return accuracy, matrix


def save_model(model, model_path="classifier.joblib"):
    """Enregistre le modèle entraîné dans un fichier."""
    joblib.dump(model, model_path)
    print(f"Modèle enregistré dans : {model_path}")


def load_model(model_path="classifier.joblib"):
    """Charge un modèle enregistré."""
    return joblib.load(model_path)
