import argparse
from pathlib import Path

import joblib

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

CSV_PATH = "Churn_Modelling.csv"
PREPARED_DATA_PATH = "prepared_data.joblib"
TRAINED_MODEL_PATH = "trained_model.joblib"
MODEL_PATH = "classifier.joblib"


def prepare_step():
    """Prépare les données et les enregistre pour l'étape suivante."""
    prepared_data = prepare_data(CSV_PATH)
    joblib.dump(prepared_data, PREPARED_DATA_PATH)
    print(f"Données préparées et enregistrées dans : {PREPARED_DATA_PATH}")


def train_step():
    """Entraîne le modèle avec les données préparées."""
    if not Path(PREPARED_DATA_PATH).exists():
        print("Lance d'abord : python main.py prepare")
        return

    X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
    model = train_model(X_train, y_train)

    joblib.dump(model, TRAINED_MODEL_PATH)
    print(f"Modèle entraîné et enregistré temporairement dans : {TRAINED_MODEL_PATH}")


def evaluate_step():
    """Évalue le modèle entraîné sur les données de test."""
    if not Path(PREPARED_DATA_PATH).exists():
        print("Lance d'abord : python main.py prepare")
        return

    if not Path(TRAINED_MODEL_PATH).exists():
        print("Lance d'abord : python main.py train")
        return

    X_train, X_test, y_train, y_test = joblib.load(PREPARED_DATA_PATH)
    model = joblib.load(TRAINED_MODEL_PATH)

    evaluate_model(model, X_test, y_test)


def save_step():
    """Enregistre le modèle entraîné dans classifier.joblib."""
    if not Path(TRAINED_MODEL_PATH).exists():
        print("Lance d'abord : python main.py train")
        return

    model = joblib.load(TRAINED_MODEL_PATH)
    save_model(model, MODEL_PATH)


def load_step():
    """Charge le modèle enregistré."""
    if not Path(MODEL_PATH).exists():
        print("Lance d'abord : python main.py save")
        return

    loaded_model = load_model(MODEL_PATH)
    print("Le modèle enregistré a été chargé avec succès.")
    print(f"Type du modèle : {type(loaded_model).__name__}")


def all_steps():
    """Exécute toutes les étapes dans l'ordre."""
    X_train, X_test, y_train, y_test = prepare_data(CSV_PATH)
    joblib.dump((X_train, X_test, y_train, y_test), PREPARED_DATA_PATH)
    print(f"Données préparées et enregistrées dans : {PREPARED_DATA_PATH}")

    model = train_model(X_train, y_train)
    joblib.dump(model, TRAINED_MODEL_PATH)
    print(f"Modèle entraîné et enregistré temporairement dans : {TRAINED_MODEL_PATH}")

    evaluate_model(model, X_test, y_test)

    save_model(model, MODEL_PATH)

    loaded_model = load_model(MODEL_PATH)
    print("Le modèle enregistré a été chargé avec succès.")
    print(f"Type du modèle : {type(loaded_model).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Exécution du pipeline de churn")
    parser.add_argument(
        "step",
        choices=["prepare", "train", "evaluate", "save", "load", "all"],
        help="Étape du pipeline à exécuter",
    )
    step = parser.parse_args().step

    if step == "prepare":
        prepare_step()
    elif step == "train":
        train_step()
    elif step == "evaluate":
        evaluate_step()
    elif step == "save":
        save_step()
    elif step == "load":
        load_step()
    elif step == "all":
        all_steps()


if __name__ == "__main__":
    main()
