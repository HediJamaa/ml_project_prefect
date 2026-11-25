import argparse
import subprocess
import sys

import main
from prefect import flow, task

CODE_FILES = ("model_pipeline.py", "main.py", "pipeline_prefect.py")


@task(name="Installer les dépendances")
def install_dependencies_task():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


@task(name="Formatage du code")
def format_code_task():
    subprocess.run(
        [sys.executable, "-m", "black", *CODE_FILES],
        check=True,
    )


@task(name="Qualité du code")
def code_quality_task():
    subprocess.run(
        [sys.executable, "-m", "pylint", "--errors-only", *CODE_FILES],
        check=True,
    )


@task(name="Sécurité du code")
def code_security_task():
    subprocess.run(
        [sys.executable, "-m", "bandit", "-ll", *CODE_FILES],
        check=True,
    )


@task(name="Tests unitaires")
def unit_tests_task():
    subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_model_pipeline"],
        check=True,
    )


@task(name="Préparer les données")
def prepare_data_task():
    main.prepare_step()


@task(name="Entraîner le modèle")
def train_model_task():
    main.train_step()


@task(name="Sauvegarder le modèle")
def save_model_task():
    main.save_step()


@task(name="Charger le modèle")
def load_model_task():
    return main.load_model(main.MODEL_PATH)


@task(name="Évaluer le modèle")
def evaluate_model_task(model=None):
    if model is None:
        # Dans le flow all, évalue le modèle temporaire entraîné.
        main.evaluate_step()
    else:
        # Dans le flow evaluate, évalue le modèle chargé.
        _, X_test, _, y_test = main.prepare_data(main.CSV_PATH)
        main.evaluate_model(model, X_test, y_test)


@flow(name="ml-pipeline-all")
def all_flow():
    install_dependencies_task()
    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()
    prepare_data_task()
    train_model_task()
    save_model_task()
    evaluate_model_task()


@flow(name="ml-pipeline-train")
def train_flow():
    prepare_data_task()
    train_model_task()


@flow(name="ml-pipeline-evaluate")
def evaluate_flow():
    model = load_model_task()
    evaluate_model_task(model)


@flow(name="ml-pipeline-code")
def code_flow():
    install_dependencies_task()
    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()


def main_cli():
    parser = argparse.ArgumentParser(description="Lancement des flows Prefect")
    parser.add_argument(
        "--flow",
        choices=["all", "train", "evaluate", "code"],
        required=True,
        help="Flow à exécuter",
    )
    selected_flow = parser.parse_args().flow

    if selected_flow == "all":
        all_flow()
    elif selected_flow == "train":
        train_flow()
    elif selected_flow == "evaluate":
        evaluate_flow()
    elif selected_flow == "code":
        code_flow()


if __name__ == "__main__":
    main_cli()
