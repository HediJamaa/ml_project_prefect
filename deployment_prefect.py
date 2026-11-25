from prefect import serve

from pipeline_prefect import (
    all_flow,
    train_flow,
    evaluate_flow,
    code_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(
            name="train",
            tags=["mlops"],
        ),
        evaluate_flow.to_deployment(
            name="evaluate",
            tags=["evaluation", "mlops"],
        ),
        code_flow.to_deployment(
            name="code",
            tags=["mlops"],
        ),
    )
