"""Step 05: scheduled deployments (acts as the worker — keep it running).

Before running:
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
"""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)

if __name__ == "__main__":
    all_dep = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",  # every day at 02:00
        tags=["full-pipeline", "mlops"],
    )
    train_dep = train_flow.to_deployment(
        name="ml-pipeline-train", tags=["training", "mlops"]
    )
    evaluate_dep = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
    )
    code_dep = code_flow.to_deployment(
        name="ml-pipeline-code", tags=["quality", "mlops"]
    )
    install_dep = install_flow.to_deployment(
        name="ml-pipeline-install", tags=["setup", "mlops"]
    )

    serve(all_dep, train_dep, evaluate_dep, code_dep, install_dep)
