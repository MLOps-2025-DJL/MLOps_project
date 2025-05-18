import pytest
from airflow.models import DagBag

DAG_ID = "retrain_model_pipeline"

@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder="airflow/dags", include_examples=False, read_dags_from_db=False)

def test_dag_loaded(dagbag):
    dag = dagbag.dags[DAG_ID]
    assert dag is not None, "DAG 'retrain_model_pipeline' failed to load"
    assert dag.dag_id == "retrain_model_pipeline"

def test_task_count(dagbag):
    dag = dagbag.dags[DAG_ID]
    assert len(dag.tasks) == 5, "Expected 5 tasks in the DAG"

def test_task_ids(dagbag):
    dag = dagbag.dags[DAG_ID]
    expected_task_ids = {
        "insert_metadata_to_postgres",
        "download_images",
        "retrain_model",
        "save_model_to_minio",
        "redeploy_model",
    }
    assert set(dag.task_ids) == expected_task_ids

def test_dependencies(dagbag):
    dag = dagbag.dags[DAG_ID]

    assert dag.get_task("insert_metadata_to_postgres").downstream_task_ids == {"download_images"}
    assert dag.get_task("download_images").downstream_task_ids == {"retrain_model"}
    assert dag.get_task("retrain_model").downstream_task_ids == {"save_model_to_minio"}
    assert dag.get_task("save_model_to_minio").downstream_task_ids == {"redeploy_model"}
