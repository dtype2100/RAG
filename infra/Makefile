# 기본 설정 변수 선언
EMBEDDING_DIR := model-serving/embedding
LLM_DIR := model-serving/llm
MLFLOW_DIR := mlflow
AIRFLOW_DIR := airflow

# 모든 서비스 일괄 실행
up-all:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml up -d
	docker compose -f $(LLM_DIR)/compose.ollama.yaml up -d
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml up -d
	docker compose -f $(AIRFLOW_DIR)/compose.airflow.yaml up -d

# 모든 서비스 일괄 재시작
restart-all:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml restart
	docker compose -f $(LLM_DIR)/compose.ollama.yaml restart
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml restart
	docker compose -f $(AIRFLOW_DIR)/compose.airflow.yaml restart

# 모든 서비스 일괄 재생성(포스 리크리에이트)
recreate-all:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml up -d --force-recreate
	docker compose -f $(LLM_DIR)/compose.ollama.yaml up -d --force-recreate
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml up -d --force-recreate
	docker compose -f $(AIRFLOW_DIR)/compose.airflow.yaml up -d --force-recreate

# 모든 서비스 일괄 중지
down-all:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml down --volumes --rmi all
	docker compose -f $(LLM_DIR)/compose.ollama.yaml down --volumes --rmi all
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml down --volumes --rmi all
	docker compose -f $(AIRFLOW_DIR)/compose.airflow.yaml down --volumes --rmi all

# 임베딩 서비스 실행/중지/재시작/재생성
up-embedding:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml up -d

down-embedding:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml down

down-embedding-all:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml down --volumes --rmi all

restart-embedding:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml restart

recreate-embedding:
	docker compose -f $(EMBEDDING_DIR)/compose.embed.yaml up -d --force-recreate

# LLM 서비스 실행/중지/재시작/재생성
up-llm:
	docker compose -f $(LLM_DIR)/compose.ollama.yaml up -d

down-llm:
	docker compose -f $(LLM_DIR)/compose.ollama.yaml down

down-llm-all:
	docker compose -f $(LLM_DIR)/compose.ollama.yaml down --volumes --rmi all

restart-llm:
	docker compose -f $(LLM_DIR)/compose.ollama.yaml restart

recreate-llm:
	docker compose -f $(LLM_DIR)/compose.ollama.yaml up -d --force-recreate

# MLflow 서비스 실행/중지/재시작/재생성
up-mlflow:
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml up -d

down-mlflow:
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml down

down-mlflow-all:
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml down --volumes --rmi all	

restart-mlflow:
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml restart

recreate-mlflow:
	docker compose -f $(MLFLOW_DIR)/compose.mlflow.yaml up -d --force-recreate

