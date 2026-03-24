from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dragonkue/multilingual-e5-small-ko",
    local_dir="./embedding-models"
)

# multilingual-e5-small
# BAAI/bge-reranker-base