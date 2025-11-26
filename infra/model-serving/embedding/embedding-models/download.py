from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./bge-m3"
)

snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir="./bge-reranker-base"
)
# BAAI/bge-m3
# intfloat/multilingual-e5-small
# dragonkue/multilingual-e5-small-ko
# BAAI/bge-reranker-base