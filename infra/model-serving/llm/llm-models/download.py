from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M",
    local_dir="."
)

# multilingual-e5-small
# BAAI/bge-reranker-base