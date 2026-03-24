from fastapi import FastAPI
from app.api.router_vector_search import router as vector_search_router

app = FastAPI()

app.include_router(vector_search_router, prefix="/api/v1", tags=["Vector Search"])

@app.get("/")
def read_root():
    return {"Connection": "Success!"}


