from fastapi import APIRouter


router = APIRouter()

@router.get("vector-search")
def vector_search():
    """
    Vector Search
    """
    try:
        return {"message": "Vector search"}
    except Exception as e:
        return {"message": str(e)}