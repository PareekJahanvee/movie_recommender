from pydantic import BaseModel

class MovieRequest(BaseModel):
    title: str
    threshold: float = 0.3
    top_k: int = 10