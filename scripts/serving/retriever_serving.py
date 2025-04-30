from fastapi import FastAPI, HTTPException
import argparse
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque

from flashrag.config import Config
from flashrag.utils import get_retriever

app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None

def init_retriever(args):
    global retriever_semaphore
    config = Config(args.config)
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False

class Document(BaseModel):
    id: str
    contents: str

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].search(query, top_n, return_score)
                return [Document(id=result['id'], contents=result['contents']) for result in results], scores
            else:
                results = retriever_list[retriever_idx].search(query, top_n, return_score)
                return [Document(id=result['id'], contents=result['contents']) for result in results]
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    async with retriever_semaphore:
        retriever_idx = available_retrievers.popleft()
        try:
            if return_score:
                results, scores = retriever_list[retriever_idx].batch_search(query, top_n, return_score)
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))], scores
            else:
                results = retriever_list[retriever_idx].batch_search(query, top_n, return_score)
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./retriever_config.yaml")
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    
    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

