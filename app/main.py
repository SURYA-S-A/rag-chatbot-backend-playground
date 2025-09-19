from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from app.models.models import (
    InitCollectionRequest,
    KnowledgeBotRequest,
    StoreDocsRequest,
    QueryRequest,
)
from fastapi.middleware.cors import CORSMiddleware
from app.services.services_registry import file_processor
from app.services.services_registry import vector_store_manager
from app.services.services_registry import knowledge_bot_app


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    root_path="/api",
    docs_url="/swagger",
    openapi_url="/docs/openapi.json",
    redoc_url="/docs",
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/init-collection")
def init_collection(req: InitCollectionRequest):
    try:
        vector_store_manager.init_collection(collection_name=req.collection_name)
        return {"status": "success", "collection": req.collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store-docs")
def store_docs(req: StoreDocsRequest):
    try:
        docs = file_processor.load_and_split(req.file_paths)
        vector_store_manager.add_documents(req.collection_name, docs)
        return {"status": "success", "docs_stored": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-docs")
async def upload_docs(
    collection_name: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),  # case 1: frontend uploads
    file_paths: Optional[List[str]] = Form(None),  # case 2: backend/local paths
):
    try:
        # Ensure collection exists
        vector_store_manager.init_collection(collection_name=collection_name)

        docs = []

        # Case 1: In-memory uploaded files
        if files:
            docs = await file_processor.load_and_split(files=files)

        # Case 2: Existing file paths
        elif file_paths:
            docs = await file_processor.load_and_split(file_paths=file_paths)

        else:
            raise HTTPException(
                status_code=400, detail="No files or file_paths provided."
            )

        # Store in vector DB
        vector_store_manager.add_documents(collection_name=collection_name, docs=docs)

        return {
            "status": "success",
            "collection": collection_name,
            "docs_stored": len(docs),
            "files": [file.filename for file in files] if files else file_paths,
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_collection(req: QueryRequest):
    try:
        results = vector_store_manager.query(
            collection_name=req.collection_name,
            query=req.query,
            filenames=req.filenames,
            k=req.k,
        )
        return {
            "status": "success",
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invoke-graph")
def invoke_knowledge_bot(knowledge_bot_request: KnowledgeBotRequest):
    try:
        result = knowledge_bot_app.run_agent(
            knowledge_bot_request=knowledge_bot_request
        )
        return {"status": "success", "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
