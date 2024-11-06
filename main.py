from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import faiss
import os

app = FastAPI()

# Initialize the Sentence-Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Dimension of embeddings produced by MiniLM
index = faiss.IndexFlatL2(dimension)
document_store = []  # Store metadata for documents


@app.post("/ingest/")
async def ingest_document(file: UploadFile):
    """
    Endpoint to upload a document for ingestion.
    """
    # Save the uploaded file
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read file content and generate embeddings
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        embedding = model.encode(content)

    # Add embedding to FAISS index
    index.add([embedding])
    document_store.append({"file_name": file.filename, "content": content})

    return {"message": f"Document '{file.filename}' ingested successfully."}


@app.post("/query/")
async def query_document(query: str = Form(...)):
    """
    Endpoint to query documents based on input text.
    """
    # Generate embedding for the query
    query_embedding = model.encode(query).reshape(1, -1)

    # Search for the nearest neighbors
    distances, indices = index.search(query_embedding, k=5)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(document_store):
            results.append({"file_name": document_store[idx]["file_name"], "distance": dist})

    return JSONResponse(content={"results": results})
