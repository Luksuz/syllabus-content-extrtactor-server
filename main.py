from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.extract_sections import router as extract_sections_router
from routes.doc_processing import router as doc_processing_router
from example_output import example_output

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_sections_router)
app.include_router(doc_processing_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

    