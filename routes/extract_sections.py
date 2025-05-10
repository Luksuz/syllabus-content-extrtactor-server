from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import asyncio
from pdf_to_img import convert_pdf_to_images
from chains import extract_vision_data, encode_image
from models import StructureVisualExtractionInput

router = APIRouter(prefix="/api")

async def process_image(image_path, model):
    base_64_img = await encode_image(image_path)
    input = StructureVisualExtractionInput(base_64_img=base_64_img, model=model)
    result = await extract_vision_data(input)
    return result.model_dump()

@router.post("/extract-sections/")
async def extract_sections(pdf: UploadFile = File(...), model: str = Form("gpt-4.1-mini")):
    try:
        print("Extracting sections from PDF")
        # Save uploaded PDF to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            content = await pdf.read()
            tmp.write(content)

        print("Converting PDF to images")
        image_paths = await convert_pdf_to_images(pdf_path)
        print("Processing images")
        tasks = [process_image(image_path, model) for image_path in image_paths]
        sections = await asyncio.gather(*tasks)
        return JSONResponse(content=sections) 
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
