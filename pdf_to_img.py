import asyncio
from pdf2image import convert_from_path
import tempfile

async def convert_pdf_to_images(pdf_path: str):
    # Convert each page in a thread
    images = await asyncio.to_thread(convert_from_path, pdf_path)

    image_paths = []
    for i, img in enumerate(images):
        # Use NamedTemporaryFile to save each image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            await asyncio.to_thread(img.save, tmp.name, 'PNG')
            image_paths.append(tmp.name)

    return image_paths
