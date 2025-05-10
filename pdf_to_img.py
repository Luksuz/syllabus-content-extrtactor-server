import asyncio
from pdf2image import convert_from_path
import os

async def convert_pdf_to_images(pdf_path: str):
    # Convert each page in a thread
    images = await asyncio.to_thread(convert_from_path, pdf_path)

    # Save each image in a thread
    os.makedirs('images', exist_ok=True)
    image_paths = []
    for i, img in enumerate(images):
        img_path = f'images/page_{i+1}.png'
        await asyncio.to_thread(img.save, img_path, 'PNG')
        image_paths.append(img_path)

    return image_paths
