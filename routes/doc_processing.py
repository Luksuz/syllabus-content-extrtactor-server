import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Optional
import logging
import asyncio # Import asyncio

# Import models and chain functions
from models import (
    ExtractTocInput, ToCItem, ExtractToCOutput,
    GenerateQuestionsInput, Question, GenerateQuestionsOutput,
    DocumentAnalysisWithQuestionsOutput
)
from chains import (
    process_document_from_bytes,
    extract_table_of_contents,
    generate_questions
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/api")

# --- Environment Variables for Google Cloud Document AI ---
# These should be set in your server's environment
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "lukaabu")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION", "eu") # Format is "us" or "eu"
GOOGLE_PROCESSOR_ID = os.getenv("GOOGLE_PROCESSOR_ID", "e74aaa04cf1d9eef")
# Optional: Override in environment if needed
GOOGLE_PROCESSOR_VERSION_ID = os.getenv("GOOGLE_PROCESSOR_VERSION_ID", "pretrained-ocr-v2.0-2023-06-02")
GOOGLE_FIELD_MASK = os.getenv("GOOGLE_FIELD_MASK", "text,entities,pages.pageNumber")


@router.post("/process_pdf_generate_questions/", response_model=DocumentAnalysisWithQuestionsOutput)
async def process_pdf_and_generate_questions_route(
    pdf: UploadFile = File(...),
    model_name: Optional[str] = Form("gpt-4o-mini") # Default model from notebook
):
    """
    Processes a PDF file to extract its table of contents and then generates 
    questions based on each ToC item asynchronously.
    Returns a list of objects, each containing a ToC item title and its generated questions.
    """
    if not GOOGLE_PROJECT_ID or not GOOGLE_LOCATION or not GOOGLE_PROCESSOR_ID:
        logger.error("Google Cloud configuration (PROJECT_ID, LOCATION, PROCESSOR_ID) is missing in environment variables.")
        raise HTTPException(status_code=500, detail="Server configuration error: Google Cloud settings missing.")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set in environment variables.")
        raise HTTPException(status_code=500, detail="Server configuration error: OPENAI_API_KEY missing.")
    
    if pdf.content_type != "application/pdf":
        logger.warning(f"Invalid file type uploaded: {pdf.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        pdf_content = await pdf.read()
        
        # Step 1: Process document with Google Document AI
        logger.info(f"Starting PDF processing for file: {pdf.filename}")
        document_text = await process_document_from_bytes(
            project_id=GOOGLE_PROJECT_ID,
            location=GOOGLE_LOCATION,
            processor_id=GOOGLE_PROCESSOR_ID,
            file_content=pdf_content,
            mime_type=pdf.content_type, # Use the actual mime_type from the uploaded file
            field_mask=GOOGLE_FIELD_MASK,
            processor_version_id=GOOGLE_PROCESSOR_VERSION_ID
        )
        logger.info(f"PDF processing complete. Extracted text length: {len(document_text)}")

        if not document_text:
            logger.warning("No text extracted from the PDF.")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Step 2: Extract Table of Contents
        logger.info(f"Extracting table of contents using model: {model_name}")
        toc_input = ExtractTocInput(text=document_text, model=model_name)
        toc_data: ExtractToCOutput = await extract_table_of_contents(toc_input)
        
        if not toc_data.items:
            logger.warning("No table of contents items found.")
            # Return empty list if no ToC items, as per user flow
            return DocumentAnalysisWithQuestionsOutput(
                table_of_contents=toc_data,
                topic_questions=[]
            )

        logger.info(f"Table of contents extracted. Found {len(toc_data.items)} items. Audience: {toc_data.audience_level}")

        # Step 3: Generate Questions for each ToC item concurrently
        question_generation_tasks = []
        for item_index, toc_item in enumerate(toc_data.items):
            logger.info(f"Preparing question generation for ToC item {item_index + 1}/{len(toc_data.items)}: '{toc_item.title}'")
            questions_input = GenerateQuestionsInput(
                toc_item_title=toc_item.title,
                toc_item_description=toc_item.description,
                toc_item_audience_level=toc_data.audience_level,
                syllabus_description=toc_data.description,
                model=model_name
            )
            # Add the coroutine to the list of tasks
            question_generation_tasks.append(generate_questions(questions_input))
        
        logger.info(f"Starting concurrent generation of questions for {len(question_generation_tasks)} ToC items.")
        # Run all question generation tasks concurrently
        generated_toc_questions: List[GenerateQuestionsOutput] = await asyncio.gather(*question_generation_tasks)
        logger.info("Concurrent question generation complete.")

        # Filter out any None results from asyncio.gather if a task failed, though gather should raise exceptions
        # Also, ensure that the output matches the expected GenerateQuestionsOutput structure, especially if generate_questions can return None
        # or something else on error.
        # For simplicity, assuming generate_questions always returns GenerateQuestionsOutput or raises an exception.
        
        # The generated_toc_questions list already has the desired structure.
        # We might want to log information about each item.
        for item_output in generated_toc_questions:
            if item_output and item_output.questions:
                logger.info(f"Collected {len(item_output.questions)} questions for '{item_output.toc_item_title}'.")
            else:
                logger.warning(f"No questions generated or collected for ToC item: '{item_output.toc_item_title if item_output else 'Unknown ToC Item (task might have failed without exception or returned None)'}'.")
        
        # Construct the final response object
        final_response = DocumentAnalysisWithQuestionsOutput(
            table_of_contents=toc_data,
            topic_questions=generated_toc_questions
        )
        
        return final_response # Return the combined data

    except HTTPException as http_exc:
        # Re-raise HTTPException
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        await pdf.close() 