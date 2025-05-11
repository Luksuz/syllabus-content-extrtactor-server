import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from models import StructureVisualExtractionInput, Sections, ExtractTocInput, ExtractToCOutput, GenerateQuestionsInput, GenerateQuestionsOutput
from dotenv import load_dotenv
import aiofiles
from typing import Union, Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud.documentai_v1.services.document_processor_service import DocumentProcessorServiceAsyncClient
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


async def encode_image(image_path: str) -> str:
    async with aiofiles.open(image_path, "rb") as image_file:
        content = await image_file.read()
        return base64.b64encode(content).decode("utf-8")

async def extract_vision_data(input: StructureVisualExtractionInput):
    llm = ChatOpenAI(model=input.model, temperature=0, seed=42).with_structured_output(Sections)
    message = HumanMessage(
        content=[
            {"type": "text", "text": 
             (
            "You are an expert syllabus analyzer.\n"
            "Your first and most important task is to classify the image as either an EXERCISE PAGE (contains practice questions to be answered) or a LECTURE PAGE (contains only explanatory text, examples, or theory, but not actual exercises for the student to solve).\n"
            "- If the image is a LECTURE PAGE (explanatory text, theory, or questions as examples, but NOT a list of questions for the student to answer), you MUST return false and an empty list.\n"
            "- If the image is an EXERCISE PAGE (contains a list of questions, problems, or exercises for the student to solve, regardless of formatting), return true and proceed to extract all questions as described below.\n"
            "\n"
            "**How to decide:**\n"
            "- If there is a title that could refer to the lecture chapter, and the majority of the text is instructional, theoretical, or provides explanations/example questions, it is a LECTURE PAGE.\n"
            "- If the majority of the text consists of questions, problems, or exercises for the student to answer, it is an EXERCISE PAGE.\n"
            "- If you are unsure, err on the side of classifying as a LECTURE PAGE and return an empty list.\n"
            "\n"
            "If and ONLY if the image is an EXERCISE PAGE, do the following:\n"
            "Your task is to extract ALL text from this image, with a special focus on identifying EVERY question present, regardless of formatting or location. Carefully review the entire image, line by line, to ensure that NO question is missed, even if it is embedded in a paragraph, formatted unusually, or separated from its answer.\n\n"
            "For each question, extract its full text and any associated answer, and structure them according to the following JSON schema and question types:\n\n"
            "There are three main question types you may encounter:\n"
            "1. Multiple Choice (`multiple_choice`):\n"
            "   - Has a prompt and a list of options.\n"
            "   - Each option has text and a boolean indicating if it is correct.\n"
            "2. Fill in the Blank (`fill_in_blank`):\n"
            "   - Has a prompt and a list of pairs, each with the text containing a blank and the correct answer for the blank.\n"
            "   - **IMPORTANT:** For every fill-in-the-blank question, you MUST provide an answer for `blank_text_answer`. It is REQUIRED to fill in every blank. If you are unsure of the answer, use your best guess based on the context. If the answer is truly not present or cannot be determined, write `\"unknown\"` or `\"N/A\"`—but NEVER leave it empty or blank.\n"
            "3. Open Ended (`open_ended`):\n"
            "   - Has a prompt only, with no options or blanks.\n\n"
            "Questions are grouped into sections, and sections may be grouped into chapters. If you can identify section or chapter structure, include it; otherwise, just extract the questions.\n\n"
            "If a question does not have an explicit answer, provide an answer.\n\n"
            "Return your results as a complete list of all questions and answers in the required JSON format. Do not summarize, skip, or merge questions—extract and list each one individually and completely."
        )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{input.base_64_img}"},
            },
        ],
    )
    response = await llm.ainvoke([message])
    return response

# --- Logic from google-docai.ipynb --- #

async def process_document_from_bytes(
    project_id: str,
    location: str,
    processor_id: str,
    file_content: bytes,
    mime_type: str,
    field_mask: Optional[str] = "text,entities,pages.pageNumber",
    processor_version_id: Optional[str] = "pretrained-ocr-v2.0-2023-06-02",
) -> str:
    """Processes a document using Google Document AI from byte content asynchronously."""
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com", credentials_file="config.json")

    async with DocumentProcessorServiceAsyncClient(client_options=opts) as client:
        if processor_version_id:
            name = client.processor_version_path(
                project_id, location, processor_id, processor_version_id
            )
        else:
            name = client.processor_path(project_id, location, processor_id)

        raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

        process_options = documentai.ProcessOptions(
            individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
                pages=[_ for _ in range(1, 16)]
            )
        )

        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            field_mask=field_mask,
            process_options=process_options,
        )

        result = await client.process_document(request=request)
        document = result.document
    return document.text


async def extract_table_of_contents(inputs: Union[ExtractTocInput, dict]):
    if isinstance(inputs, dict):
        inputs = ExtractTocInput(**inputs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert grammar syllabus analyzer. Your task is to identify and extract the table of contents 
                from the provided syllabus text, as well as determine the knowledge level or audience this syllabus is intended for.
                The syllabus may provide a list of topics and exam questions for each topic. 
                Description should reflect the topic purpose of what grammar student will learn in the topic.
                Keep in mind that this description is later used to generate practice questions for the student.
                
                Look for patterns that indicate a table of contents such as:
                - Numbered or bulleted lists of topics
                - Page numbers following topic titles
                - Hierarchical organization of topics
                - Section titles with corresponding page numbers
                
                Also analyze the content to determine the intended audience level (e.g., beginner, intermediate, advanced, 
                undergraduate, graduate, professional, etc.) based on:
                - Explicit statements about the intended audience
                - Complexity of topics covered
                - Terminology used
                - Prerequisites mentioned
                - Overall difficulty level of the material
                
                Return a list of items found in the table of contents as a list of {{title: str, description: str}} and the audience level as a string.
                Do not include page numbers or other formatting elements in the titles.
                """
            ),
            (
                "human",
                "Document text:\\n{text}",
            ),
        ]
    )

    llm = ChatOpenAI(
        model=inputs.model,
        temperature=0,
    ).with_structured_output(ExtractToCOutput)

    chain = (
        {
            "text": lambda x: x["text"],
        }
        | prompt
        | llm
    )

    result = await chain.ainvoke({"text": inputs.text})
    return result


async def generate_questions(inputs: Union[GenerateQuestionsInput, dict]):
    if isinstance(inputs, dict):
        inputs = GenerateQuestionsInput(**inputs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert educational content creator. Your task is to generate grammar practice questions based on a topic from a syllabus.
                description might be about different life situations that the student may encounter. If that is the case, generate grammar questions that are relevant to the topic
                and the student might have encountered in their life.
                
                Create a diverse set of questions covering the provided topic, including:
                1. Multiple Choice questions (`multiple_choice`):
                   - Create a prompt and 4-5 options
                   - Mark exactly one option as correct
                
                2. Fill in the Blank questions (`fill_in_blank`):
                   - Create sentences with blanks to be filled
                   - Provide the correct answer for each blank
                
                3. Open Ended questions (`open_ended`):
                   - Create thought-provoking questions that require explanatory answers
                
                Ensure the questions are appropriate for the specified audience level and cover the topic thoroughly.
                Generate at least 2 questions of each type (6+ questions total).
                """
            ),
            (
                "human",
                "Topic Title: {toc_item_title}\\nTopic Description: {toc_item_description}\\nAudience Level: {toc_item_audience_level}"
            ),
        ]
    )

    llm = ChatOpenAI(
        model=inputs.model,
        temperature=0.2,
    ).with_structured_output(GenerateQuestionsOutput)

    chain = (
        {
            "toc_item_title": lambda x: x["toc_item_title"],
            "toc_item_description": lambda x: x["toc_item_description"],
            "toc_item_audience_level": lambda x: x["toc_item_audience_level"],
        }
        | prompt
        | llm
    )

    result = await chain.ainvoke({
        "toc_item_title": inputs.toc_item_title,
        "toc_item_description": inputs.toc_item_description,
        "toc_item_audience_level": inputs.toc_item_audience_level
    })
    return result

# --- End of logic from google-docai.ipynb --- #