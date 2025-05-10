import base64
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from models import StructureVisualExtractionInput, Sections
from dotenv import load_dotenv
import aiofiles

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