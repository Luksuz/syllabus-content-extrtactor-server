from pydantic import BaseModel, Field
from typing import Literal, Union

class Option(BaseModel):
    text: str
    correct: bool

class MultipleChoiceQuestion(BaseModel):
    question_type: Literal["multiple_choice"] 
    prompt: str
    options: list[Option]

class QaPair(BaseModel):
    text_with_blank: str = Field(description="The existing text with a blank to fill in")
    blank_text_answer: str = Field(description="The text to fill in the blank(the answer)")

class FillInBlankQuestion(BaseModel):
    question_type: Literal["fill_in_blank"]
    prompt: str
    qa_pairs: list[QaPair] = Field(description="A list of existing text and answer blank text for the fill in the missing text")

class OpenEndedQuestion(BaseModel):
    question_type: Literal["open_ended"]
    prompt: str
    answer: str = Field(description="The answer to the question")

class Section(BaseModel):
    section_number: float
    section_questions: list[Union[MultipleChoiceQuestion, FillInBlankQuestion, OpenEndedQuestion]]

class Sections(BaseModel):
    is_exercise_page: bool
    sections: list[Section]

class Chapter(BaseModel):
    chapter_name: str
    sections: list[Section] = Field(description="A list of sections in the chapter")

class StructuredChapterList(BaseModel):
    chapters: list[Chapter]

class StructureVisualExtractionInput(BaseModel):
    base_64_img: str
    model: str