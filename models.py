from pydantic import BaseModel, Field
from typing import Literal, Union, Optional

class Option(BaseModel):
    text: str
    is_correct: bool

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

# --- Models from google-docai.ipynb --- #

class ExtractTocInput(BaseModel):
    text: str
    model: str = "gpt-4o-mini"

class ToCItem(BaseModel):
    title: str
    description: str = Field(description="A short description of what the topic covers")

class ExtractToCOutput(BaseModel):
    items: list[ToCItem] = Field(description="List of titles found in the table of contents")
    description: str = Field(description="A description of what the syllabus covers")
    audience_level: str = Field(description="The knowledge level or audience this syllabus is intended for")

class FillInBlankPair(BaseModel):
    blank_text: str
    blank_text_answer: str

class Question(BaseModel):
    question_type: Literal["multiple_choice", "fill_in_blank", "open_ended"]
    prompt: str
    options: Optional[list[Option]] = Field(default_factory=list)
    fill_in_blank_pairs: Optional[list[FillInBlankPair]] = Field(default_factory=list)
    open_ended_answer: Optional[str]

class GenerateQuestionsInput(BaseModel):
    toc_item_title: str
    toc_item_description: str
    toc_item_audience_level: str
    model: str = "gpt-4o-mini"

class GenerateQuestionsOutput(BaseModel):
    toc_item_title: str
    questions: list[Question]
# --- End of models from google-docai.ipynb --- #

class DocumentAnalysisWithQuestionsOutput(BaseModel):
    table_of_contents: ExtractToCOutput
    topic_questions: list[GenerateQuestionsOutput]