import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

#Load OpenAI API Key
load_dotenv(find_dotenv())
openai.api_key=os.getenv("OPENAI_API_KEY")

#Set llm model
llm_model = init_chat_model("gpt-4o-mini", model_provider="openai")

input_message = """This product is shitty af. I want my money back"""
tone = "polite"
language = "Hindi"

template_string = """Rephrase the below text 
{input_message} to 
{tone} tone and translate to 
{language}"""

prompt_template =   ChatPromptTemplate.from_template(template_string)

translation_message = prompt_template.format_messages(
    input_message = input_message,
    tone = tone,
    language = language)

response = llm_model.invoke(translation_message)

print(response)