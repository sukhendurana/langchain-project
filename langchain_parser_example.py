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