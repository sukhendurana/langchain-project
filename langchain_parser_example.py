import os
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

#Load OpenAI API Key
load_dotenv(find_dotenv())
openai.api_key=os.getenv("OPENAI_API_KEY")

#Set llm model
llm_model = init_chat_model("gpt-4o-mini", model_provider="openai")

email_response = """
Here's out itinerary for our upcoming trip to Europe.
We leave from Denver, Colorado Airport at 8.45pm, arrive in Amsterdam 10 hours flight and land at Schipol Airport.
We'll grab a ride to our airbnb and maybe stop somewhere for breakfast before taking a nap.

Some sighseeing will follow for couple of hours
We will then go for shopping to bring back to our kids.

The next orning, at 7.45am we'll drive to Belgium, Brussels - it should only take a few hours. While in Brussels, we want to explore the city to its fullest.
"""

# desired_format = {
#     "leave_time" : "8.45 pm",
#     "leave_from" : "Denver, Colorado",
#     "cities_to_visit" : ["Amsterdam","Brussels"]
# }

# email_template = """
# From the following email, extract the following information:

# leave_time : when they are leaving for vacation to Europe. If there is an actual time written, use it, if not, write unknown.

# leave_from : where they are leaving from, the airport, city name or state if available

# cities_to_visit : extract the cities they are planning to visit. If there are more than one then, put them in square brackets like this ["city1", "city2"].

# Format the output as JSON with the following keys:
# leave_time
# leave_from
# cities_to_visit

# email : {email}
# """

#promp_template = ChatPromptTemplate.from_template(email_template)

#print(promp_template)

#messages = promp_template.format_messages(email = email_response)

#response = llm_model.invoke(messages)

#print(type(response.content))

#Create schema
leave_time_schema = ResponseSchema(name="leave_time", description="When they are leaving, usually it is the numeric value of time, if not available then n/a", type="string")
leave_from_schema = ResponseSchema(name="leave_from", description="Where they are going to, usually it is the city or airport name", type="string")
cities_to_visit_schema = ResponseSchema(name="cities_to_visit", description="Names of the cities they would be visiting as part of the trip. This will be a list", type="list")

response_schema = [leave_time_schema, leave_from_schema, cities_to_visit_schema]

#setup output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()

#print(format_instructions)

email_template_formatted = """
From the following email, extract the following information:

leave_time : when they are leaving for vacation to Europe. If there is an actual time written, use it, if not, write unknown.

leave_from : where they are leaving from, the airport, city name or state if available

cities_to_visit : extract the cities they are planning to visit. If there are more than one then, put them in square brackets like this ["city1", "city2"].

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

email : {email}
{format_instructions}
"""

updated_prompt_template = ChatPromptTemplate.from_template(email_template_formatted)
updated_messages = updated_prompt_template.format_messages(email = email_response, format_instructions= format_instructions)

updated_response = llm_model.invoke(updated_messages)

#print(updated_response.content)

output_dict = output_parser.parse(updated_response.content)

#print(type(output_dict))

print(output_dict['cities_to_visit'][0])



