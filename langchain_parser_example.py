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

email_template = """
From the following email, extract the following information:

leave_time : when they are leaving for vacation to Europe. If there is an actual time written, use it, if not, write unknown.

leave_from : where they are leaving from, the airport, city name or state if available

cities_to_visit : extract the cities they are planning to visit. If there are more than one then, put them in square brackets like this ["city1", "city2"].

Format the output as JSON with the following keys:
leave_time
leave_from
cities_to_visit

Format the output as **raw JSON only** (no markdown or code block formatting).

email : {email}
"""

promp_template = ChatPromptTemplate.from_template(email_template)

#print(promp_template)

messages = promp_template.format_messages(email = email_response)

response = llm_model.invoke(messages)

print(response.content)