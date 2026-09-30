from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="labs-leanstral-1-5", temperature=0.9, max_tokens=20)

response = llm.invoke("write a poem on ai")

print(response.content)