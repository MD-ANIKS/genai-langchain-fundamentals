from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(
  model = 'mistral-embed'
)

vector = embeddings.embed_query("You are goint to learn Gen AI")

print(vector)