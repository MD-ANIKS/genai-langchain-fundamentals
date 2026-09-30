from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import os
print(os.getenv("HF_TOKEN") is not None)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    # task="text-generation",
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
    # provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("who are you ?")

print(response.content)