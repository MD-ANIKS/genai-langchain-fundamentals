from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512},
)


chat_model = ChatHuggingFace(llm=hf)

result = chat_model.invoke("who are you?")

print(result.content)