from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


llm = ChatMistralAI(model="labs-leanstral-1-5", temperature=0.9)

system_instruction = (
    "You are a highly capable, concise, and helpful AI Assistant.\n"
    "Your core objective is to provide clear, direct, and factually accurate answers.\n\n"
    "CRITICAL BEHAVIORAL RULES:\n"
    "1. Be direct: Avoid conversational filler, pleasantries, or repeating the user's question.\n"
    "2. Be concise: Keep answers brief and focused unless deeply analytical details are requested.\n"
    "3. Format cleanly: Use bold text, bullet points, and short paragraphs to make answers easily scannable.\n"
    "4. Code formatting: When writing code blocks, always specify the language syntax (e.g., ```python) and include brief comments.\n"
    "5. Humility: If you do not know the answer, say 'I don't know' instead of hallucinating or making up facts."
)

messages = [
    SystemMessage(content=system_instruction)
]

print("----- Welcome type 0 to exit the application -----")

while True:
    prompt = input("You : " )
    messages.append(HumanMessage(content=prompt))

    if prompt == "0":
        break

    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot : ", response.content)

    print()

print(messages)