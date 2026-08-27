from dotenv import load_dotenv
from decouple import config
from google import genai


client = genai.Client(api_key=config("GEMINI_API_KEYS"))

result=client.models.embed_content(
    model="gemini-embedding-2.0",
    contents="Hello, my dog is cute"
)
print("Embedding result:", result)