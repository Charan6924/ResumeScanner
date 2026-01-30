from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('Pinecone works')
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)
print('OpenAI API works')

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")


supabase = create_client(str(url),str(key))

print('Supabase works')
