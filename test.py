import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai

# Load environment variables from the .env file
load_dotenv()

# Set API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Define a prompt template for the chatbot
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

# Create a message history to maintain context
message_history = ChatMessageHistory()

def chatbot_response(user_input):
    # Add user message to history
    message_history.add_user_message(user_input)

    # Prepare the prompt with user input and chat history
    prompt = prompt_template.format(user_input=user_input)
    
    # Get the response from the chat model
    response = chat_model.invoke(prompt)
    
    # Add model response to history
    message_history.add_ai_message(response)
    
    return response

# Example usage
print(chatbot_response("What is LangChain?"))