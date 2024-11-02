from langchain_openai import OpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

load_dotenv()
video_url = "https://youtu.be/-Osca2Zax4Y"
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

print(create_db_from_youtube_video_url(video_url))

def get_response_from_query(db, query, k=4):
    # Retrieve the most similar documents based on the query
    docs = db.similarity_search(query, k=k)
    docs_page_content = "".join(d.page_content for d in docs)
    
    # Initialize the language model with chat-compatible settings
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Define the prompt in a format suitable for chat models
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            You are a helpful assistant that answers questions about YouTube videos 
            based on the video's transcript.

            Answer the following question: {question}
            By referencing the following video transcript: {docs}

            Only use the factual information from the transcript to answer the question.

            If you feel like you don't have enough information to answer the question, say "I don't know."

            Your answers should be verbose and detailed.
        """
    )

    # Create a chain with the LLM and prompt template
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the input query and documents
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    
    return response, docs