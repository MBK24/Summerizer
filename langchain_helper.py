from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain 
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

#videoUrl = "https://www.youtube.com/watch?v=f6x9rGu9aGo" test url 

def youtubeVectorDb(videoUrl:str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(videoUrl)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs  = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs,embeddings)
    #return docs  this prints the transcript in chunks 
    return db 
#print(youtubeVectorDb(videoUrl))

def responseFromQuery(db,query,k=4):
    # text-davinci can only handle 4097 tokens 

    docs = db.similarity_search(query, k=k)
    docsPageContent = " ".join([d.page_content for d in docs])

    llm = OpenAI(max_tokens= 3220,model = "gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables=["question","docs"], # the query and the similarity search 
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """, 
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question = query, docs=docsPageContent)
    response = response.replace("\n","")
    return response, docs  