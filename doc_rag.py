import streamlit as st
import os
from PIL import Image
import PyPDF2
from crewai_tools import PDFSearchTool
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import warnings
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

st.title("Document RAG")
name='_'
with st.sidebar:
    st.header("Upload Files")
    uploaded_file = st.file_uploader( 
        "Upload PDF or Image files",
        type=["pdf", "png", "jpg", "jpeg"],
    )

if uploaded_file is not None: 
    file_name = uploaded_file.name
    name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    save_path = os.path.join("uploads", file_name)
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        
    save_uploaded_file(uploaded_file, save_path)
    
    if file_extension == ".pdf":
        st.write(f"PDF file {file_name} uploaded successfully.")
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption=file_name)
        st.write(f"Image file {file_name} uploaded successfully.")
            
save_path = os.path.join("uploads", name)
            
pdf = PDFSearchTool(pdf=save_path, config=None)
web=SerperDevTool()

search_agent=Agent(
    role="Search Agent",
    goal="Search for relevant infromation information on the web related to the user {query}",
    backstory=(
        '''
            "You are a search assistant that retrieves and synthesizes accurate information from the web. "
            "Focus on understanding user {query}, searching efficiently, and summarizing results clearly and concisely. "
            "Prioritize reliable sources (academic papers, official reports, or trusted news outlets) and avoid bias. "
            "Organize responses with headers, bullet points, and links using Markdown formatting for readability. "
            "Respect user privacy and avoid restricted or sensitive content."
        '''),
    allow_delegation=True,
    verbose=True,
    markdown=True,
    llm=llm
)

doc_reader_agent=Agent(
    role="Document Reader",
    goal="Read and find relevant information to the user's {query} in the document",
    backstory=(
        '''
            "You are a document reader assistant that specializes in extracting valuable insights from documents. "
            "Your task is to read and comprehend the document to provide accurate and concise information related to user {query}. "
            "Summarize key points, definitions, and examples from the document to address the user's query effectively. "
            "Use Markdown formatting to structure your response clearly and provide references when necessary."
        '''),
    allow_delegation=True,
    verbose=True,
    markdown=True,
    llm=llm
)

blog_writer_agent=Agent(
    role="Professional Blog Writer",
    goal="Take the response from the search agent and doc reader agent and write a compelling and aesthetic blog post on the topic of {query}",
    backstory=(
        """
        You are a blog writer assistant that specializes in creating engaging and informative blog posts.
        Your task is to take the responses from the search agent and document reader agent and craft a well-structured blog post on the topic of {query}.
        Focus on creating a compelling narrative, using clear and concise language, and incorporating visuals if necessary.
        Use the information provided by the agents to create an engaging and informative blog post that educates and entertains the readers.
        """
    ),
    allow_delegation=True,
    verbose=True,
    markdown=True,
    llm=llm
)


search_result = Task(
    description=(
        '''
            "{query}" is the user's query. Your task is to search for relevant information related to this query. "
            "Use the tools provided to you to find accurate and reliable information from the web. "
            "Summarize the search results, highlight key points, and provide references to the sources. "
            "Organize the information in a structured format using Markdown for clarity and readability."
        '''
    ),
    expected_output=(
        '''
            "A detailed summary of the search results related to the user's query "{query}". "
            "The response should include key points, definitions, examples, and references to the sources. "
            "Use Markdown formatting to structure the information clearly and provide links to the sources for further exploration."
            "Give links to the sources for further exploration."
        '''
    ),
    tools=[web],
    agent=search_agent,
)

doc_reader_result = Task(
    description=(
        '''
            "{query}" is the user's query. Your task is to find relevant information in the document. "
            "Use the tools provided to you to extract valuable insights from the document related to this query. "
            "Summarize key points, definitions, and examples from the document to address the user's query effectively. "
            "Organize the information in a structured format using Markdown for clarity and readability."
        '''
    ),
    expected_output=(
        '''
            "A detailed summary of the information extracted from the document related to the user's query "{query}". "
            "The response should include key points, definitions, examples, and references to the document. "
            "Use Markdown formatting to structure the information clearly and provide citations for further reference."
        '''
    ),
    tools=[pdf],
    agent=doc_reader_agent,
)

final_blog_post = Task(
    description=(
        '''
            "{query}" is the user's query. Your task is to write a compelling and aesthetic blog post on the topic of {query}. "
            "Use the information provided by the search agent and document reader agent to craft a well-structured blog post. "
            "Focus on creating a compelling narrative, using clear and concise language, and incorporating visuals if necessary. "
            "Provide valuable insights, examples, and references to create an engaging and informative blog post."
        '''
    ),
    expected_output=(
        '''
            "A well-crafted blog post on the topic of {query} based on the information provided by the search agent and document reader agent. "
            "The blog post should be engaging, informative, and well-structured, with valuable insights, examples, and references. "
            "Use clear and concise language, incorporate visuals if necessary, and provide links to the sources for further exploration."
            "Use markdown formatting to structure the information clearly and provide citations for further reference."
            "Ask search agent for the links to the sources for further exploration."
        '''
    ),
    agent=blog_writer_agent,
)

crew = Crew(
    agents=[doc_reader_agent, search_agent, blog_writer_agent],
    tasks=[doc_reader_result, search_result, final_blog_post],
    verbose=True,
    memory=True
)

st.write("Please provide a query to get started.")
query = st.text_input("Enter your query:")
model_options = ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]
chosen_model = st.selectbox("Select a model:", model_options)
if chosen_model == "gpt-4":
    llm = ChatOpenAI(model="gpt-4")
elif chosen_model == "gpt-4o":
    llm = ChatOpenAI(model="gpt-4o-2024-08-06")
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
if st.button("Submit"):
    result = crew.kickoff(inputs={"query": query})
    st.markdown(result)
    st.write("Model used:", chosen_model)
    
    




    