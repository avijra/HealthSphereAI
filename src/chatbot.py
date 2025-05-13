import glob
import json
import os
import re
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import httpx

# Set up logging first
logger = logging.getLogger(__name__)

load_dotenv()
import httpx
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Imports from other local python files
from NEO4J_Graph import Graph
from FHIR_to_graph import resource_to_node, resource_to_edges

NEO4J_URI = os.getenv('NEO4J_URI')
USERNAME = os.getenv('NEO4J_USERNAME')
PASSWORD = os.getenv('NEO4J_PASSWORD')
DATABASE = os.getenv('NEO4J_DATABASE')
VLLM_URL = os.getenv('VLLM_URL')
VLLM_MODEL = os.getenv('VLLM_MODEL', 'mistral')

from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage
from langchain.chat_models.base import BaseChatModel
from pydantic import Field, BaseModel
from typing import List, Optional, Dict, Any

class MistralChatModel(BaseChatModel):
    """Custom Mistral chat model to handle parameter conversion and API calls."""
    
    base_url: str = Field(..., description="Base URL for the Mistral API")
    model: str = Field(default="mistral", description="Model name")
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    _http_client: Optional[httpx.Client] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        if 'max_completion_tokens' in kwargs:
            kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
        if 'openai_api_base' in kwargs:
            kwargs['base_url'] = kwargs.pop('openai_api_base')
            
        super().__init__(**kwargs)
        self._http_client = httpx.Client(verify=False)
        logger.info(f"Initialized MistralChatModel with base_url: {self.base_url}, model: {self.model}")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Convert messages to the format expected by Mistral API
        formatted_messages = []
        for msg in messages:
            if msg.type == "system":
                formatted_messages.append({"role": "system", "content": msg.content})
            elif msg.type == "human":
                formatted_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        if stop:
            payload["stop"] = stop
            
        try:
            logger.info(f"Making request to Mistral API at: {self.base_url}/chat/completions")
            logger.info(f"Request headers: {headers}")
            logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = self._http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response body: {response.text}")
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                raise Exception(f"API Error: {response.status_code} - {response.text}")
                
            response_data = response.json()
            
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise Exception("No choices in response")
            
            # Extract the message content from the response
            message_content = response_data["choices"][0]["message"]["content"]
            
            # Create the ChatGeneration with proper structure
            generation = ChatGeneration(
                message=AIMessage(content=message_content),
                generation_info=response_data["choices"][0]
            )
            
            # Create the ChatResult with proper structure
            return ChatResult(
                generations=[generation],
                llm_output={
                    "token_usage": response_data.get("usage", {}),
                    "model_name": self.model
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Mistral API call: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def __del__(self):
        if self._http_client:
            self._http_client.close()

def get_vllm_instance():
    """Create and return a configured Mistral chat model instance."""
    return MistralChatModel(
        base_url=VLLM_URL,
        model=VLLM_MODEL,
        max_tokens=1024,
        temperature=1,
        request_timeout=300,
    )

graph = Graph(NEO4J_URI, USERNAME, PASSWORD, DATABASE)
print(graph.resource_metrics())
print(graph.database_metrics())

vector_index = None

def refresh_vector_index():
    global vector_index
    vector_index = Neo4jVector.from_existing_index(
        HuggingFaceBgeEmbeddings(model_name=os.getenv('EMBEDDING_MODEL')),
        url=NEO4J_URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        index_name='fhir_text'
    )

def get_vector_index():
    global vector_index
    if vector_index is None:
        refresh_vector_index()
    return vector_index

default_prompt='''
System: Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
Human: {question}
'''

my_prompt='''
System: The following information contains entries about the patient. 
Use the primary entry and then the secondary entries to answer the user's question.
Each entry is its own type of data and secondary entries are supporting data for the primary one.
Ensure that you always look into secondary entries for information.
You should restrict your answer to using the information in the entries provided. but be very detailed in your answer.Ensure no detail in primary or secondary entries are missed.
If you are asked about the patient's name and one the entries is of type patient, you should look for the first given name and family name and answer with: [given] [family]
Highlights the important information in the entries to make it easier for the user to understand.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
----------------
User: {question}
'''

my_prompt_2='''
System: The context below contains entries about the patient's healthcare. 
Please limit your answer to the information provided in the context. Do not make up facts. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If you are asked about the patient's name and one the entries is of type patient, you should look for the first given name and family name and answer with: [given] [family]
----------------
{context}
Human: {question}
'''

prompt = PromptTemplate.from_template(my_prompt)

#ollama_model = os.getenv('OLLAMA_MODEL', 'mistral') # mistral, orca-mini, llama2
#ollama_model = get_ollama_instance()
vllm_model = get_vllm_instance()

k_nearest = int(os.getenv('K_NEAREST', 500))

def date_for_question(question_to_find_date, model):
    _llm = model if model else get_vllm_instance()
    prompt = f'''
    system:Given the following question from the user, extract the date the question is asking about.
    Return the answer formatted as JSON only, as a single line.
    Use the form:
    
    {{"date":"[THE DATE IN THE QUESTION]"}}
    
    Use the date format of month/day/year.
    Use two digits for the month and day.
    Use four digits for the year.
    So 3/4/23 should be returned as {{"date":"03/04/2023"}}.
    So 04/14/89 should be returned as {{"date":"04/14/1989"}}.
    
    Please do not include any special formatting characters, like new lines or "\\n".
    Please do not include the word "json".
    Please do not include triple quotes.
    
    If there is no date, do not make one up. 
    If there is no date return the word "none", like: {{"date":"none"}}
    
    user:{question_to_find_date}
    '''
    _response = _llm.invoke(prompt)
    try:
        date_json = json.loads(_response.content)
        return date_json['date']
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {_response.content}")
        return None

def create_contextualized_vectorstore_with_date(date_to_look_for):
    if date_to_look_for == 'none':
        contextualize_query_with_date = """
        match (node)<-[]->(sc:resource)
        with node.text as self, reduce(s="", item in collect(distinct sc.text)[..5] | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata limit 10
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    else:
        contextualize_query_with_date = f"""
        match (node)<-[]->(sc:resource)
        where exists {{
             (node)-[]->(d:Date {{id: '{date_to_look_for}'}})
        }}
        with node.text as self, reduce(s="", item in collect(distinct sc.text)[..5] | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {{}} as metadata limit 10
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    
    _contextualized_vectorstore_with_date = Neo4jVector.from_existing_index(
        HuggingFaceBgeEmbeddings(model_name=os.getenv('EMBEDDING_MODEL')),
        url=NEO4J_URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        index_name='fhir_text',
        retrieval_query=contextualize_query_with_date,
    )
    return _contextualized_vectorstore_with_date

# Define a custom document prompt that doesn't require the 'source' metadata
CUSTOM_DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)

def ask_date_question(question_to_ask, model=vllm_model, prompt_to_use=prompt):
    logger.info(f"Received question: {question_to_ask}")
    _date_str = date_for_question(question_to_ask, model)
    logger.info(f"Extracted date: {_date_str}")
    _index = create_contextualized_vectorstore_with_date(_date_str)
    
    # Create a retriever
    retriever = _index.as_retriever(search_kwargs={'k': k_nearest})
    
    # Determine the correct document variable name
    if isinstance(prompt_to_use, PromptTemplate):
        doc_var_name = 'summaries' if 'summaries' in prompt_to_use.input_variables else 'context'
    else:
        doc_var_name = 'context'  # default to 'context' if not a PromptTemplate
    logger.info(f"Using document variable name: {doc_var_name}")
    
    # Create the RetrievalQAWithSourcesChain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_to_use,
            "document_variable_name": doc_var_name,
            "document_prompt": CUSTOM_DOCUMENT_PROMPT
        }
    )
    
    # Run the query
    try:
        logger.info("Running QA chain")
        result = qa_chain.invoke({"question": question_to_ask})
        logger.info(f"QA chain result: {result}")
    except Exception as e:
        logger.error(f"Error running QA chain: {str(e)}")
        return {
            "formatted_answer": f"An error occurred: {str(e)}",
            "raw_answer": "",
            "date": _date_str if _date_str != "none" else None,
            "confidence": 0,
            "sources": ""
        }
    
    # Extract the answer and sources
    answer = result.get('answer', '')
    sources = result.get('sources', 'No sources provided')
    
    if not answer:
        logger.warning("No answer generated")
        answer = "I'm sorry, but I couldn't generate an answer based on the available information."
    
    # Format the final response
    formatted_response = f"""{answer}"""
    
    logger.info(f"Formatted response: {formatted_response}")
    
    return {
        "formatted_answer": formatted_response,
        "raw_answer": answer,
        "date": _date_str if _date_str != "none" else None,
        "confidence": 0.95 if answer else 0,
        "sources": sources
    }

logger = logging.getLogger(__name__)
def get_all_patient_names():
    query = """
    MATCH (p:Patient)
    RETURN p.name as name
    LIMIT 100
    """
    try:
        results = graph.query(query)
        logger.debug(f"Query results: {results}")

        patient_names = []
        if results and isinstance(results, tuple) and len(results) > 0:
            # Extract the list of patient names from the first element of the tuple
            patient_list = results[0]
            for patient in patient_list:
                if isinstance(patient, list) and len(patient) > 0:
                    patient_names.append(patient[0])
                elif isinstance(patient, str):
                    patient_names.append(patient)

        logger.info(f"Retrieved {len(patient_names)} patient names")
        return patient_names
    except Exception as e:
        logger.error(f"Error fetching patient names: {str(e)}", exc_info=True)
        return []

def get_all_hospital_names():
    query = """
    MATCH (o:Organization)
    RETURN o.name as name
    LIMIT 100
    """
    try:
        results = graph.query(query)
        logger.debug(f"Query results for organizations: {results}")

        organization_names = []
        if results and isinstance(results, tuple) and len(results) > 0:
            org_list = results[0]
            for org in org_list:
                if isinstance(org, list) and len(org) > 0:
                    organization_names.append(org[0])
                elif isinstance(org, str):
                    organization_names.append(org)

        logger.info(f"Retrieved {len(organization_names)} organization names")
        
        if len(organization_names) == 0:
            # If no organizations found, let's check what types of nodes exist
            check_query = """
            MATCH (n)
            RETURN DISTINCT labels(n) as node_types
            """
            check_results = graph.query(check_query)
            logger.debug(f"Available node types: {check_results}")

        return organization_names
    except Exception as e:
        logger.error(f"Error fetching organization names: {str(e)}", exc_info=True)
        return []

# Make sure this line is at the end of your chatbot.py file
__all__ = ['ask_date_question', 'get_all_patient_names', 'get_all_hospital_names', 'refresh_vector_index']
