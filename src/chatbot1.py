import glob
import json
import os
import re
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import httpx
from typing import List, Optional, Dict, Any
from langchain.schema import BaseMessage, ChatResult
from langchain.chat_models.base import BaseChatModel
from pydantic import Field, BaseModel
from langchain.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Imports from other local python files
from NEO4J_Graph import Graph
from FHIR_to_graph import resource_to_node, resource_to_edges

# Load environment variables
load_dotenv()

# Environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
USERNAME = os.getenv('NEO4J_USERNAME')
PASSWORD = os.getenv('NEO4J_PASSWORD')
DATABASE = os.getenv('NEO4J_DATABASE')
VLLM_URL = os.getenv('VLLM_URL')
VLLM_MODEL = os.getenv('VLLM_MODEL', 'mistral')

# Configure logging
logger = logging.getLogger(__name__)

class MistralChatModel(BaseChatModel):
    """Custom Mistral chat model to handle parameter conversion and API calls."""
    
    # Define the fields properly
    base_url: str = Field(..., description="Base URL for the Mistral API")
    model: str = Field(default="mistral", description="Model name")
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Private attributes for internal use
    _http_client: Optional[httpx.Client] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        # Handle the max_completion_tokens conversion
        if 'max_completion_tokens' in kwargs:
            kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
        
        # Convert openai_api_base to base_url if provided
        if 'openai_api_base' in kwargs:
            kwargs['base_url'] = kwargs.pop('openai_api_base')
            
        super().__init__(**kwargs)
        self._http_client = httpx.Client(verify=False)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": msg.type, "content": msg.content} for msg in messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if stop:
            payload["stop"] = stop
            
        try:
            response = self._http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            return response.json()
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

# Initialize Neo4j graph
graph = Graph(NEO4J_URI, USERNAME, PASSWORD, DATABASE)
print(graph.resource_metrics())
print(graph.database_metrics())

# Vector store management
vector_index = None

def refresh_vector_index():
    """Refresh the vector index from Neo4j."""
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
    """Get or create vector index."""
    global vector_index
    if vector_index is None:
        refresh_vector_index()
    return vector_index

# Prompt templates
my_prompt = '''
System: The following information contains entries about the patient. 
Use the primary entry and then the secondary entries to answer the user's question.
Each entry is its own type of data and secondary entries are supporting data for the primary one.
Ensure that you always look into secondary entries for information.
You should restrict your answer to using the information in the entries provided. but be very detailed in your answer.
Ensure no detail in primary or secondary entries are missed.
If you are asked about the patient's name and one the entries is of type patient, you should look for the first given name and family name and answer with: [given] [family]
Highlights the important information in the entries to make it easier for the user to understand.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
----------------
User: {question}
'''

prompt = PromptTemplate.from_template(my_prompt)

# Initialize model
vllm_model = get_vllm_instance()
k_nearest = int(os.getenv('K_NEAREST', 500))

def date_for_question(question_to_find_date: str, model: Optional[BaseChatModel] = None) -> Optional[str]:
    """Extract date from a question using the LLM."""
    _llm = model if model else get_vllm_instance()
    prompt = f'''
    system:Given the following question from the user, extract the date the question is asking about.
    Return the answer formatted as JSON only, as a single line.
    Use the form: {{"date":"[THE DATE IN THE QUESTION]"}}
    Use the date format of month/day/year.
    Use two digits for the month and day.
    Use four digits for the year.
    If there is no date, return {{"date":"none"}}.
    
    user:{question_to_find_date}
    '''
    try:
        _response = _llm.invoke(prompt)
        date_json = json.loads(_response.content)
        return date_json['date']
    except Exception as e:
        logger.error(f"Error processing date: {str(e)}")
        return None

def create_contextualized_vectorstore_with_date(date_to_look_for: str) -> Neo4jVector:
    """Create a contextualized vector store based on the date."""
    if date_to_look_for == 'none':
        contextualize_query = """
        match (node)<-[]->(sc:resource)
        with node.text as self, reduce(s="", item in collect(distinct sc.text)[..5] | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata limit 10
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    else:
        contextualize_query = f"""
        match (node)<-[]->(sc:resource)
        where exists {{
             (node)-[]->(d:Date {{id: '{date_to_look_for}'}})
        }}
        with node.text as self, reduce(s="", item in collect(distinct sc.text)[..5] | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {{}} as metadata limit 10
        return "Primary Entry:\n" + self + ctxt as text, score, metadata
        """
    
    return Neo4jVector.from_existing_index(
        HuggingFaceBgeEmbeddings(model_name=os.getenv('EMBEDDING_MODEL')),
        url=NEO4J_URI,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        index_name='fhir_text',
        retrieval_query=contextualize_query,
    )

# Custom document prompt
CUSTOM_DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)

def ask_date_question(question_to_ask: str, model: Optional[BaseChatModel] = None, prompt_to_use: PromptTemplate = prompt) -> Dict[str, Any]:
    """Main function to process questions and generate answers."""
    logger.info(f"Received question: {question_to_ask}")
    
    model = model or vllm_model
    _date_str = date_for_question(question_to_ask, model)
    logger.info(f"Extracted date: {_date_str}")
    
    _index = create_contextualized_vectorstore_with_date(_date_str)
    retriever = _index.as_retriever(search_kwargs={'k': k_nearest})
    
    doc_var_name = 'summaries' if isinstance(prompt_to_use, PromptTemplate) and 'summaries' in prompt_to_use.input_variables else 'context'
    
    try:
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_to_use,
                "document_variable_name": doc_var_name,
                "document_prompt": CUSTOM_DOCUMENT_PROMPT,
                "max_tokens": 1024
            }
        )
        
        result = qa_chain.invoke({"question": question_to_ask})
        logger.info(f"QA chain result: {result}")
        
        answer = result.get('answer', '')
        sources = result.get('sources', 'No sources provided')
        
        if not answer:
            logger.warning("No answer generated")
            answer = "I'm sorry, but I couldn't generate an answer based on the available information."
        
        return {
            "formatted_answer": answer,
            "raw_answer": answer,
            "date": _date_str if _date_str != "none" else None,
            "confidence": 0.95 if answer else 0,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error in QA chain: {str(e)}")
        return {
            "formatted_answer": f"An error occurred: {str(e)}",
            "raw_answer": "",
            "date": _date_str if _date_str != "none" else None,
            "confidence": 0,
            "sources": ""
        }

def get_all_patient_names() -> List[str]:
    """Retrieve all patient names from the database."""
    query = """
    MATCH (p:Patient)
    RETURN p.name as name
    LIMIT 100
    """
    try:
        results = graph.query(query)
        patient_names = []
        if results and isinstance(results, tuple) and len(results) > 0:
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

def get_all_hospital_names() -> List[str]:
    """Retrieve all hospital/organization names from the database."""
    query = """
    MATCH (o:Organization)
    RETURN o.name as name
    LIMIT 100
    """
    try:
        results = graph.query(query)
        organization_names = []
        if results and isinstance(results, tuple) and len(results) > 0:
            org_list = results[0]
            for org in org_list:
                if isinstance(org, list) and len(org) > 0:
                    organization_names.append(org[0])
                elif isinstance(org, str):
                    organization_names.append(org)
        
        if not organization_names:
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

# Export public interface
__all__ = ['ask_date_question', 'get_all_patient_names', 'get_all_hospital_names', 'refresh_vector_index']
