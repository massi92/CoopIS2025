import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any

import chromadb
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import Ollama, OllamaLLM
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# Langchain Agent and Tool imports
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import Agent

class MultiVectorRAGPipeline:
    def __init__(self,
                 atomic_data_folder: str,
                 scenarios_folder: str,
                 embedding_model='mixedbread-ai/mxbai-embed-large-v1',
                 llm_model='granite3-dense-8b'):
        """
        Initialize MultiVector RAG Pipeline with Ollama and Hugging Face embeddings

        :param atomic_data_folder: Path to folder with atomic data JSON files
        :param scenarios_folder: Path to folder with scenario XML files
        :param embedding_model: Hugging Face embedding model
        :param llm_model: Ollama language model for summarization and RAG
        """
        # Set up Hugging Face embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Set up Ollama LLM
        self.llm = OllamaLLM(
            model=llm_model,
            temperature=0.3
        )

        # Set up folders
        self.atomic_data_folder = atomic_data_folder
        self.scenarios_folder = scenarios_folder

        # Text splitter for chunk generation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # MultiVector Retriever components
        self.atomic_retriever = None
        self.scenario_retriever = None

    def load_and_process_json_files(self) -> List[Document]:
        """
        Load, process, and generate multiple representations of JSON files

        Returns:
        - List of original documents
        - List of summary documents
        - List of chunked documents
        """
        original_docs = []
        summary_docs = []
        chunked_docs = []

        for filename in os.listdir(self.atomic_data_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(self.atomic_data_folder, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)

                        # Extract service type
                        service_type = data.get('x-atomic-service-type', 'unknown')

                        # Original Document
                        original_doc = Document(
                            page_content=json.dumps(data),
                            metadata={
                                'source': filename,
                                'atomic_service_type': service_type
                            }
                        )
                        original_docs.append(original_doc)

                        # Generate Summary
                        summary = self.llm(f"Summarize the key points of this document:\n{original_doc.page_content}")
                        summary_doc = Document(
                            page_content=summary,
                            metadata={
                                'source': filename,
                                'atomic_service_type': service_type,
                                'doc_type': 'summary'
                            }
                        )
                        summary_docs.append(summary_doc)

                        # Generate Chunks
                        doc_chunks = self.text_splitter.split_documents([original_doc])
                        for chunk in doc_chunks:
                            chunk.metadata.update({
                                'source': filename,
                                'atomic_service_type': service_type,
                                'doc_type': 'chunk'
                            })
                        chunked_docs.extend(doc_chunks)

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

        return original_docs, summary_docs, chunked_docs

    def load_and_process_xml_files(self) -> List[Document]:
        """
        Load, process, and generate multiple representations of XML files

        Returns similar to load_and_process_json_files
        """
        original_docs = []
        summary_docs = []
        chunked_docs = []

        for filename in os.listdir(self.scenarios_folder):
            if filename.endswith('.xml'):
                filepath = os.path.join(self.scenarios_folder, filename)
                try:
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    xml_content = ET.tostring(root, encoding='unicode')

                    # Original Document
                    original_doc = Document(
                        page_content=xml_content,
                        metadata={'source': filename}
                    )
                    original_docs.append(original_doc)

                    # Generate Summary
                    summary = self.llm(f"Summarize the key points of this XML document:\n{xml_content}")
                    summary_doc = Document(
                        page_content=summary,
                        metadata={
                            'source': filename,
                            'doc_type': 'summary'
                        }
                    )
                    summary_docs.append(summary_doc)

                    # Generate Chunks
                    doc_chunks = self.text_splitter.split_documents([original_doc])
                    for chunk in doc_chunks:
                        chunk.metadata.update({
                            'source': filename,
                            'doc_type': 'chunk'
                        })
                    chunked_docs.extend(doc_chunks)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        return original_docs, summary_docs, chunked_docs

    def create_multi_vector_retriever(self, collection_type: str):
        """
        Create a MultiVector Retriever for the specified collection type

        :param collection_type: 'atomic' or 'scenario'
        """
        # Load and process documents
        if collection_type == 'atomic':
            original_docs, summary_docs, chunked_docs = self.load_and_process_json_files()
            docstore = InMemoryStore()
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name="atomic_multi_vector"
            )
        else:
            original_docs, summary_docs, chunked_docs = self.load_and_process_xml_files()
            docstore = InMemoryStore()
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name="scenario_multi_vector"
            )

        # Create MultiVector Retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",
        )

        # Add documents to retriever
        doc_ids = []
        for doc in original_docs:
            doc_id = retriever.docstore.add_documents([doc])
            doc_ids.append(doc_id[0])
            retriever.vectorstore.add_documents([doc], ids=[doc_id[0]])

        for doc in summary_docs:
            doc_id = retriever.docstore.add_documents([doc])
            retriever.vectorstore.add_documents([doc], ids=[doc_id[0]])

        for doc in chunked_docs:
            doc_id = retriever.docstore.add_documents([doc])
            retriever.vectorstore.add_documents([doc], ids=[doc_id[0]])

        return retriever

    def query(
        self,
        query: str,
        prompt: str,
        collection_type: str = 'atomic',
        service_type: Optional[str] = None,
        k: int = 3
    ):
        """
        Run a query using MultiVector Retriever

        :param query: User's query
        :param prompt: Prompt template to use for generation
        :param collection_type: 'atomic' or 'scenario'
        :param service_type: Optional filter for atomic service type
        :param k: Number of documents to retrieve
        :return: RAG query results
        """
        # Create MultiVector Retriever
        if collection_type == 'atomic':
            retriever = self.create_multi_vector_retriever('atomic')
            # Apply service type filter if specified
            if service_type:
                filtered_docs = [
                    doc for doc in retriever.docstore.mget(retriever.docstore.yield_keys())
                    if doc.metadata.get('atomic_service_type') == service_type
                ]
                # Recreate retriever with filtered docs
                retriever.vectorstore.add_documents(filtered_docs)
        else:
            retriever = self.create_multi_vector_retriever('scenario')

        # Create custom prompt template
        rag_prompt = PromptTemplate(
            template=prompt + "\n\nContext:\n{summaries}\n\nQuestion: {question}",
            input_variables=["summaries", "question"]
        )

        # Create RAG chain
        rag_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": rag_prompt}
        )

        # Run query
        result = rag_chain({"question": query})

        return result

    def create_atomic_data_tool(self, prompt: str) -> BaseTool:
        """
        Create a tool for querying atomic data services

        :param prompt: Custom prompt template for atomic data query
        """
        class AtomicDataTool(BaseTool):
            name = "atomic_data_query"
            description = "Useful for querying atomic data services with optional type filtering"

            def __init__(self, rag_pipeline, prompt):
                super().__init__()
                self.rag_pipeline = rag_pipeline
                self.prompt = prompt

            def _run(self, query: str, service_type: Optional[str] = None) -> str:
                """
                Run query on atomic data services
                """
                result = self.rag_pipeline.query(
                    query,
                    self.prompt,
                    collection_type='atomic',
                    service_type=service_type,
                    k=3
                )

                return result['answer']

            def _arun(self, query: str):
                raise NotImplementedError("Async not supported")

        return AtomicDataTool(self, prompt)

    def create_scenario_tool(self, prompt: str) -> BaseTool:
        """
        Create a tool for querying scenario documents

        :param prompt: Custom prompt template for scenario query
        """
        class ScenarioTool(BaseTool):
            name = "scenario_query"
            description = "Useful for querying scenario documents and extracting detailed context"

            def __init__(self, rag_pipeline, prompt):
                super().__init__()
                self.rag_pipeline = rag_pipeline
                self.prompt = prompt

            def _run(self, query: str) -> str:
                """
                Run query on scenario documents
                """
                result = self.rag_pipeline.query(
                    query,
                    self.prompt,
                    collection_type='scenario',
                    k=3
                )

                return result['answer']

            def _arun(self, query: str):
                raise NotImplementedError("Async not supported")

        return ScenarioTool(self, prompt)

    def create_routing_agent(self, atomic_prompt: str, scenario_prompt: str):
        """
        Create a routing agent that can choose between atomic data and scenario tools

        :param atomic_prompt: Prompt for atomic data tool
        :param scenario_prompt: Prompt for scenario tool
        """
        # Create tools with custom prompts
        atomic_tool = self.create_atomic_data_tool(atomic_prompt)
        scenario_tool = self.create_scenario_tool(scenario_prompt)

        # Initialize the routing agent
        tools = [atomic_tool, scenario_tool]

        routing_agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        return routing_agent

    def route_query(self, query: str, atomic_prompt: str, scenario_prompt: str):
        """
        Route the query using the multi-vector agent

        :param query: User's input query
        :param atomic_prompt: Prompt for atomic data tool
        :param scenario_prompt: Prompt for scenario tool
        :return: Query result from the most appropriate tool
        """
        routing_agent = self.create_routing_agent(atomic_prompt, scenario_prompt)

        # Run the query through the routing agent
        response = routing_agent.run(query)

        return response

def main():
    # Initialize MultiVector RAG Pipeline
    rag_pipeline = MultiVectorRAGPipeline(
        atomic_data_folder='./atomic_data_services',
        scenarios_folder='./analysis_scenarios'
    )

    # Custom prompts for each collection
    atomic_services_prompt_template = PromptTemplate.from_template("...") #template regarding data services

    scenarios_prompt_template = PromptTemplate.from_template("...") #template regarding data services

    # Query
    result = rag_pipeline.route_query(
        query,
        atomic_services_prompt_template.format(...), #to fill placeholders 
        scenarios_prompt_template.format(...) #to fill placeholders
    )
    print("Response:", result)

if __name__ == "__main__":
    main()
