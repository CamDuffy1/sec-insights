import os
import numpy as np
import asyncio
from app.chat.engine import get_chat_engine, get_tool_service_context, fetch_and_read_document, index_to_query_engine, build_doc_id_to_index_map
from app.chat.messaging import ChatCallbackHandler
from app import schema
import requests
import anyio
from typing import List
from llama_index import StorageContext, load_index_from_storage
from llama_index.schema import Document as LlamaIndexDocument
from llama_index.schema import TextNode
from llama_index import ServiceContext
from llama_index.node_parser.text.sentence_window import SentenceWindowNodeParser
from llama_index.response.schema import Response
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
import random
from llama_index.evaluation import (
    DatasetGenerator,
    QueryResponseDataset,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    PairwiseComparisonEvaluator,
    BatchEvalRunner
)
from llama_index.evaluation.eval_utils import (
    get_responses,
    get_results_df
)
from collections import defaultdict
import pandas as pd
# from IPython.display import display
from llama_index.llms.openai import OpenAI


def create_conversation_via_api(document_ids):
    '''
    Create a new conversation by making a POST request to the conversation API endpoint.
    Args:
        document_ids: The IDs of the documents to include in the conversation.
    Returns: The ID of the conversation created
    '''
    base_url = "http://localhost:8000"
    # base_url = "https://improved-capybara-pjww49wwqvr3rv7x-8000.app.github.dev"
    req_body = {"document_ids": document_ids}
    response = requests.post(f"{base_url}/api/conversation/", json=req_body)
    print(response)
    if response.status_code == 200:
        conversation_id = response.json()["id"]
        print(f"Created conversation with ID {conversation_id}")
        return conversation_id
    else:
        print(f"Error: {response.text}")
        return False
    
def get_conversation_via_api(conversation_id):
    '''
    Gets a conversation by making a GET request to the conversation API endpoint.
    Args:
        conversation_id: The ID of the conversation to get
    Returns: The response object to the API request.
    '''
    base_url = "http://localhost:8000"
    req_body = {"conversation_id": conversation_id}
    response = requests.get(f"{base_url}/api/conversation/", json=req_body)
    print(type(response))
    print(response)
    return response

def get_document_via_api(document_id):
    '''
    Gets a document by making a GET request to the document API endpoint.
    Args:
        document_id: The ID of the conversation to get
    Returns: A schema.Document object of the retrieved document.
    '''
    base_url = "http://localhost:8000"
    response = requests.get(f"{base_url}/api/document/{document_id}")
    data = response.json()
    doc_args = {
        "id": data['id'],
        "created_at": data['created_at'],
        "updated_at": data['updated_at'],
        "url": data['url'],
        "metadata_map":  data['metadata_map']
    }
    doc = schema.Document(**doc_args)
    return doc

def get_dummy_doc():
    """
    This function is created as a workaround to the breaking changes to the FAST API component
    caused by the upgrade from Pydantic V1 -> V2.
    It uses the information about an SEC document that could be retrieved from the app's FAST API docs
    before the breaking change.
    Run this function after setting up localstack, seeding the local db, and migrating the db.

    Retruns:
        A Pydantic Document schema object that can be used to create a conversation and run the query engine against.
    """
    doc_args = {
        "id": "4d24de4e-63ee-4af5-9c97-ccae008ad887",
        "created_at": "2024-02-13T03:56:11.322253",
        "updated_at": "2024-02-13T03:56:11.322253",
        "url": "http://llama-app-web-assets-local.s3-website.localhost.localstack.cloud:4566/sec-edgar-filings/0001326801/10-K/0001326801-23-000013/primary-document.pdf",
        "metadata_map":  {
            "sec_document": {
                "cik": "0001326801",
                "year": 2022,
                "doc_type": "10-K",
                "company_name": "Meta Platforms, Inc.",
                "company_ticker": "META",
                "accession_number": "0001326801-23-000013",
                "filed_as_of_date": "2023-02-02T00:00:00",
                "date_as_of_change": "2023-02-01T00:00:00",
                "period_of_report_date": "2022-12-31T00:00:00"
            }
        }
    }
    doc = schema.Document(**doc_args)
    return doc

def format_pdf_text(text):
    # Replace tabs and multiple new lines with a single space
    if text != None:
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        text = " ".join(text.split())
    return text

def pprint_sentence_window(response: Response, node: int = None) -> None:
    '''
    Function to pretty print the setence-window context vs the original setence retrieved.
    Args:
        response (Resonse): llama_index.response.schema.Response object
        node (int): The node to print sentence-window vs original sentence for. Use when only want to print for a single node.
    Returns: None
    '''
    if node != None:
        print(f"################## SOURCE NODE {node} ##################")
        window = format_pdf_text(response.source_nodes[node].node.metadata.get("window"))
        sentence = format_pdf_text(response.source_nodes[node].node.metadata.get("original_text"))
        print(f"WINDOW: {window}")
        print("------------------")
        print(f"ORIGINAL SENTENCE: {sentence}")
        return
    else:
        for i in range(len(response.source_nodes)):
            print(f"\n################## SOURCE NODE {i+1} ##################")
            window = format_pdf_text(response.source_nodes[i].node.metadata.get("window"))
            sentence = format_pdf_text(response.source_nodes[i].node.metadata.get("original_text"))
            print(f"WINDOW: {window}")
            print("------------------")
            print(f"ORIGINAL SENTENCE: {sentence}")

def get_nodes(document: LlamaIndexDocument, node_parser: SentenceWindowNodeParser):

    document = fetch_and_read_document(document)
    nodes = node_parser.get_nodes_from_documents(document)

    '''Code to get some visibility into nodes'''
    print(f"Total nodes: {len(nodes)}")
    for i in range(800, 830):
        print(f"NODE {i} TEXT: {format_pdf_text(nodes[i].text)}")
        print(f"NODE {i} WINDOW: {format_pdf_text(nodes[i].metadata.get('window'))}")

    return nodes

async def generate_dataset(nodes: List[TextNode], service_context: ServiceContext, num_nodes_eval: int = 2, file_path: str = '/workspaces/sec-insights/backend/eval/eval_dataset.json') -> QueryResponseDataset:
    '''
    Generate a dataset to be used for evaluation.
    Check if dataset already exists at file_path. If not, generate it and save it there so it does not
    need to be generated again.
    Args:
        nodes (List[TextNode]): Text nodes from the document being used for evaluation.
        service_context (ServiceContext): The LlamaIndex Service Context to use to generate questions from nodes for the evaluation dataset.
        num_nodes_eval (int): The number of nodes (randomly sampled from total nodes) to use for generating evaluation questions.
        file_path (str): The path to save the evaluation dataset file so it does not have to be created again.
    Returns:
        The QueryResponseDataset object of the evaluation dataset.
    '''
    if not os.path.exists(file_path):
        print("Generating evaluation dataset")
        sample_eval_nodes = random.sample(nodes[500:1500], num_nodes_eval)

        dataset_generator = DatasetGenerator(
            nodes=sample_eval_nodes,
            # llm=OpenAI(model="gpt-4"),
            service_context=service_context,
            num_questions_per_chunk=2,
            show_progress=True,
        )
        eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()
        eval_dataset.save_json("/workspaces/sec-insights/backend/eval/eval_dataset.json")
        print(f"Saved evaluation dataset at: {file_path}")
    else:
        print(f"Evaluation dataset already exists at: {file_path}")
        eval_dataset = QueryResponseDataset.from_json(file_path)
    return eval_dataset

async def evaluate(original_nodes: List[TextNode], sentence_window_index: VectorStoreIndex, eval_dataset: QueryResponseDataset, max_samples: int = 2):
    '''
    Evaluate responses based on the following metrics:
        Correctness:            The correctness of a response - A score between 1 (worst) and 5 (best).
        Semantic Similarity:    The similarity between embeddings of the generated answer and reference answer.
        Relevance:              The relevance of retrieved context and response to the query. Considers the query string, retrieved context, and response string.
        Faithfulness:           How well the response is supported by the retrieved context (i.e., Is there hallucination?)
    Saves the evaluation in csv format at: /workspaces/sec-insights/backend/eval/results.csv
    Args:
        original_nodes(List[TextNode]):
            Nodes parsed using the original NodeParser from SEC-Insights. These nodes are used to 
            These nodes are used to create a baseline index for comparing the performance of other RAG configurations.
        sentence_window_index(VectorStoreIndex):
            The VectorStoreIndex created using the sentence-window node parser.
        eval_dataset(QueryResponseDataset):
            The Query Response dataset containing query-resposne pairs used to evaluate performance.
        max_samples(int):
            The number of queries to use from the Query Response dataset to evaluate performance.
    Returns:
        results_df(DataFrame): Pandas DataFrame containing the evaluated response metrics.
            
    Note - The following code snippet had to be added to the CorrectnessEvaluator class within the llama-index v"0.9.7" package at llama_index/evaluation/correctness.py before line: score_str, reasoning_str = eval_response.split("\n", 1)
    This avoids an error where eval_resonse is created beginning with a newline character, resulting in an error trying to convert an empty str to a float on line: score = float(score_str).
    Code snippet:
        print(f"eval_response: {eval_response}\n")
        if eval_response[0] == '\n':
            print("removing newline")
            eval_response = eval_response[1:]
            print(f"updated eval_response: {eval_response}\n")            
    '''
    evaluator_c = CorrectnessEvaluator()
    evaluator_s = SemanticSimilarityEvaluator()
    evaluator_r = RelevancyEvaluator()
    evaluator_f = FaithfulnessEvaluator()
    # pairwise_evaluator = PairwiseComparisonEvaluator()

    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

    PERSIST_DIR = '/workspaces/sec-insights/backend/eval/storage'           # local dir to store original index for evaluation commparison
    if not os.path.exists(PERSIST_DIR):                                     # check if storage already exists
        index_original = VectorStoreIndex(original_nodes)                   # create the index
        index_original.storage_context.persist(persist_dir=PERSIST_DIR)     # store it for later
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)     # load the existing index
        index_original = load_index_from_storage(storage_context)

    query_engine_original = index_original.as_query_engine(                 # construct base query engine (for comparison)
        similarity_top_k=3
    )
    query_engine_sentence_window = sentence_window_index.as_query_engine(   # construct sentence window query engine
        similarity_top_k=2,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
    )

    pred_responses_original = get_responses(
        eval_qs[:max_samples], query_engine_original, show_progress=True
    )
    pred_responses_sentence_window = get_responses(
        eval_qs[:max_samples], query_engine_sentence_window, show_progress=True
    )

    pred_responses_original_strs = [str(p) for p in pred_responses_original]
    pred_responses_sentence_window_strs = [str(p) for p in pred_responses_sentence_window]

    evaluator_dict = {
        "correctness": evaluator_c,
        "faithfulness": evaluator_f,
        "relevancy": evaluator_r,
        "semantic_similarity": evaluator_s,
    }
    batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

    # '''Generate output for troublshooting'''
    # queries=eval_qs[:max_samples],
    # print("############### ORIGINAL RETRIEVER ############### ")
    # print("############### QUERIES ############### ")
    # for i in queries:
    #     print(f"type: {type(i)} query: {i}")
    # print()
    # print("############### RESPONSES ############### ")
    # for j in pred_responses_original:
    #     print(f"type: {type(j)} query: {j}")
    # print()
    # print("############### REFERENCE RESPONSE STRINGS ############### ")
    # for k in ref_response_strs:
    #     print(f"type: {type(k)} query: {k}")
    # print()

    # print("############### SENTENCE WINDOW RETRIEVER ############### ")
    # print("############### QUERIES ############### ")
    # for l in queries:
    #     print(f"type: {type(l)} query: {l}")
    # print()
    # print("############### RESPONSES ############### ")
    # for m in pred_responses_sentence_window:
    #     print(f"type: {type(m)} query: {m}")
    # print()
    # print("############### REFERENCE RESPONSE STRINGS ############### ")
    # for n in ref_response_strs:
    #     print(f"type: {type(n)} query: {n}")
    # print()
    
    print(f"################# EVALUATING RESPONSES FROM ORIGINAL RETRIEVER #################")
    eval_results_original = await batch_runner.aevaluate_responses(
        queries=eval_qs[:max_samples],
        responses=pred_responses_original[:max_samples],
        reference=ref_response_strs[:max_samples],
    )

    print(f"\n################# EVALUATING RESPONSES FROM SENTENCE WINDOW RETRIEVER #################")
    eval_results_sentence_window = await batch_runner.aevaluate_responses(
        queries=eval_qs[:max_samples],
        responses=pred_responses_sentence_window[:max_samples],
        reference=ref_response_strs[:max_samples],
    )

    results_df = get_results_df(
        [eval_results_sentence_window, eval_results_original],
        ["Sentence Window Retriever", "Base Retriever"],
        ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
    )
    print(results_df)

    OUTPUT_PATH = '/workspaces/sec-insights/backend/eval/results.csv'
    results_df.to_csv(OUTPUT_PATH, index=False)

    return results_df


async def main():
    
    doc = get_document_via_api(document_id="fbadfd55-17e5-4d67-a6a1-cfe00043c7a0")
    # doc = get_dummy_doc()
    conv_args = {
        "messages": [],
        "documents": [doc]
    }
    conversation = schema.Conversation(**conv_args)
    send_chan, recv_chan = anyio.create_memory_object_stream(100)
    callback_handler = ChatCallbackHandler(send_chan)
    chat_engine, doc_id_to_index = await get_chat_engine(callback_handler, conversation, return_doc_id_to_index=True)

    # response = chat_engine.query("Tell me about the company's finances")
    # response = chat_engine.query("Tell me about the company's management")
    # print(f"################## FINAL RESPONSE ##################\n{response}\n")
    # pprint_sentence_window(response, node=4)

    document = fetch_and_read_document(doc)     # merges document pages into a single document

    original_service_context, original_node_parser = get_tool_service_context(callback_handlers=[callback_handler], node_parser_type="original", return_node_parser=True)
    # original_node_parser = original_service_context.node_parser
    # original_nodes = original_node_parser.get_nodes_from_documents(document)
    
    # original_nodes = get_nodes(doc, original_node_parser)
    original_nodes = original_node_parser.get_nodes_from_documents(document)

    sentence_window_service_context, sentence_window_node_parser = get_tool_service_context(callback_handlers=[callback_handler], node_parser_type="setence-window", return_node_parser=True)
    # nodes_sentence_window = get_nodes(doc, sentence_window_node_parser)
    nodes_sentence_window = sentence_window_node_parser.get_nodes_from_documents(document)

    hierarchical_service_context, hierarchical_node_parser = get_tool_service_context(callback_handlers=[callback_handler], node_parser_type="hierarchical", return_node_parser=True)
    hierarchical_nodes = hierarchical_node_parser.get_nodes_from_documents(document)
    leaf_nodes = get_leaf_nodes(hierarchical_nodes)

    # service_context = get_tool_service_context(callback_handlers=[callback_handler])
    eval_dataset = await generate_dataset(
        file_path="/workspaces/sec-insights/backend/eval/eval_dataset.json",
        nodes=nodes_sentence_window,
        num_nodes_eval=20,
        # service_context=service_context,
        service_context=original_service_context,
    )
    
    # # sentence_window_index = doc_id_to_index[str(doc.id)]
    # sentence_window_index = VectorStoreIndex.from_documents(
    #     document, service_context=service_context
    # )

    # results_df = await evaluate(
    #     original_nodes=original_nodes,
    #     sentence_window_index=sentence_window_index,
    #     eval_dataset=eval_dataset,
    #     max_samples=20,
    # )

    return



if __name__ == "__main__":
    asyncio.run(main())


