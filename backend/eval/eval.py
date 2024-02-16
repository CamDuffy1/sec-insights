import os
import asyncio
import numpy as np
# from trulens_eval import (
#     Tru,
#     Feedback,
#     TruLlama,
#     OpenAI
# )
# from trulens_eval.feedback import Groundedness
# from trulens_eval.feedback.provider.hugs import Huggingface

from app.chat.engine import get_chat_engine
from app.chat.messaging import (
    StreamedMessage,
    StreamedMessageSubProcess,
    ChatCallbackHandler,
    handle_chat_message
)
from app.api.endpoints.conversation import (
    create_conversation,
    get_conversation
)
from app import schema
import requests
from app.api.deps import get_db
from llama_index.callbacks.base import BaseCallbackHandler, CallbackManager
import anyio
import requests
from llama_index import SimpleDirectoryReader
from llama_index.readers.file.docs_reader import PDFReader
from tempfile import TemporaryDirectory
from pathlib import Path
from app.chat.constants import DB_DOC_ID_KEY
from llama_index.schema import Document as LlamaIndexDocument


# absolute path of text file containing questions for evaluation
file_path = '/workspaces/sec-insights/backend/eval/eval_questions.txt'

def get_eval_questions(file_path):
    eval_questions = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            eval_questions.append(item)
    return eval_questions

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)

def get_trulens_recorder_openai(query_engine, app_id):
    openai = OpenAI()

    qa_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    grounded = Groundedness(groundedness_provider=openai)

    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_trulens_recorder_huggingface(query_engine, app_id):
    huggingface = Huggingface()
    feedback = Feedback(huggingface.language_match).on_input_output()
    feedbacks = [feedback]
    
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

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





async def main():
    
    doc = get_document_via_api(document_id="fbadfd55-17e5-4d67-a6a1-cfe00043c7a0")
    # doc = get_dummy_doc()
    conv_args = {
        "messages": [],
        "documents": [doc]
    }
    conversation = schema.Conversation(**conv_args)
    send_chan, recv_chan = anyio.create_memory_object_stream(100)
    chat_engine = await get_chat_engine(ChatCallbackHandler(send_chan), conversation)
    
    # response = chat_engine.query("Tell me about the company's finances")
    response = chat_engine.query("Tell me about the company's management")
    print(response)

    # eval_questions = get_eval_questions(file_path)
    # for question in eval_questions[:2]:
    #     response = chat_engine.query(question)
    #     print(response)

    # Tru().reset_database()
    # # tru_recorder_1 = get_trulens_recorder_openai(chat_engine, app_id="base_engine_1")
    # tru_recorder_1 = get_trulens_recorder_huggingface(chat_engine, app_id="base_engine_1")
    # run_evals(eval_questions, tru_recorder_1, chat_engine)
    
    # Tru().run_dashboard()



    return



if __name__ == "__main__":

    asyncio.run(main())

    # doc = get_document_via_api(document_id="fbadfd55-17e5-4d67-a6a1-cfe00043c7a0")
    # print(doc)


    # print(doc.metadata_map['sec_document']['company_ticker'])
    # print(type(doc.metadata_map['sec_document']['company_ticker']))
    # print(f"MERGING DOCUMENT: {doc.metadata_map['sec_document']['company_ticker']} {doc.metadata_map['sec_document']['doc_type']} {doc.metadata_map['sec_document']['year']}")
    # response = requests.get(doc.url)
    # with TemporaryDirectory() as temp_dir:
    #     temp_file_path = Path(temp_dir) / f"{str(doc.id)}.pdf"
    #     with open(temp_file_path, "wb") as temp_file:
    #         temp_file.write(response.content)
    
    #     documents = SimpleDirectoryReader(
    #         input_files=[temp_file_path],
    #     ).load_data()
            
        # reader = PDFReader()
        # documents = reader.load_data(
        #     temp_file_path, extra_info={DB_DOC_ID_KEY: str(doc.id)}
        # )

    # merged_document = LlamaIndexDocument(text="\n\n".join([doc.text for doc in documents]), extra_info={DB_DOC_ID_KEY: str(doc.id)})

    # print("\n\nAfter document merging:")
    # print(type(merged_document), "\n")
    # print(merged_document.extra_info)

    



    








