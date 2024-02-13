import os
import asyncio
import numpy as np
from eval_utils import get_trulens_recorder
from trulens_eval import (
    Tru,
    Feedback,
    TruLlama
    # OpenAI
)
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

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

def get_trulens_recorder(query_engine, app_id):
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


async def main():
    
    doc = get_document_via_api(document_id="4d24de4e-63ee-4af5-9c97-ccae008ad887")
    conv_args = {
        "messages": [],
        "documents": [doc]
    }
    conversation = schema.Conversation(**conv_args)
    send_chan, recv_chan = anyio.create_memory_object_stream(100)
    chat_engine = await get_chat_engine(ChatCallbackHandler(send_chan), conversation)

    eval_questions = get_eval_questions(file_path)
    Tru().reset_database()
    tru_recorder_1 = get_trulens_recorder(chat_engine, app_id="base_engine_1")
    run_evals(eval_questions, tru_recorder_1, chat_engine)
    
    Tru().run_dashboard()


    return



if __name__ == "__main__":
    asyncio.run(main())
    
    # eval_questions = get_eval_questions(file_path)
    # Tru().reset_database()
    # tru_recorder_1 = get_trulens_recorder







