import asyncio
from app.chat.engine import get_chat_engine
from app.chat.messaging import ChatCallbackHandler
from app import schema
import requests
import anyio
# from app.api.deps import get_db
# from llama_index.callbacks.base import BaseCallbackHandler, CallbackManager

from llama_index.schema import Document as LlamaIndexDocument
from llama_index.response.schema import Response

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
        response: llama_index.response.schema.Response object
        node [int]: The node to print sentence-window vs original sentence for. Use when only want to print for a single node.
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
    
    response = chat_engine.query("Tell me about the company's finances")
    # response = chat_engine.query("Tell me about the company's management")
    print(f"################## FINAL RESPONSE ##################\n{response}\n")
    # pprint_sentence_window(response, node=4)



    return



if __name__ == "__main__":

    asyncio.run(main())




