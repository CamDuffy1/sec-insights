from trulens_eval import (
    Tru,
    Feedback,
    TruLlama,
    OpenAI
)
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.hugs import Huggingface
from app.chat.engine import get_chat_engine
import numpy as np
from llama_index.schema import Document as LlamaIndexDocument
from app import schema
import asyncio
import anyio
from app.chat.messaging import ChatCallbackHandler

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

async def run_trulens_evaluation(document: LlamaIndexDocument, file_path: str = '/workspaces/sec-insights/backend/eval/eval_questions.txt'):
    '''
    Function to evaluate the response of an LLM based on several metrics using TruLens.
    Args:
        document [LlamaIndexDocument]: The document to ask questions about.
        file_path [str]: The path to the file containing questions to use to evaluate the LLM's resonse.
    '''
    conv_args = {
        "messages": [],
        "documents": [document]
    }
    conversation = schema.Conversation(**conv_args)
    send_chan, recv_chan = anyio.create_memory_object_stream(100)
    chat_engine = await get_chat_engine(ChatCallbackHandler(send_chan), conversation)
    
    eval_questions = get_eval_questions(file_path)
    Tru().reset_database()
    # tru_recorder_1 = get_trulens_recorder_openai(chat_engine, app_id="base_engine_1")
    # run_evals(eval_questions, tru_recorder_1, chat_engine)
    
    # Tru().run_dashboard()



