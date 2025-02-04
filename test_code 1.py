from typing import Annotated
import asyncio
from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse

# from src.api.auth import api_key_auth
from src.api.models.bedrock import BedrockModel
from src.api.schema import ChatRequest, ChatResponse, ChatStreamResponse
from src.api.setting import DEFAULT_MODEL

router = APIRouter(
    prefix="/chat"
)

@router.post("/completions", response_model=ChatResponse | ChatStreamResponse, response_model_exclude_unset=True)
async def chat_completions(
        chat_request: Annotated[
            ChatRequest,
            Body(
                examples=[
                    {
                        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"},
                        ],
                    }
                ],
            ),
        ]
):
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)

    params = {
        'input': {
            'text': chat_request.messages[-1].content
        },
        'retrieveAndGenerateConfiguration': {
            'knowledgeBaseConfiguration': {
                'generationConfiguration': {
                    'guardrailConfiguration': {
                        'guardrailId': '6nsnq854bfxa',
                        'guardrailVersion': 'DRAFT'
                    },
                'promptTemplate': {
                    'textPromptTemplate': 'You are a question answering agent. I will provide you with a set of search results. The user will provide you with a question. Your job is to answer the user question using only information from the search results.'
                 }
                },
                'knowledgeBaseId': 'your-knowledge-base-id',
                'modelArn': 'your-model-arn',
                 'orchestrationConfiguration': {
                     'queryTransformationConfiguration': {
                         'type': 'QUERY_DECOMPOSITION'
                     },
                'promptTemplate': {
                    'textPromptTemplate': 'You are a query creation agent. You will be provided with a function and a description of what it searches over. The user will provide you a question, and your job is to determine the optimal query to use based on the user question.'
                }
                  },
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'numberOfResults': 5,
                        'overrideSearchType': 'HYBRID',
                        'rerankingConfiguration': {
                        'bedrockRerankingConfiguration': {
                            'metadataConfiguration': {
                                'selectionMode': 'ALL',
                            },
                            'modelConfiguration': {
                                'modelArn': 'arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0'
                            },
                            'numberOfRerankedResults': 1
                        },
                        'type': 'BEDROCK_RERANKING_MODEL'
                      }
                    }
                }
            },
            'type': 'KNOWLEDGE_BASE'
        },
        'sessionConfiguration':{
        'kmsKeyArn': 'arn:aws:kms:us-west-2:664418992117:key/6e3b3e5f-9fab-40ee-aa0d-2b7bd35a1621'
        }
    }

    if chat_request.stream:
        return StreamingResponse(
            content=model.chat_stream(chat_request, params), media_type="text/event-stream"
        )
    return model.chat(chat_request, params)