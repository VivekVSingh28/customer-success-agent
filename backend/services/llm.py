"""
LLM Service Module for AI Customer Success Agent

This module handles interactions with Azure GPT-4.1 API for conversational AI,
context management, and customer success automation. It supports both
non-streaming and streaming chat completions.

Dependencies:
- requests: For HTTP API calls
- json: For API request/response handling
- os: For environment variable access
- structlog: For structured logging
- time: For retry mechanisms (exponential backoff)
- typing: For type hints
- datetime: For conversation timestamping
- uuid: For generating unique conversation IDs
- tiktoken: For accurate token counting (crucial for context window management)
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Iterator, Union
import requests
import structlog

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import get_config

# Load configuration
config = get_config()

# Import tiktoken for token counting. Ensure it's in requirements.txt.
try:
    import tiktoken
except ImportError:
    print("Warning: 'tiktoken' not found. Token counting will be approximate (character-based). "
          "Install with 'pip install tiktoken' for accurate token management.")
    tiktoken = None # Set to None if not available

# Initialize structured logger for this module
logger = structlog.get_logger(__name__)


class LLMAPIError(Exception):
    """Custom exception for LLM API related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        # Log the error immediately when the exception is created
        logger.error("LLM API Error", message=message, status_code=status_code, response_data=response_data)


class ConversationContext:
    """
    Manages conversation context and history for customer success interactions.
    Handles context windowing, token management, and conversation persistence.
    """
    def __init__(self, max_history_length: int = 20, max_tokens_per_context: int = 4000):
        # Note: max_tokens_per_context here should be less than the LLM's actual context window
        # to leave room for the LLM's response. GPT-4.1 might have 8k, 16k, or 32k context.
        # 4000 is a safe starting point for messages + prompt.
        self.conversation_id = str(uuid.uuid4())
        self.messages: List[Dict[str, Any]] = []
        self.max_history_length = max_history_length
        self.max_tokens_per_context = max_tokens_per_context
        self.customer_info: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Any] = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'interaction_count': 0
        }
        # Initialize tokenizer if tiktoken is available
        # For 'openai/gpt-4.1', 'gpt-4' encoding should be compatible.
        self.tokenizer = tiktoken.encoding_for_model("gpt-4") if tiktoken else None

    def _count_tokens(self, text: str) -> int:
        """
        Estimates token count for a given text. Uses tiktoken if available,
        otherwise provides a character-based approximation.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback for approximate token counting (e.g., 1 token ~ 4 characters)
        return len(text) // 4

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation context and manage history/tokens."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.session_metadata['last_updated'] = datetime.now(timezone.utc).isoformat()
        self.session_metadata['interaction_count'] += 1

        self._manage_context_window()

    def _manage_context_window(self):
        """
        Manages the conversation history to stay within `max_history_length`
        and `max_tokens_per_context`. Prioritizes keeping recent messages.
        """
        # First, trim by history length
        if len(self.messages) > self.max_history_length:
            # Keep system message(s) at the beginning, remove oldest user/assistant messages
            system_messages = [msg for msg in self.messages if msg['role'] == 'system']
            other_messages = [msg for msg in self.messages if msg['role'] != 'system']
            
            # Ensure we don't remove all messages if max_history_length is very small
            num_to_keep = max(0, self.max_history_length - len(system_messages))
            other_messages = other_messages[-num_to_keep:]
            self.messages = system_messages + other_messages
            logger.debug("Trimmed conversation by length", new_length=len(self.messages))

        # Second, trim by token count (more aggressive if needed)
        current_tokens = sum(self._count_tokens(msg['content']) for msg in self.messages)
        while current_tokens > self.max_tokens_per_context and len(self.messages) > 1: # Always keep at least system message
            # Find the oldest non-system message and remove it
            removed = False
            for i in range(len(self.messages)):
                if self.messages[i]['role'] != 'system':
                    removed_message = self.messages.pop(i)
                    current_tokens -= self._count_tokens(removed_message['content'])
                    logger.warning("Trimmed conversation by tokens, removed oldest non-system message",
                                   removed_role=removed_message['role'],
                                   removed_content_preview=removed_message['content'][:50],
                                   current_tokens=current_tokens)
                    removed = True
                    break
            if not removed: # Should not happen if there's more than 1 message and not all are system
                break # All remaining messages are system messages, cannot trim further

    def get_context_for_api(self) -> List[Dict[str, str]]:
        """
        Get formatted messages for API consumption (only 'role' and 'content').
        This also ensures the system message is at the beginning.
        """
        api_messages = []
        system_messages = []
        for msg in self.messages:
            if msg['role'] == 'system':
                system_messages.append({'role': msg['role'], 'content': msg['content']})
            else:
                api_messages.append({'role': msg['role'], 'content': msg['content']})
        
        return system_messages + api_messages

    def set_customer_info(self, customer_data: Dict[str, Any]):
        """Set customer information for personalized responses"""
        self.customer_info.update(customer_data)
        # Optionally, update the system prompt with customer info if it's dynamic
        # This would require re-adding the system message to `self.messages`
        # For simplicity, we assume system prompt is set once at conversation creation.

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            'conversation_id': self.conversation_id,
            'message_count': len(self.messages),
            'customer_info': self.customer_info,
            'session_metadata': self.session_metadata
        }


class AzureGPTClient:
    """
    Client for interacting with Azure GPT-4.1 API using GitHub PAT authentication.
    Handles conversation management, streaming responses, and customer success workflows.
    """

    def __init__(self):
        """
        Initialize the Azure GPT client.
        Loads GitHub PAT from environment variables and other settings from config.
        """
        # Load sensitive API key from environment variable
        self.github_token = os.getenv('GITHUB_TOKEN')
        # Add logic to strip 'Bearer ' prefix if present in the token
        if self.github_token and self.github_token.lower().startswith("bearer "):
            self.github_token = self.github_token[len("bearer "):]
            logger.info("Stripped 'Bearer ' prefix from GITHUB_TOKEN.")

        if not self.github_token:
            logger.critical("GITHUB_TOKEN environment variable not set or empty. Cannot initialize Azure GPT client.")
            raise LLMAPIError("GitHub token is required. Set GITHUB_TOKEN environment variable in .env file.")

        # Load configuration from config.py
        self.azure_gpt_endpoint = config['AZURE_GPT_ENDPOINT']
        self.model_name = config['AZURE_GPT_MODEL']

        # Default headers for API requests.
        # For `https://models.github.ai/inference`, use Authorization Bearer token format.
        self.headers = {
            'Authorization': f'Bearer {self.github_token}',
            'Content-Type': 'application/json'
        }

        # Default conversation configuration from config.py
        self.default_config = {
            'max_tokens': config['LLM_MAX_TOKENS'],
            'temperature': config['LLM_TEMPERATURE'],
            'top_p': config['LLM_TOP_P'],
            'frequency_penalty': config['LLM_FREQUENCY_PENALTY'],
            'presence_penalty': config['LLM_PRESENCE_PENALTY'],
            'stop_sequences': config['LLM_STOP_SEQUENCES'].split(',') if config['LLM_STOP_SEQUENCES'] else None
        }

        # Request timeout and retry configuration from config.py
        self.timeout = config['LLM_REQUEST_TIMEOUT']
        self.max_retries = config['LLM_MAX_RETRIES']
        self.retry_delay_base = config['LLM_RETRY_DELAY_BASE']

        # Customer success system prompt (loaded from .env or default)
        self.system_prompt = self._load_system_prompt()

        # Active conversations storage (in production, consider Redis or a database)
        self.conversations: Dict[str, ConversationContext] = {}

        logger.info("Azure GPT client initialized",
                    endpoint=self.azure_gpt_endpoint,
                    model=self.model_name,
                    max_tokens_response=self.default_config['max_tokens'])

    def _load_system_prompt(self) -> str:
        """Load the system prompt for customer success agent from environment or default."""
        # As per your request, no custom LLM_SYSTEM_PROMPT from .env.
        # This will always use the hardcoded default prompt.
        default_prompt = """You are an AI Customer Success Agent designed to help customers with their inquiries, resolve issues, and ensure customer satisfaction.

Your key responsibilities:
- Provide helpful, accurate, and empathetic responses.
- Escalate complex issues to human agents when necessary.
- Maintain conversation context and customer history.
- Offer proactive solutions and recommendations.
- Ensure customer satisfaction and retention.

Guidelines:
- Be professional, friendly, and solution-oriented.
- Ask clarifying questions when needed.
- Provide step-by-step instructions for complex processes.
- Acknowledge customer frustrations and show empathy.
- Always aim to resolve issues in the first interaction when possible.
- If you cannot resolve an issue, clearly explain next steps.

Remember: Customer satisfaction is the primary goal. Be helpful, understanding, and thorough in your responses."""

        return default_prompt # Directly return the default prompt

    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None,
                      params: Optional[Dict] = None, stream: bool = False) -> requests.Response:
        """
        Make HTTP request to Azure GPT API with retry logic and comprehensive error handling.

        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint path (e.g., 'chat/completions'). Note: self.azure_gpt_endpoint
                            should already contain the full base URL including deployment and api-version.
            json_data (dict, optional): Request payload for JSON body.
            params (dict, optional): Query parameters.
            stream (bool): Whether to stream the response (for LLM streaming).

        Returns:
            requests.Response: API response object.

        Raises:
            LLMAPIError: If the request fails after all retries or due to an API-specific error.
        """
        # The self.azure_gpt_endpoint is the base URL: https://models.github.ai/inference
        # For chat completions, we need to append the correct path
        if endpoint:
            url = f"{self.azure_gpt_endpoint}/{endpoint}"
        else:
            url = self.azure_gpt_endpoint

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Making Azure GPT API request",
                             method=method,
                             url=url,
                             attempt=attempt + 1,
                             max_retries=self.max_retries,
                             stream=stream,
                             json_data_present=bool(json_data),
                             params_present=bool(params))

                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=json_data,
                    params=params,
                    timeout=self.timeout,
                    stream=stream
                )

                response.raise_for_status()
                logger.debug("Azure GPT API request successful", status_code=response.status_code)
                return response

            except requests.exceptions.HTTPError as e:
                error_data = None
                try:
                    error_data = e.response.json()
                except (json.JSONDecodeError, AttributeError):
                    error_data = {'raw_response': getattr(e.response, 'text', str(e))}

                logger.error("Azure GPT API HTTP error",
                             status_code=e.response.status_code,
                             error_details=error_data,
                             url=url,
                             attempt=attempt + 1)

                # For 4xx errors, don't retry
                if 400 <= e.response.status_code < 500:
                    error_message = error_data.get('error', {}).get('message', 'Unknown client error')
                    raise LLMAPIError(
                        f"Azure GPT API client error: {e.response.status_code} - {error_message}",
                        status_code=e.response.status_code,
                        response_data=error_data
                    )

                # For 5xx errors, retry if attempts remain
                elif attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Azure GPT API server error, retrying...",
                                   error=str(e),
                                   status_code=e.response.status_code,
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    raise LLMAPIError(
                        f"Azure GPT API server error after {self.max_retries} retries: {str(e)}",
                        status_code=e.response.status_code,
                        response_data=error_data
                    )

            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Azure GPT API connection error, retrying...",
                                   error=str(e),
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    raise LLMAPIError(f"Azure GPT API connection failed after {self.max_retries} retries: {str(e)}")

            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Azure GPT API request timed out, retrying...",
                                   error=str(e),
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    raise LLMAPIError(f"Azure GPT API request timed out after {self.max_retries} retries: {str(e)}")

            except json.JSONDecodeError as e:
                # This can happen if response.raise_for_status() passes, but content is not JSON
                logger.error("Failed to decode JSON response from Azure GPT API", error=str(e), raw_response=response.text)
                raise LLMAPIError(f"Invalid JSON response from Azure GPT API: {str(e)}")

            except Exception as e:
                logger.critical("An unhandled error occurred in _make_request for Azure GPT API", error=str(e))
                raise LLMAPIError(f"An unhandled error occurred during Azure GPT API request: {str(e)}")

    def create_conversation(self, customer_info: Optional[Dict] = None,
                            max_history_length: Optional[int] = None,
                            max_tokens_per_context: Optional[int] = None) -> str:
        """
        Create a new conversation context.

        Args:
            customer_info (dict, optional): Customer information for personalization.
            max_history_length (int, optional): Max number of messages to keep in history.
            max_tokens_per_context (int, optional): Max tokens for the context window.

        Returns:
            str: Conversation ID.
        """
        context = ConversationContext(
            max_history_length=max_history_length if max_history_length is not None else config['LLM_MAX_HISTORY_LENGTH'],
            max_tokens_per_context=max_tokens_per_context if max_tokens_per_context is not None else config['LLM_MAX_TOKENS_PER_CONTEXT']
        )

        if customer_info:
            context.set_customer_info(customer_info)

        # Add system message with customer context
        system_message_content = self.system_prompt
        if customer_info:
            system_message_content += f"\n\nCustomer Context: {json.dumps(customer_info, indent=2)}"

        context.add_message('system', system_message_content)

        self.conversations[context.conversation_id] = context

        logger.info("New conversation created",
                    conversation_id=context.conversation_id,
                    has_customer_info=bool(customer_info),
                    initial_context_tokens=context._count_tokens(system_message_content))

        return context.conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context by ID."""
        return self.conversations.get(conversation_id)

    def chat_completion(self, message: str, conversation_id: Optional[str] = None,
                        customer_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion response (non-streaming).

        Args:
            message (str): User message.
            conversation_id (str, optional): Existing conversation ID. If None, a new conversation is created.
            customer_info (dict, optional): Customer information for new conversations.
            **kwargs: Additional generation parameters (e.g., max_tokens, temperature).

        Returns:
            dict: Response with generated text and metadata.
        """
        # Get or create conversation context
        if conversation_id and conversation_id in self.conversations:
            context = self.conversations[conversation_id]
        else:
            conversation_id = self.create_conversation(customer_info)
            context = self.conversations[conversation_id]

        # Add user message to context. Context manager will handle trimming.
        context.add_message('user', message)

        # Prepare request payload
        config = self.default_config.copy()
        config.update(kwargs)

        request_data = {
            'model': self.model_name,
            'messages': context.get_context_for_api(), # Get messages after potential trimming
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'frequency_penalty': config['frequency_penalty'],
            'presence_penalty': config['presence_penalty'],
            'stream': False
        }

        if config['stop_sequences']:
            request_data['stop'] = config['stop_sequences']

        logger.info("Generating chat completion (non-streaming)",
                    conversation_id=conversation_id,
                    message_length=len(message),
                    context_messages_count=len(context.messages),
                    context_tokens_estimate=sum(context._count_tokens(m['content']) for m in context.messages))

        try:
            response = self._make_request(
                method='POST',
                endpoint='chat/completions', # Standard OpenAI API path
                json_data=request_data
            )

            result = response.json()

            if 'choices' not in result or not result['choices']:
                logger.error("No choices returned from Azure GPT API", response_data=result)
                raise LLMAPIError("No choices returned from Azure GPT API")

            assistant_message = result['choices'][0]['message']['content']

            # Add assistant response to context
            context.add_message('assistant', assistant_message)

            # Prepare response
            response_data = {
                'response': assistant_message,
                'conversation_id': conversation_id,
                'usage': result.get('usage', {}),
                'model': result.get('model', self.model_name),
                'finish_reason': result['choices'][0].get('finish_reason', 'stop'),
                'conversation_summary': context.get_conversation_summary()
            }

            logger.info("Chat completion generated successfully",
                        conversation_id=conversation_id,
                        response_length=len(assistant_message),
                        tokens_used=result.get('usage', {}).get('total_tokens', 0))

            return response_data

        except LLMAPIError:
            raise
        except Exception as e:
            logger.error("Unexpected error during non-streaming chat completion", error=str(e))
            raise LLMAPIError(f"Chat completion failed: {str(e)}")

    def chat_completion_streaming(self, message: str, conversation_id: Optional[str] = None,
                                  customer_info: Optional[Dict] = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Generate a streaming chat completion response.

        Args:
            message (str): User message.
            conversation_id (str, optional): Existing conversation ID. If None, a new conversation is created.
            customer_info (dict, optional): Customer information for new conversations.
            **kwargs: Additional generation parameters (e.g., max_tokens, temperature).

        Yields:
            dict: Response chunks with partial content.
        """
        # Get or create conversation context
        if conversation_id and conversation_id in self.conversations:
            context = self.conversations[conversation_id]
        else:
            conversation_id = self.create_conversation(customer_info)
            context = self.conversations[conversation_id]

        # Add user message to context. Context manager will handle trimming.
        context.add_message('user', message)

        # Prepare request payload
        config = self.default_config.copy()
        config.update(kwargs)

        request_data = {
            'model': self.model_name,
            'messages': context.get_context_for_api(), # Get messages after potential trimming
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'frequency_penalty': config['frequency_penalty'],
            'presence_penalty': config['presence_penalty'],
            'stream': True # Request streaming response from LLM
        }

        if config['stop_sequences']:
            request_data['stop'] = config['stop_sequences']

        logger.info("Starting streaming chat completion",
                    conversation_id=conversation_id,
                    message_length=len(message),
                    context_messages_count=len(context.messages),
                    context_tokens_estimate=sum(context._count_tokens(m['content']) for m in context.messages))

        full_assistant_message = "" # To reconstruct the full message for context
        try:
            response = self._make_request(
                method='POST',
                endpoint='chat/completions', # Standard OpenAI API path
                json_data=request_data,
                stream=True
            )

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix

                        if data_str.strip() == '[DONE]':
                            break

                        try:
                            chunk_data = json.loads(data_str)

                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]

                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    full_assistant_message += content

                                    yield {
                                        'content': content,
                                        'conversation_id': conversation_id,
                                        'finish_reason': choice.get('finish_reason'),
                                        'usage': chunk_data.get('usage') # Usage might be in final chunk only
                                    }
                                elif 'delta' in choice and 'role' in choice['delta']:
                                    # This is typically the first chunk, indicating the role
                                    pass
                                elif 'finish_reason' in choice and choice['finish_reason']:
                                    # This is the last chunk, containing finish_reason and potentially usage
                                    yield {
                                        'content': '', # No content in this chunk
                                        'conversation_id': conversation_id,
                                        'finish_reason': choice['finish_reason'],
                                        'usage': chunk_data.get('usage')
                                    }
                            else:
                                logger.warning("Streaming chunk missing 'choices' or empty", chunk_data=chunk_data)

                        except json.JSONDecodeError as e:
                            logger.error("Failed to decode JSON from streaming chunk", error=str(e), raw_chunk=line_str)
                            # Continue processing other chunks, but log the error
                            continue
                        except Exception as e:
                            logger.error("Error processing streaming chunk", error=str(e), chunk_data=chunk_data)
                            continue

            # Add complete assistant response to context after streaming finishes
            if full_assistant_message:
                context.add_message('assistant', full_assistant_message)

            logger.info("Streaming chat completion completed",
                        conversation_id=conversation_id,
                        response_length=len(full_assistant_message))

        except LLMAPIError:
            raise # Re-raise custom API errors
        except Exception as e:
            logger.error("Unexpected error during streaming chat completion", error=str(e))
            raise LLMAPIError(f"Streaming chat completion failed: {str(e)}")

    def analyze_customer_sentiment(self, conversation_id: str) -> Dict[str, Any]:
        """
        Analyze customer sentiment from conversation history using the LLM.

        Args:
            conversation_id (str): Conversation ID to analyze.

        Returns:
            dict: Sentiment analysis results.

        Raises:
            LLMAPIError: If conversation not found or analysis fails.
        """
        if conversation_id not in self.conversations:
            raise LLMAPIError(f"Conversation {conversation_id} not found for sentiment analysis.")

        context = self.conversations[conversation_id]

        # Get customer messages only
        customer_messages = [msg['content'] for msg in context.messages if msg['role'] == 'user']

        if not customer_messages:
            logger.info("No customer messages to analyze for sentiment.", conversation_id=conversation_id)
            return {'sentiment': 'neutral', 'confidence': 0.0, 'analysis': 'No customer messages to analyze', 'recommendations': []}

        # Create sentiment analysis prompt
        analysis_prompt = f"""Analyze the sentiment of the following customer messages and provide:
1. Overall sentiment (positive, negative, neutral, mixed)
2. Confidence score (0.0 to 1.0)
3. Brief analysis of key emotional indicators
4. Recommendations for customer success actions based on sentiment

Customer Messages:
{json.dumps(customer_messages, indent=2)}

Respond ONLY in JSON format with keys: "sentiment", "confidence", "analysis", "recommendations".
Example: {{"sentiment": "positive", "confidence": 0.9, "analysis": "Customer expressed satisfaction.", "recommendations": ["Offer loyalty discount"]}}
"""
        logger.info("Sending sentiment analysis request to LLM", conversation_id=conversation_id)
        try:
            # Create a temporary conversation context to avoid polluting main history with analysis prompt
            temp_context_id = self.create_conversation(
                max_history_length=2, # Only need system + user prompt for this
                max_tokens_per_context=1000 # Smaller context for analysis
            )
            temp_context = self.conversations[temp_context_id]
            # Override system prompt for this specific analysis if desired, or rely on default
            temp_context.messages[0]['content'] = "You are an expert sentiment analysis AI. Your task is to analyze customer messages and provide sentiment in a structured JSON format."


            result = self.chat_completion(
                message=analysis_prompt,
                conversation_id=temp_context_id,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=500 # Limit response length for analysis
            )

            # Clean up temporary conversation
            del self.conversations[temp_context_id]

            # Parse JSON response
            try:
                sentiment_data = json.loads(result['response'])
                # Basic validation of expected keys
                if not all(k in sentiment_data for k in ['sentiment', 'confidence', 'analysis', 'recommendations']):
                    raise ValueError("Missing expected keys in sentiment analysis JSON.")
                return sentiment_data
            except json.JSONDecodeError as e:
                logger.error("LLM sentiment analysis returned invalid JSON", error=str(e), raw_response=result['response'])
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'analysis': f"LLM returned non-JSON response for sentiment: {result['response']}",
                    'recommendations': []
                }
            except ValueError as e:
                 logger.error("LLM sentiment analysis JSON missing expected keys", error=str(e), parsed_data=sentiment_data)
                 return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'analysis': f"LLM sentiment analysis JSON missing keys: {str(e)}. Raw: {result['response']}",
                    'recommendations': []
                }

        except LLMAPIError:
            raise
        except Exception as e:
            logger.error("Error analyzing customer sentiment", error=str(e), conversation_id=conversation_id)
            raise LLMAPIError(f"Failed to analyze sentiment: {str(e)}")

    def generate_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Generate a summary of the conversation for handoff or reporting using the LLM.

        Args:
            conversation_id (str): Conversation ID to summarize.

        Returns:
            dict: Conversation summary.

        Raises:
            LLMAPIError: If conversation not found or summary generation fails.
        """
        if conversation_id not in self.conversations:
            raise LLMAPIError(f"Conversation {conversation_id} not found for summary generation.")

        context = self.conversations[conversation_id]

        # Create summary prompt
        summary_prompt = f"""Please provide a comprehensive summary of this customer service conversation including:
1. Customer's main issue or request
2. Key points discussed
3. Resolution status (e.g., resolved, pending, escalated)
4. Next steps or follow-up required
5. Overall customer satisfaction level (e.g., high, medium, low)

Conversation:
{json.dumps(context.get_context_for_api(), indent=2)}

Provide a professional summary suitable for handoff to human agents."""
        logger.info("Sending summary generation request to LLM", conversation_id=conversation_id)
        try:
            # Create a temporary conversation context for summary generation
            temp_context_id = self.create_conversation(
                max_history_length=len(context.messages) + 2, # Ensure full conversation fits + prompt
                max_tokens_per_context=context.max_tokens_per_context # Use same token limit
            )
            temp_context = self.conversations[temp_context_id]
            # Add the full conversation history to the temporary context
            for msg in context.messages:
                temp_context.add_message(msg['role'], msg['content'], msg.get('metadata'))
            # Add the summary prompt
            temp_context.add_message('user', summary_prompt)

            result = self.chat_completion(
                message=summary_prompt, # The actual prompt is added to temp_context, this is just for the call
                conversation_id=temp_context_id,
                temperature=0.3,
                max_tokens=700 # Limit response length for summary
            )

            # Clean up temporary conversation
            del self.conversations[temp_context_id]

            return {
                'summary': result['response'],
                'conversation_stats': context.get_conversation_summary(),
                'generated_at': datetime.now(timezone.utc).isoformat()
            }

        except LLMAPIError:
            raise
        except Exception as e:
            logger.error("Error generating conversation summary", error=str(e), conversation_id=conversation_id)
            raise LLMAPIError(f"Failed to generate summary: {str(e)}")

    def cleanup_conversation(self, conversation_id: str) -> bool:
        """
        Clean up conversation from memory.

        Args:
            conversation_id (str): Conversation ID to clean up.

        Returns:
            bool: True if successfully cleaned up, False otherwise.
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info("Conversation cleaned up", conversation_id=conversation_id)
            return True
        logger.warning("Attempted to clean up non-existent conversation", conversation_id=conversation_id)
        return False

    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        return list(self.conversations.keys())

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate LLM configuration parameters.

        Args:
            config (dict): Configuration to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        logger.debug("Validating LLM configuration", config=config)
        try:
            # Validate temperature
            temperature = config.get('temperature')
            if temperature is not None and not (0.0 <= temperature <= 2.0):
                logger.warning("Invalid temperature value", temperature=temperature)
                return False

            # Validate top_p
            top_p = config.get('top_p')
            if top_p is not None and not (0.0 <= top_p <= 1.0):
                logger.warning("Invalid top_p value", top_p=top_p)
                return False

            # Validate max_tokens (for response generation)
            max_tokens = config.get('max_tokens')
            if max_tokens is not None and not (1 <= max_tokens <= 4096): # Common max for GPT-4 responses
                logger.warning("Invalid max_tokens value", max_tokens=max_tokens)
                return False

            # Validate penalty values
            for penalty in ['frequency_penalty', 'presence_penalty']:
                value = config.get(penalty)
                if value is not None and not (-2.0 <= value <= 2.0):
                    logger.warning(f"Invalid {penalty} value", value=value)
                    return False

            return True

        except Exception as e:
            logger.error("Configuration validation failed due to unexpected error", error=str(e), config=config)
            return False


# Convenience function for module-level usage
def create_llm_service() -> AzureGPTClient:
    """
    Create and return a configured Azure GPT client.
    This function strictly loads configuration from environment variables.

    Returns:
        AzureGPTClient: Configured client instance.
    """
    return AzureGPTClient()


# Example usage and testing
if __name__ == '__main__':
    # To run this example, ensure you have a .env file in the project root with:
    # GITHUB_TOKEN="your_github_pat_azure_gpt_4_1_here"
    # AZURE_GPT_ENDPOINT="https://models.github.ai/inference"
    # AZURE_GPT_MODEL="openai/gpt-4.1"
    # LLM_MAX_TOKENS="1000"
    # LLM_TEMPERATURE="0.7"
    # LLM_TOP_P="0.95"
    # LLM_FREQUENCY_PENALTY="0.0"
    # LLM_PRESENCE_PENALTY="0.0"
    # LLM_REQUEST_TIMEOUT="60"
    # LLM_MAX_RETRIES="3"
    # LLM_RETRY_DELAY_BASE="2"
    # LLM_MAX_HISTORY_LENGTH="20"
    # LLM_MAX_TOKENS_PER_CONTEXT="4000" # Adjust based on your LLM's context window and response size

    from dotenv import load_dotenv
    import sys

    # Load environment variables
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

    # Configure structlog for example
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    try:
        llm_client = create_llm_service()

        print("\n--- Testing Basic Chat Completion ---")
        customer_info = {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'plan': 'Premium',
            'issue_type': 'Technical Support'
        }

        # Create conversation
        conversation_id = llm_client.create_conversation(customer_info=customer_info)
        print(f"Created conversation: {conversation_id}")

        # Test chat completion
        response = llm_client.chat_completion(
            message="Hi, I'm having trouble with my account. Can you help me?",
            conversation_id=conversation_id
        )

        print(f"Response: {response['response'][:200]}...")
        print(f"Tokens used: {response['usage'].get('total_tokens', 'N/A')}")
        print(f"Current conversation messages: {len(llm_client.get_conversation(conversation_id).messages)}")

        print("\n--- Testing Streaming Chat Completion ---")
        print("Streaming response:")
        full_response = ""
        for chunk in llm_client.chat_completion_streaming(
            message="Can you explain the steps to reset my password?",
            conversation_id=conversation_id
        ):
            if chunk.get('content'):
                print(chunk['content'], end='', flush=True)
                full_response += chunk['content']
            # Optional: print finish reason or usage from the last chunk
            if chunk.get('finish_reason'):
                print(f"\nFinish Reason: {chunk['finish_reason']}")
            if chunk.get('usage'):
                print(f"Usage from last chunk: {chunk['usage']}")

        print(f"\n\nFull streaming response length: {len(full_response)}")
        print(f"Current conversation messages: {len(llm_client.get_conversation(conversation_id).messages)}")

        print("\n--- Testing Sentiment Analysis ---")
        sentiment = llm_client.analyze_customer_sentiment(conversation_id)
        print(f"Sentiment: {sentiment.get('sentiment', 'N/A')}")
        print(f"Confidence: {sentiment.get('confidence', 'N/A')}")
        print(f"Analysis: {sentiment.get('analysis', 'N/A')}")
        print(f"Recommendations: {sentiment.get('recommendations', 'N/A')}")

        print("\n--- Testing Conversation Summary ---")
        summary = llm_client.generate_summary(conversation_id)
        print(f"Summary: {summary['summary'][:200]}...")

        print("\n--- Active Conversations ---")
        active = llm_client.get_active_conversations()
        print(f"Active conversations: {len(active)}")

        # Cleanup
        llm_client.cleanup_conversation(conversation_id)
        print(f"Cleaned up conversation: {conversation_id}")
        active_after_cleanup = llm_client.get_active_conversations()
        print(f"Active conversations after cleanup: {len(active_after_cleanup)}")

    except LLMAPIError as e:
        print(f"LLM API Error: {e.message} (Status: {e.status_code})")
    except Exception as e:
        print(f"Unexpected error: {e}")

