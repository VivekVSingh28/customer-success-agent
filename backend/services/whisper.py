"""
OpenAI Whisper STT Service Module

This module handles speech-to-text conversion using OpenAI's Whisper API
via the official OpenAI API endpoint, supporting the customer success agent's
voice input capabilities.

Dependencies:
- requests: For HTTP API calls
- json: For API request/response handling
- os: For environment variable access
- structlog: For structured logging
- time: For retry mechanisms (exponential backoff)
- typing: For type hints
- tempfile: For temporary file handling
- io: For byte stream handling
- base64: For audio data encoding
"""

import os
import json
import time
import tempfile
import base64
from typing import Optional, Dict, Any, Union
from io import BytesIO
import requests
import structlog

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import get_config

# Load configuration
config = get_config()

# Initialize structured logger for this module
logger = structlog.get_logger(__name__)


class WhisperAPIError(Exception):
    """Custom exception for Whisper API related errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        # Log the error immediately when the exception is created
        logger.error("Whisper API Error", message=message, status_code=status_code, response_data=response_data)


class WhisperSTTClient:
    """
    Client for OpenAI Whisper Speech-to-Text API using OpenAI API key authentication.
    Handles audio file transcription with support for multiple audio formats.
    """

    def __init__(self):
        """
        Initialize the Whisper STT client.
        Loads OpenAI API key from environment variables and other settings from config.
        """
        # Load sensitive API key from environment variable
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.critical("OPENAI_API_KEY environment variable not set or empty. Cannot initialize Whisper STT client.")
            raise WhisperAPIError("OpenAI API key is required. Set OPENAI_API_KEY environment variable in .env file.")

        # Load configuration from config.py
        self.whisper_endpoint = config['WHISPER_ENDPOINT']
        self.model_name = config['WHISPER_MODEL']

        # Default headers for API requests
        self.headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            # Note: Content-Type will be set automatically for multipart/form-data
        }

        # Default transcription configuration from config.py
        self.default_config = {
            'language': config['WHISPER_LANGUAGE'],
            'response_format': config['WHISPER_RESPONSE_FORMAT'],
            'temperature': config['WHISPER_TEMPERATURE']
        }

        # Request timeout and retry configuration from config.py
        self.timeout = config['WHISPER_REQUEST_TIMEOUT']
        self.max_retries = config['WHISPER_MAX_RETRIES']
        self.retry_delay_base = config['WHISPER_RETRY_DELAY_BASE']

        # File size and format limits
        self.max_file_size = config['WHISPER_MAX_FILE_SIZE']
        self.supported_formats = config['WHISPER_SUPPORTED_FORMATS']

        logger.info("Whisper STT client initialized",
                    endpoint=self.whisper_endpoint,
                    model=self.model_name,
                    language=self.default_config['language'],
                    supported_formats=self.supported_formats)

    def _validate_audio_file(self, audio_data: bytes, audio_format: str) -> bool:
        """
        Validate audio file size and format.

        Args:
            audio_data (bytes): Raw audio data
            audio_format (str): Audio format (e.g., 'wav', 'mp3')

        Returns:
            bool: True if valid, False otherwise

        Raises:
            WhisperAPIError: If validation fails
        """
        # Check file size
        if len(audio_data) > self.max_file_size:
            raise WhisperAPIError(
                f"Audio file size ({len(audio_data)} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)",
                status_code=413
            )

        # Check format
        if audio_format.lower() not in self.supported_formats:
            raise WhisperAPIError(
                f"Audio format '{audio_format}' not supported. Supported formats: {', '.join(self.supported_formats)}",
                status_code=400
            )

        return True

    def _make_request(self, audio_data: bytes, audio_format: str, **kwargs) -> requests.Response:
        """
        Make HTTP request to Whisper API with retry logic and comprehensive error handling.

        Args:
            audio_data (bytes): Raw audio data
            audio_format (str): Audio format
            **kwargs: Additional parameters for the API request

        Returns:
            requests.Response: API response object

        Raises:
            WhisperAPIError: If the request fails after all retries
        """
        # Validate audio file first
        self._validate_audio_file(audio_data, audio_format)

        # Prepare the request - use OpenAI's direct endpoint
        url = self.whisper_endpoint
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Making Whisper API request",
                             attempt=attempt + 1,
                             max_retries=self.max_retries,
                             audio_size=len(audio_data),
                             audio_format=audio_format)

                # Create temporary file for audio data
                with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name

                try:
                    # Prepare multipart form data for OpenAI API
                    files = {
                        'file': (f'audio.{audio_format}', open(temp_file_path, 'rb'), f'audio/{audio_format}')
                    }
                    
                    data = {
                        'model': self.model_name,
                        'language': kwargs.get('language', self.default_config['language']),
                        'response_format': kwargs.get('response_format', self.default_config['response_format']),
                        'temperature': kwargs.get('temperature', self.default_config['temperature'])
                    }

                    # Make the request with multipart form data
                    response = requests.post(
                        url,
                        headers={'Authorization': self.headers['Authorization']},  # Don't set Content-Type for multipart
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )

                    # Close the file
                    files['file'][1].close()

                    # Raise an HTTPError for bad responses (4xx or 5xx)
                    response.raise_for_status()

                    logger.debug("Whisper API request successful", status_code=response.status_code)
                    return response

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            except requests.exceptions.HTTPError as e:
                error_data = None
                try:
                    error_data = e.response.json()
                except json.JSONDecodeError:
                    error_data = {'raw_response': e.response.text}
                
                logger.error("Whisper API HTTP error",
                             status_code=e.response.status_code,
                             error_details=error_data,
                             url=url,
                             attempt=attempt + 1)
                
                # For 4xx errors (client errors), don't retry, raise immediately
                if 400 <= e.response.status_code < 500:
                    raise WhisperAPIError(
                        f"Whisper API client error: {e.response.status_code} - {error_data.get('error', {}).get('message', 'Unknown client error')}",
                        status_code=e.response.status_code,
                        response_data=error_data
                    )
                # For 5xx errors (server errors), retry if attempts remain
                elif attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Whisper API server error, retrying...",
                                   error=str(e),
                                   status_code=e.response.status_code,
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Whisper API server error failed after all retries", error=str(e))
                    raise WhisperAPIError(f"Whisper API server error after {self.max_retries} retries: {str(e)}",
                                         status_code=e.response.status_code,
                                         response_data=error_data)

            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Whisper API connection error, retrying...",
                                   error=str(e),
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Whisper API connection failed after all retries", error=str(e))
                    raise WhisperAPIError(f"Whisper API connection failed after {self.max_retries} retries: {str(e)}")

            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Whisper API request timed out, retrying...",
                                   error=str(e),
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Whisper API request timed out after all retries", error=str(e))
                    raise WhisperAPIError(f"Whisper API request timed out after {self.max_retries} retries: {str(e)}")

            except requests.exceptions.RequestException as e:
                logger.error("An unexpected requests error occurred with Whisper API", error=str(e))
                raise WhisperAPIError(f"An unexpected API request error occurred: {str(e)}")

            except Exception as e:
                logger.critical("An unhandled error occurred in Whisper API request", error=str(e))
                raise WhisperAPIError(f"An unhandled error occurred during Whisper API request: {str(e)}")

    def transcribe_audio(self, audio_data: bytes, audio_format: str = 'wav', 
                        language: Optional[str] = None, 
                        response_format: Optional[str] = None,
                        temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Transcribe audio to text using OpenAI Whisper API.

        Args:
            audio_data (bytes): Raw audio data
            audio_format (str): Audio format (wav, mp3, etc.)
            language (str, optional): Language code (defaults to 'en')
            response_format (str, optional): Response format (json, text, etc.)
            temperature (float, optional): Temperature for transcription consistency

        Returns:
            dict: Transcription response containing text and metadata

        Raises:
            WhisperAPIError: If transcription fails
        """
        logger.info("Starting audio transcription with Whisper",
                    audio_size=len(audio_data),
                    audio_format=audio_format,
                    language=language or self.default_config['language'])

        try:
            # Make the API request
            response = self._make_request(
                audio_data=audio_data,
                audio_format=audio_format,
                language=language,
                response_format=response_format,
                temperature=temperature
            )

            # Parse response based on format
            response_format = response_format or self.default_config['response_format']
            
            if response_format == 'json':
                result = response.json()
                transcribed_text = result.get('text', '')
                
                if not transcribed_text:
                    logger.warning("Whisper API returned empty transcription", response_data=result)
                    return {
                        'text': '',
                        'language': language or self.default_config['language'],
                        'confidence': 0.0,
                        'processing_time': 0.0
                    }

                logger.info("Audio transcription successful",
                            text_preview=transcribed_text[:100],
                            text_length=len(transcribed_text))

                return {
                    'text': transcribed_text,
                    'language': result.get('language', language or self.default_config['language']),
                    'confidence': result.get('confidence', 1.0),  # Whisper doesn't always return confidence
                    'segments': result.get('segments', []),
                    'processing_time': result.get('processing_time', 0.0)
                }
            
            else:
                # For text format, return the raw response
                transcribed_text = response.text.strip()
                
                if not transcribed_text:
                    logger.warning("Whisper API returned empty transcription")
                    return {
                        'text': '',
                        'language': language or self.default_config['language'],
                        'confidence': 0.0,
                        'processing_time': 0.0
                    }

                logger.info("Audio transcription successful",
                            text_preview=transcribed_text[:100],
                            text_length=len(transcribed_text))

                return {
                    'text': transcribed_text,
                    'language': language or self.default_config['language'],
                    'confidence': 1.0,
                    'segments': [],
                    'processing_time': 0.0
                }

        except WhisperAPIError:
            raise  # Re-raise custom API errors
        except Exception as e:
            logger.error("An unexpected error occurred during Whisper transcription", error=str(e))
            raise WhisperAPIError(f"Audio transcription failed: {str(e)}")

    def get_supported_formats(self) -> list:
        """
        Get list of supported audio formats.

        Returns:
            list: List of supported audio formats
        """
        return self.supported_formats.copy()

    def get_max_file_size(self) -> int:
        """
        Get maximum allowed file size in bytes.

        Returns:
            int: Maximum file size in bytes
        """
        return self.max_file_size


# Convenience function for module-level usage
def create_whisper_client() -> WhisperSTTClient:
    """
    Create and return a configured Whisper STT client.

    Returns:
        WhisperSTTClient: Configured client instance
    """
    return WhisperSTTClient()


# Example usage for testing
if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

    # Configure structlog for the example usage
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
        whisper_client = create_whisper_client()
        
        print(f"Whisper STT client initialized successfully!")
        print(f"Supported formats: {whisper_client.get_supported_formats()}")
        print(f"Max file size: {whisper_client.get_max_file_size() / 1024 / 1024:.1f} MB")
        
        # Example: Test with a dummy audio file (you would need an actual audio file)
        # with open("test_audio.wav", "rb") as f:
        #     audio_data = f.read()
        #     result = whisper_client.transcribe_audio(audio_data, "wav")
        #     print(f"Transcription: {result['text']}")
        
    except WhisperAPIError as e:
        print(f"Whisper API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}") 