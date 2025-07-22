"""
Murf API Service Module

This module handles interactions with the Murf API for Text-to-Speech (TTS) functionality,
including true streaming audio capabilities via the dedicated streaming endpoint.

Note: Speech-to-Text (STT) functionality has been moved to the Whisper service.

Dependencies:
- requests: For HTTP API calls
- base64: For audio data encoding/decoding (as Murf TTS non-streaming returns base64)
- json: For API request/response handling
- os: For environment variable access
- structlog: For structured logging
- time: For retry mechanisms (exponential backoff)
- pydub: (Optional, for example usage only) For audio manipulation, e.g., creating dummy audio.
         Ensure 'pydub' and its dependencies (like 'ffmpeg' or 'libav') are installed if used.
"""

import os
import json
import base64
import time
import requests
from typing import Optional, Dict, Any, Iterator, Union
import structlog

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import get_config

# Load configuration
config = get_config()

# Initialize structured logger for this module.
# Global structlog configuration should be done in app.py or a central config.
logger = structlog.get_logger(__name__)


class MurfAPIError(Exception):
    """Custom exception for Murf API related errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        # Log the error immediately when the exception is created
        logger.error("Murf API Error", message=message, status_code=status_code, response_data=response_data)


class MurfAPIClient:
    """
    Client for interacting with Murf API services (STT and TTS).
    API keys and configurations are strictly loaded from environment variables.
    """

    def __init__(self):
        """
        Initialize the Murf API client.
        Loads API key from environment variables and other settings from config.
        """
        # Load sensitive API key from environment variable
        self.api_key = os.getenv('MURF_API_KEY')
        # Load configuration from config.py
        self.base_url = config['MURF_BASE_URL']
        self.stream_url = config['MURF_STREAM_URL']

        if not self.api_key:
            logger.critical("MURF_API_KEY environment variable not set. Cannot initialize Murf API client.")
            raise MurfAPIError("MURF_API_KEY is required. Set MURF_API_KEY environment variable in .env file.")

        # Default headers for API requests
        self.headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json', # Default for JSON payloads
            'Accept': 'application/json'
        }

        # Default TTS configuration from config.py
        self.default_tts_config = {
            'voice_id': config['MURF_DEFAULT_VOICE_ID'],
            'speed': config['MURF_DEFAULT_SPEED'],
            'pitch': config['MURF_DEFAULT_PITCH'],
            'volume': config['MURF_DEFAULT_VOLUME'],
            'audio_format': config['MURF_DEFAULT_AUDIO_FORMAT'],
            'sample_rate': config['MURF_DEFAULT_SAMPLE_RATE']
        }

        # Request timeout and retry configuration from config.py
        self.timeout = config['MURF_REQUEST_TIMEOUT']
        self.max_retries = config['MURF_MAX_RETRIES']
        self.retry_delay_base = config['MURF_RETRY_DELAY_BASE']

        logger.info("Murf API client initialized",
                    base_url=self.base_url,
                    stream_url=self.stream_url,
                    default_voice_id=self.default_tts_config['voice_id'])

    def _make_request(self, method: str, url: str, json_data: Optional[Dict] = None,
                      files: Optional[Dict] = None, params: Optional[Dict] = None, stream: bool = False) -> requests.Response:
        """
        Make HTTP request to Murf API with retry logic and comprehensive error handling.

        Args:
            method (str): HTTP method (GET, POST, etc.)
            url (str): The full URL for the API endpoint.
            json_data (dict, optional): Request payload for JSON body.
            files (dict, optional): Files to upload (for multipart/form-data).
            params (dict, optional): Query parameters for GET requests.
            stream (bool): Whether to keep the connection open to read response in chunks.

        Returns:
            requests.Response: API response object.

        Raises:
            MurfAPIError: If the request fails after all retries or due to an API-specific error.
        """
        headers = self.headers.copy()

        # Adjust headers for file uploads (requests library handles Content-Type for multipart)
        if files:
            headers.pop('Content-Type', None)
        
        # For the streaming endpoint, the Content-Type might need to be 'audio/mpeg' or similar
        # if the request body is raw audio, or 'application/json' if it's a JSON payload.
        # We'll assume 'application/json' for the text-to-speech_streaming request,
        # and 'audio/wav' or similar for STT file uploads.

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Making Murf API request",
                             method=method,
                             url=url,
                             attempt=attempt + 1,
                             max_retries=self.max_retries,
                             json_data_present=bool(json_data),
                             files_present=bool(files),
                             params_present=bool(params),
                             stream_request=stream)

                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data, # Use json for JSON payloads
                    files=files,    # Use files for multipart/form-data
                    params=params,  # Use params for URL query parameters (GET requests)
                    timeout=self.timeout,
                    stream=stream   # Keep connection open for reading large responses
                )

                # Raise an HTTPError for bad responses (4xx or 5xx)
                response.raise_for_status()

                logger.debug("Murf API request successful", status_code=response.status_code)
                return response

            except requests.exceptions.HTTPError as e:
                error_data = None
                try:
                    error_data = e.response.json()
                except json.JSONDecodeError:
                    error_data = {'raw_response': e.response.text}
                
                logger.error("Murf API HTTP error",
                             status_code=e.response.status_code,
                             error_details=error_data,
                             url=url,
                             attempt=attempt + 1)
                
                # For 4xx errors (client errors), don't retry, raise immediately
                if 400 <= e.response.status_code < 500:
                    raise MurfAPIError(
                        f"Murf API client error: {e.response.status_code} - {error_data.get('message', error_data.get('error', 'Unknown client error'))}",
                        status_code=e.response.status_code,
                        response_data=error_data
                    )
                # For 5xx errors (server errors), retry if attempts remain
                elif attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Murf API server error, retrying...",
                                   error=str(e),
                                   status_code=e.response.status_code,
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Murf API server error failed after all retries", error=str(e))
                    raise MurfAPIError(f"Murf API server error after {self.max_retries} retries: {str(e)}",
                                       status_code=e.response.status_code,
                                       response_data=error_data)

            except requests.exceptions.ConnectionError as e:
                # Handle network-related errors (e.g., DNS failure, refused connection)
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Murf API connection error, retrying...",
                                   error=str(e),
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Murf API connection failed after all retries", error=str(e))
                    raise MurfAPIError(f"Murf API connection failed after {self.max_retries} retries: {str(e)}")

            except requests.exceptions.Timeout as e:
                # Handle request timeouts
                if attempt < self.max_retries:
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning("Murf API request timed out, retrying...",
                                   error=str(e),
                                   wait_time=wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Murf API request timed out after all retries", error=str(e))
                    raise MurfAPIError(f"Murf API request timed out after {self.max_retries} retries: {str(e)}")

            except requests.exceptions.RequestException as e:
                # Catch any other requests-related exceptions
                logger.error("An unexpected requests error occurred with Murf API", error=str(e))
                raise MurfAPIError(f"An unexpected API request error occurred: {str(e)}")

            except json.JSONDecodeError as e:
                # Handle cases where response is not valid JSON when expected
                logger.error("Failed to decode JSON response from Murf API", error=str(e), raw_response=response.text)
                raise MurfAPIError(f"Invalid JSON response from Murf API: {str(e)}")

            except Exception as e:
                # Catch any other unexpected exceptions
                logger.critical("An unhandled error occurred in _make_request for Murf API", error=str(e))
                raise MurfAPIError(f"An unhandled error occurred during Murf API request: {str(e)}")



    def text_to_speech(self, text: str,
                       voice_id: Optional[str] = None,
                       speed: Optional[float] = None,
                       pitch: Optional[float] = None,
                       volume: Optional[float] = None,
                       audio_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert text to speech using Murf TTS API (`POST /v1/synthesize`).
        This method is for non-streaming synthesis, returning base64 encoded audio.

        Args:
            text (str): Text to convert to speech.
            voice_id (str, optional): Voice ID to use. Defaults to MURF_DEFAULT_VOICE_ID.
            speed (float, optional): Speech speed (0.5 to 2.0).
            pitch (float, optional): Pitch adjustment (-20 to 20).
            volume (float, optional): Volume level (0.0 to 2.0).
            audio_format (str, optional): Output audio format (e.g., 'mp3', 'wav').

        Returns:
            dict: TTS response containing base64 encoded audio data and metadata.

        Raises:
            MurfAPIError: If TTS conversion fails.
        """
        logger.info("Starting text-to-speech conversion (non-streaming) with Murf TTS",
                    text_length=len(text),
                    voice_id=voice_id or self.default_tts_config['voice_id'])

        tts_full_url = f"{self.base_url}/speech/generate" # Updated to current Murf API endpoint

        # Use default values if not provided
        config = {
            'voice_id': voice_id or self.default_tts_config['voice_id'],
            'speed': speed if speed is not None else self.default_tts_config['speed'],
            'pitch': pitch if pitch is not None else self.default_tts_config['pitch'],
            'volume': volume if volume is not None else self.default_tts_config['volume'],
            'audio_format': audio_format or self.default_tts_config['audio_format'],
            'sample_rate': self.default_tts_config['sample_rate']
        }

        request_data = {
            'text': text,
            'voice_id': config['voice_id'], # Murf docs show voice_id directly in payload
            'audio_format': config['audio_format'],
            'speed': config['speed'],
            'pitch': config['pitch'],
            'volume': config['volume'],
            'sample_rate': config['sample_rate']
        }

        try:
            response = self._make_request(
                method='POST',
                url=tts_full_url,
                json_data=request_data # Send as JSON body
            )

            result = response.json()
            audio_file_url = result.get('audioFile', '')
            encoded_audio = result.get('encodedAudio', '')
            
            # Murf API returns either a direct audio file URL or base64 encoded audio
            if audio_file_url:
                # Download the audio file from the URL
                audio_response = requests.get(audio_file_url)
                if audio_response.status_code == 200:
                    audio_data_base64 = base64.b64encode(audio_response.content).decode('utf-8')
                else:
                    logger.warning("Failed to download audio file from URL", 
                                   url=audio_file_url, 
                                   status_code=audio_response.status_code)
                    return {'audio_data': '', 'audio_format': config['audio_format']}
            elif encoded_audio:
                # Use the base64 encoded audio directly
                audio_data_base64 = encoded_audio
            else:
                logger.warning("Murf TTS returned empty audio data.", response_data=result)
                return {'audio_data': '', 'audio_format': config['audio_format']}

            logger.info("Text-to-speech conversion successful",
                        audio_duration=result.get('audioLengthInSeconds', 0.0),
                        audio_format=config['audio_format'],
                        audio_url=audio_file_url[:100] if audio_file_url else None)

            return {
                'audio_data': audio_data_base64,
                'audio_format': config['audio_format'],
                'duration': result.get('audioLengthInSeconds', 0.0),
                'sample_rate': config['sample_rate'],
                'voice_id': config['voice_id'],
                'processing_time': result.get('processing_time', 0.0),
                'word_durations': result.get('wordDurations', [])
            }

        except MurfAPIError:
            raise
        except Exception as e:
            logger.error("An unexpected error occurred during Murf TTS conversion", error=str(e))
            raise MurfAPIError(f"TTS conversion failed: {str(e)}")

    def text_to_speech_streaming(self, text: str,
                                 voice_id: Optional[str] = None,
                                 **kwargs) -> Iterator[bytes]:
        """
        Convert text to speech with true streaming audio output using Murf's dedicated streaming API.
        Yields audio chunks as they are received.

        Args:
            text (str): Text to convert to speech.
            voice_id (str, optional): Voice ID to use. Defaults to MURF_DEFAULT_VOICE_ID.
            **kwargs: Additional TTS parameters like speed, pitch, volume, audio_format, chunk_size.

        Yields:
            bytes: Audio data chunks.

        Raises:
            MurfAPIError: If TTS synthesis or audio processing fails.
        """
        logger.info("Starting true streaming text-to-speech conversion with Murf TTS",
                    text_length=len(text),
                    voice_id=voice_id or self.default_tts_config['voice_id'],
                    stream_url=self.stream_url)

        # Use default values if not provided
        config = {
            'voice_id': voice_id or self.default_tts_config['voice_id'],
            'speed': kwargs.get('speed', self.default_tts_config['speed']),
            'pitch': kwargs.get('pitch', self.default_tts_config['pitch']),
            'volume': kwargs.get('volume', self.default_tts_config['volume']),
            'audio_format': kwargs.get('audio_format', self.default_tts_config['audio_format']),
            'sample_rate': self.default_tts_config['sample_rate'],
            'chunk_size': kwargs.get('chunk_size', 4096) # Default chunk size for iteration
        }

        # IMPORTANT: The request payload for the streaming endpoint might be different
        # from the non-streaming one. Assuming a JSON payload with 'text' and voice config.
        # Use the exact payload format from your working previous project
        payload = {
            "voiceId": config['voice_id'],
            "text": text,
            "format": "mp3",
            "streaming": True
        }
        
        # Use the same headers as your working project (no extra Accept header)
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            # Use the dedicated streaming URL with exact format from working project
            response = requests.post(
                url=self.stream_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True # Essential for streaming response from requests library
            )
            response.raise_for_status()

            # Iterate over the response content in chunks.
            # This assumes the API directly streams raw audio bytes.
            for chunk in response.iter_content(chunk_size=config['chunk_size']):
                if chunk:
                    yield chunk

            logger.info("True streaming text-to-speech conversion completed successfully.")

        except MurfAPIError:
            raise
        except Exception as e:
            logger.error("An unexpected error occurred during true Murf streaming TTS", error=str(e))
            raise MurfAPIError(f"True streaming TTS failed: {str(e)}")

    def get_available_voices(self, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Get list of available voices from Murf API (`GET /v1/voices`).

        Args:
            language (str, optional): Filter voices by language code (e.g., 'en-US').

        Returns:
            dict: List of available voices with metadata.

        Raises:
            MurfAPIError: If voice list retrieval fails.
        """
        logger.info("Retrieving available voices from Murf API", language=language)

        voices_full_url = f"{self.base_url}/voices" # Endpoint confirmed from Murf docs

        params = {}
        if language:
            params['language'] = language # Murf docs show 'language' as query param

        try:
            response = self._make_request(
                method='GET',
                url=voices_full_url,
                params=params # Use 'params' for GET query parameters
            )

            result = response.json()

            # Murf's response structure for voices is a list of voice objects.
            # The example in docs shows a direct array.
            if not isinstance(result, list): # If it's not a list, try to get 'voices' key
                 if 'voices' in result and isinstance(result['voices'], list):
                     result = result['voices']
                 else:
                     logger.warning("Murf voices API returned unexpected structure.", response_data=result)
                     return {'voices': []} # Return empty if structure is unexpected

            logger.info("Available voices retrieved successfully",
                        voice_count=len(result))

            return {'voices': result} # Wrap in 'voices' key for consistency with other methods' dict returns

        except MurfAPIError:
            raise
        except Exception as e:
            logger.error("An unexpected error occurred during Murf voice retrieval", error=str(e))
            raise MurfAPIError(f"Voice retrieval failed: {str(e)}")

    def validate_voice_config(self, voice_config: Dict[str, Any]) -> bool:
        """
        Validate voice configuration parameters against Murf's typical ranges.
        This is a client-side validation for common parameters.

        Args:
            voice_config (dict): Voice configuration to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        logger.debug("Validating voice configuration", config=voice_config)
        try:
            # Validate speed
            speed = voice_config.get('speed')
            if speed is not None and not (0.5 <= speed <= 2.0):
                logger.warning("Invalid speed value in voice config", speed=speed)
                return False

            # Validate pitch
            pitch = voice_config.get('pitch')
            if pitch is not None and not (-20 <= pitch <= 20):
                logger.warning("Invalid pitch value in voice config", pitch=pitch)
                return False

            # Validate volume
            volume = voice_config.get('volume')
            if volume is not None and not (0.0 <= volume <= 2.0):
                logger.warning("Invalid volume value in voice config", volume=volume)
                return False

            # Voice ID presence (optional, as default might be used)
            if 'voice_id' not in voice_config or not voice_config['voice_id']:
                logger.warning("Voice ID missing or empty in voice config. Default will be used if applicable.")
                # Depending on strictness, this could be a False or just a warning.
                # For now, it's a warning as a default exists.

            return True

        except Exception as e:
            logger.error("Voice configuration validation failed due to unexpected error", error=str(e), config=voice_config)
            return False


# Convenience function for module-level usage (strictly loads from .env)
def create_murf_client() -> MurfAPIClient:
    """
    Create and return a configured Murf API client.
    This function strictly loads configuration from environment variables.

    Returns:
        MurfAPIClient: Configured client instance.
    """
    return MurfAPIClient()

# Example Usage (for testing purposes, not part of the main app flow)
if __name__ == '__main__':
    # To run this example, ensure you have a .env file in the project root with:
    # MURF_API_KEY="your_actual_murf_api_key_here"
    # MURF_BASE_URL="https://api.murf.ai/v1"
    # MURF_STREAM_URL="https://api.murf.ai/v1/speech/stream"
    # MURF_DEFAULT_VOICE_ID="en-UK-gabriel" # Recommended from your input
    # MURF_DEFAULT_SPEED="1.0"
    # MURF_DEFAULT_PITCH="0.0"
    # MURF_DEFAULT_VOLUME="1.0"
    # MURF_DEFAULT_AUDIO_FORMAT="mp3"
    # MURF_DEFAULT_SAMPLE_RATE="22050"
    # MURF_REQUEST_TIMEOUT="30"
    # MURF_MAX_RETRIES="3"
    # MURF_RETRY_DELAY_BASE="2"

    from dotenv import load_dotenv
    import sys
    # Load .env file from the project root (customer-success-agent/.env)
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

    # Configure structlog for the example usage
    # In a real app, this would be done once at startup in app.py
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() # For development, use ConsoleRenderer
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    try:
        murf_client = create_murf_client()

        print("\n--- Testing get_available_voices ---")
        try:
            voices_response = murf_client.get_available_voices()
            voices = voices_response.get('voices', [])
            print(f"Found {len(voices)} voices.")
            if voices:
                print(f"First voice: {voices[0].get('name')} (ID: {voices[0].get('id')})")
                print(f"Consider setting MURF_DEFAULT_VOICE_ID='{voices[0].get('id')}' in your .env file.")
            else:
                print("No voices found. Check Murf API key and account status.")
        except MurfAPIError as e:
            print(f"Error getting voices: {e.message} (Status: {e.status_code})")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print("\n--- Testing text_to_speech_streaming (True Streaming) ---")
        test_text_stream = "Hello, this is a test of the Murf true streaming text-to-speech service. It should provide audio in real-time chunks."
        output_stream_file = "test_output_true_stream.mp3"
        print(f"Synthesizing streaming speech for: '{test_text_stream[:70]}...' to '{output_stream_file}'")
        try:
            # Ensure MURF_DEFAULT_VOICE_ID is set in .env or pass it here
            # voice_id_for_test = "YOUR_SPECIFIC_TEST_VOICE_ID"
            with open(output_stream_file, "wb") as f:
                for audio_chunk in murf_client.text_to_speech_streaming(test_text_stream):
                    f.write(audio_chunk)
            print(f"Streaming audio saved to {output_stream_file}")
        except MurfAPIError as e:
            print(f"Error during true streaming TTS: {e.message} (Status: {e.status_code})")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print("\n--- Note: STT functionality has been moved to Whisper service ---")
        print("Speech-to-Text is now handled by the Whisper service, not Murf.")


    except MurfAPIError as e:
        print(f"Murf API Client Initialization Error: {e.message} (Status: {e.status_code})")
    except Exception as e:
        print(f"An unhandled error occurred during Murf service testing: {e}")

