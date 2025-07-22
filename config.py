"""
Configuration File for AI-Powered Customer Success Agent

This file contains non-sensitive configuration settings.
"""

import os
from typing import Dict, Any, Optional

# --- Application Information ---
APP_NAME = "AI Customer Success Agent"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Real-time voice-powered customer success agent with AI capabilities"

# --- Murf AI API Configuration ---
MURF_BASE_URL = "https://api.murf.ai/v1"
MURF_STREAM_URL = "https://api.murf.ai/v1/speech/stream"

# Default Murf TTS Voice Configuration
MURF_DEFAULT_VOICE_ID = "en-UK-gabriel"  # Previously 'en-US-natalie'
MURF_DEFAULT_SPEED = 1.0
MURF_DEFAULT_PITCH = 0.0
MURF_DEFAULT_VOLUME = 1.0
MURF_DEFAULT_AUDIO_FORMAT = "mp3"
MURF_DEFAULT_SAMPLE_RATE = 24000

# Murf API Request Configuration
MURF_REQUEST_TIMEOUT = 30
MURF_MAX_RETRIES = 3
MURF_RETRY_DELAY_BASE = 2

# --- OpenAI GPT LLM Configuration ---
OPENAI_GPT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_GPT_MODEL = "gpt-4o"  # or "gpt-4" depending on your preference

# LLM Generation Parameters (adjust these for desired response behavior)
LLM_MAX_TOKENS = 1000
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95
LLM_FREQUENCY_PENALTY = 0.0
LLM_PRESENCE_PENALTY = 0.0
LLM_STOP_SEQUENCES = ""  # Comma-separated list of stop sequences (e.g., "STOP,END")

# LLM Request Configuration
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY_BASE = 2

# LLM Conversation Context Management
# Max number of messages to keep in conversation history
LLM_MAX_HISTORY_LENGTH = 20
# Max tokens for the entire conversation context (messages + system prompt)
# Adjust based on your LLM's context window (e.g., 8192, 16384, 32768 for GPT-4 variants)
# Keep it less than the model's actual context window to allow for the response.
LLM_MAX_TOKENS_PER_CONTEXT = 4000

# --- OpenAI Whisper STT Configuration ---
# OpenAI Whisper API endpoint (using official OpenAI API)
WHISPER_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"

# The Whisper model name
WHISPER_MODEL = "whisper-1"

# Whisper STT Configuration
WHISPER_LANGUAGE = "en"  # English only for now
WHISPER_RESPONSE_FORMAT = "json"  # json, text, srt, verbose_json, vtt
WHISPER_TEMPERATURE = 0.0  # Lower temperature for more consistent transcription

# Whisper Request Configuration
WHISPER_REQUEST_TIMEOUT = 30
WHISPER_MAX_RETRIES = 3
WHISPER_RETRY_DELAY_BASE = 2

# Audio file size limits for Whisper
WHISPER_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (OpenAI's limit)
WHISPER_SUPPORTED_FORMATS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

# --- Flask Application Configuration ---
FLASK_SECRET_KEY = "Vivek@1234!"  # Set as requested
FLASK_DEBUG = True  # Set to False in production
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_CORS_ORIGINS = "*"  # Comma-separated list of allowed origins (e.g., "http://localhost:3000,https://yourfrontend.com")

# --- Flask-SocketIO Configuration ---
SOCKETIO_CORS_ORIGINS = "http://localhost:3000"  # Comma-separated list of allowed origins for SocketIO
SOCKETIO_ASYNC_MODE = "threading"  # Can be 'eventlet', 'gevent', 'threading'
SOCKETIO_PING_TIMEOUT = 60  # seconds
SOCKETIO_PING_INTERVAL = 25  # seconds
SOCKETIO_MAX_HTTP_BUFFER_SIZE = 1048576  # 1MB (bytes)

# --- Session Management ---
SESSION_INACTIVITY_TIMEOUT_SECONDS = 3600  # 1 hour
SESSION_CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes
MURF_STREAM_CHUNK_SIZE = 4096  # Chunk size for streaming audio from Murf

# --- Input/Output Configuration ---
# Supported input methods
SUPPORTED_INPUT_METHODS = ["voice", "text"]
DEFAULT_INPUT_METHOD = "voice"

# Text input settings
TEXT_INPUT_MAX_LENGTH = 1000  # Maximum characters for text input
TEXT_INPUT_RATE_LIMIT = 10  # Maximum text requests per minute per session

# Audio input settings
AUDIO_INPUT_MAX_DURATION = 300  # Maximum audio duration in seconds (5 minutes)
AUDIO_INPUT_SUPPORTED_FORMATS = ["wav", "mp3", "webm", "ogg"]

# --- Environment-based Configuration Override ---
def get_config_value(key: str, default: Any = None, config_type: type = str) -> Any:
    """
    Get configuration value with environment variable override support.
    
    Args:
        key (str): Configuration key name
        default (Any): Default value if not found
        config_type (type): Type to convert the value to
        
    Returns:
        Any: Configuration value
    """
    # Check environment variable first
    env_value = os.getenv(key)
    if env_value is not None:
        if config_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif config_type == int:
            return int(env_value)
        elif config_type == float:
            return float(env_value)
        else:
            return env_value
    
    # Use default from this config file
    return default

# --- Configuration Dictionary for Easy Access ---
CONFIG = {
    # Application
    "APP_NAME": APP_NAME,
    "APP_VERSION": APP_VERSION,
    "APP_DESCRIPTION": APP_DESCRIPTION,
    
    # Murf API
    "MURF_BASE_URL": MURF_BASE_URL,
    "MURF_STREAM_URL": MURF_STREAM_URL,
    "MURF_DEFAULT_VOICE_ID": MURF_DEFAULT_VOICE_ID,
    "MURF_DEFAULT_SPEED": MURF_DEFAULT_SPEED,
    "MURF_DEFAULT_PITCH": MURF_DEFAULT_PITCH,
    "MURF_DEFAULT_VOLUME": MURF_DEFAULT_VOLUME,
    "MURF_DEFAULT_AUDIO_FORMAT": MURF_DEFAULT_AUDIO_FORMAT,
    "MURF_DEFAULT_SAMPLE_RATE": MURF_DEFAULT_SAMPLE_RATE,
    "MURF_REQUEST_TIMEOUT": MURF_REQUEST_TIMEOUT,
    "MURF_MAX_RETRIES": MURF_MAX_RETRIES,
    "MURF_RETRY_DELAY_BASE": MURF_RETRY_DELAY_BASE,
    
    # OpenAI GPT LLM
    "OPENAI_GPT_ENDPOINT": OPENAI_GPT_ENDPOINT,
    "OPENAI_GPT_MODEL": OPENAI_GPT_MODEL,
    "LLM_MAX_TOKENS": LLM_MAX_TOKENS,
    "LLM_TEMPERATURE": LLM_TEMPERATURE,
    "LLM_TOP_P": LLM_TOP_P,
    "LLM_FREQUENCY_PENALTY": LLM_FREQUENCY_PENALTY,
    "LLM_PRESENCE_PENALTY": LLM_PRESENCE_PENALTY,
    "LLM_STOP_SEQUENCES": LLM_STOP_SEQUENCES,
    "LLM_REQUEST_TIMEOUT": LLM_REQUEST_TIMEOUT,
    "LLM_MAX_RETRIES": LLM_MAX_RETRIES,
    "LLM_RETRY_DELAY_BASE": LLM_RETRY_DELAY_BASE,
    "LLM_MAX_HISTORY_LENGTH": LLM_MAX_HISTORY_LENGTH,
    "LLM_MAX_TOKENS_PER_CONTEXT": LLM_MAX_TOKENS_PER_CONTEXT,
    
    # OpenAI Whisper STT
    "WHISPER_ENDPOINT": WHISPER_ENDPOINT,
    "WHISPER_MODEL": WHISPER_MODEL,
    "WHISPER_LANGUAGE": WHISPER_LANGUAGE,
    "WHISPER_RESPONSE_FORMAT": WHISPER_RESPONSE_FORMAT,
    "WHISPER_TEMPERATURE": WHISPER_TEMPERATURE,
    "WHISPER_REQUEST_TIMEOUT": WHISPER_REQUEST_TIMEOUT,
    "WHISPER_MAX_RETRIES": WHISPER_MAX_RETRIES,
    "WHISPER_RETRY_DELAY_BASE": WHISPER_RETRY_DELAY_BASE,
    "WHISPER_MAX_FILE_SIZE": WHISPER_MAX_FILE_SIZE,
    "WHISPER_SUPPORTED_FORMATS": WHISPER_SUPPORTED_FORMATS,
    
    # Flask
    "FLASK_SECRET_KEY": FLASK_SECRET_KEY,
    "FLASK_DEBUG": FLASK_DEBUG,
    "FLASK_HOST": FLASK_HOST,
    "FLASK_PORT": FLASK_PORT,
    "FLASK_CORS_ORIGINS": FLASK_CORS_ORIGINS,
    
    # SocketIO
    "SOCKETIO_CORS_ORIGINS": SOCKETIO_CORS_ORIGINS,
    "SOCKETIO_ASYNC_MODE": SOCKETIO_ASYNC_MODE,
    "SOCKETIO_PING_TIMEOUT": SOCKETIO_PING_TIMEOUT,
    "SOCKETIO_PING_INTERVAL": SOCKETIO_PING_INTERVAL,
    "SOCKETIO_MAX_HTTP_BUFFER_SIZE": SOCKETIO_MAX_HTTP_BUFFER_SIZE,
    
    # Session Management
    "SESSION_INACTIVITY_TIMEOUT_SECONDS": SESSION_INACTIVITY_TIMEOUT_SECONDS,
    "SESSION_CLEANUP_INTERVAL_SECONDS": SESSION_CLEANUP_INTERVAL_SECONDS,
    "MURF_STREAM_CHUNK_SIZE": MURF_STREAM_CHUNK_SIZE,
    
    # Input/Output
    "SUPPORTED_INPUT_METHODS": SUPPORTED_INPUT_METHODS,
    "DEFAULT_INPUT_METHOD": DEFAULT_INPUT_METHOD,
    "TEXT_INPUT_MAX_LENGTH": TEXT_INPUT_MAX_LENGTH,
    "TEXT_INPUT_RATE_LIMIT": TEXT_INPUT_RATE_LIMIT,
    "AUDIO_INPUT_MAX_DURATION": AUDIO_INPUT_MAX_DURATION,
    "AUDIO_INPUT_SUPPORTED_FORMATS": AUDIO_INPUT_SUPPORTED_FORMATS,
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary with environment variable overrides.
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary
    """
    # Apply environment variable overrides
    config_copy = CONFIG.copy()
    
    # Override with environment variables if they exist
    for key, default_value in config_copy.items():
        config_copy[key] = get_config_value(key, default_value, type(default_value))
    
    return config_copy 