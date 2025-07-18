"""
WebSocket Handler Module for AI Customer Success Agent

This module handles real-time audio conversations via WebSocket connections,
orchestrating the complete workflow:
1. Receive audio chunks from frontend
2. Convert speech to text using Whisper STT
3. Generate intelligent responses using LLM
4. Convert responses to streaming audio using Murf TTS
5. Stream audio back to frontend

Dependencies:
- Flask-SocketIO: For WebSocket communication (implicitly used via socketio object)
- structlog: For structured logging
- uuid: For session management
- time: For timing measurements and delays
- io: For audio buffer management
- threading: For concurrent audio streaming and background cleanup
- os: For environment variable access
"""

import os
import uuid
import time
import threading
from io import BytesIO
from collections import defaultdict
from typing import Dict, Any, Optional, List
import structlog
import base64 # For encoding/decoding audio chunks for transmission

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import get_config

# Load configuration
config = get_config()

# Import our service modules
from .llm import create_llm_service, LLMAPIError, ConversationContext
from .murf import create_murf_client, MurfAPIError
from .whisper import create_whisper_client, WhisperAPIError

# Initialize structured logger
logger = structlog.get_logger(__name__)


class AudioSessionManager:
    """
    Manages audio streaming sessions for WebSocket connections.
    Handles conversation context, audio buffering, and session state.
    """

    def __init__(self):
        # Session storage: session_id -> session_data
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.audio_buffers: Dict[str, BytesIO] = {}
        # Using defaultdict ensures a lock is created if accessed for a new session_id
        self.session_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Initialize service clients
        try:
            self.llm_client = create_llm_service()
            self.murf_client = create_murf_client()
            self.whisper_client = create_whisper_client()
            logger.info("Audio session manager initialized successfully")
        except (LLMAPIError, MurfAPIError, WhisperAPIError) as e:
            logger.critical("Failed to initialize service clients. Check API keys and network.", error=str(e))
            # Re-raise to prevent the application from starting if services can't be initialized
            raise
        except Exception as e:
            logger.critical("An unexpected error occurred during service client initialization.", error=str(e))
            raise

        # Configuration for inactive session cleanup from config.py
        self.inactive_session_timeout = config['SESSION_INACTIVITY_TIMEOUT_SECONDS']
        self.cleanup_interval = config['SESSION_CLEANUP_INTERVAL_SECONDS']
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup_event = threading.Event()

    def start_background_cleanup_thread(self):
        """Starts a background thread to periodically clean up inactive sessions."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            logger.info("Starting background session cleanup thread.")
            self._cleanup_thread = threading.Thread(target=self._run_cleanup_loop, daemon=True)
            self._cleanup_thread.start()
        else:
            logger.info("Background session cleanup thread is already running.")

    def stop_background_cleanup_thread(self):
        """Stops the background session cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.info("Stopping background session cleanup thread.")
            self._stop_cleanup_event.set()
            self._cleanup_thread.join(timeout=self.cleanup_interval + 5) # Give it a bit more time to finish
            if self._cleanup_thread.is_alive():
                logger.warning("Background cleanup thread did not terminate gracefully.")
            self._cleanup_thread = None
            self._stop_cleanup_event.clear()

    def _run_cleanup_loop(self):
        """The main loop for the background session cleanup thread."""
        while not self._stop_cleanup_event.is_set():
            self.cleanup_inactive_sessions(self.inactive_session_timeout)
            # Wait for the next cleanup interval or until stop event is set
            self._stop_cleanup_event.wait(self.cleanup_interval)
        logger.info("Background session cleanup loop terminated.")


    def create_session(self, session_id: str, customer_info: Optional[Dict] = None,
                       llm_config: Optional[Dict] = None) -> str:
        """
        Create a new audio conversation session.

        Args:
            session_id (str): WebSocket session identifier.
            customer_info (dict, optional): Customer information for personalization.
            llm_config (dict, optional): LLM specific configuration for this session (e.g., max_tokens, temperature).

        Returns:
            str: LLM conversation ID.
        """
        with self.session_locks[session_id]:
            if session_id in self.sessions:
                logger.warning("Attempted to create an existing session.", session_id=session_id)
                # Return existing conversation ID if session already exists
                return self.sessions[session_id]['conversation_id']

            try:
                # Create LLM conversation context, passing config for context window
                conversation_id = self.llm_client.create_conversation(
                    customer_info=customer_info or {},
                    max_history_length=config['LLM_MAX_HISTORY_LENGTH'],
                    max_tokens_per_context=config['LLM_MAX_TOKENS_PER_CONTEXT']
                )

                # Initialize session data
                self.sessions[session_id] = {
                    'conversation_id': conversation_id,
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'customer_info': customer_info or {},
                    'audio_format': 'wav',  # Default format for STT, client should specify
                    'is_processing': False,
                    'message_count': 0,
                    'total_audio_duration': 0.0,
                    'llm_session_config': llm_config or {} # Store LLM config specific to this session
                }

                # Initialize audio buffer
                self.audio_buffers[session_id] = BytesIO()

                logger.info("Created new audio session",
                           session_id=session_id,
                           conversation_id=conversation_id,
                           has_customer_info=bool(customer_info))

                return conversation_id

            except Exception as e:
                logger.error("Failed to create audio session",
                           session_id=session_id, error=str(e))
                raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by session ID."""
        return self.sessions.get(session_id)

    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session."""
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = time.time()
            logger.debug("Updated session activity", session_id=session_id)

    def cleanup_session(self, session_id: str):
        """Clean up session data and resources."""
        with self.session_locks[session_id]:
            try:
                # Clean up LLM conversation
                if session_id in self.sessions:
                    conversation_id = self.sessions[session_id]['conversation_id']
                    self.llm_client.cleanup_conversation(conversation_id)
                    del self.sessions[session_id]
                    logger.info("Removed session data", session_id=session_id)
                else:
                    logger.warning("Attempted to clean up non-existent session data", session_id=session_id)

                # Clean up audio buffer
                if session_id in self.audio_buffers:
                    self.audio_buffers[session_id].close()
                    del self.audio_buffers[session_id]
                    logger.debug("Closed and removed audio buffer", session_id=session_id)
                else:
                    logger.warning("Attempted to close non-existent audio buffer", session_id=session_id)
                
                # Note: The lock itself (from defaultdict) will remain in `self.session_locks`
                # but will be garbage collected if no longer referenced.
                # If you want to explicitly remove it:
                # if session_id in self.session_locks:
                #     del self.session_locks[session_id]

                logger.info("Cleaned up audio session resources", session_id=session_id)

            except Exception as e:
                logger.error("Error during session cleanup",
                           session_id=session_id, error=str(e))

    def accumulate_audio_chunk(self, session_id: str, audio_chunk: bytes) -> int:
        """
        Accumulate audio chunk in session buffer.

        Args:
            session_id (str): Session identifier
            audio_chunk (bytes): Audio data chunk

        Returns:
            int: Total bytes accumulated
        """
        if session_id not in self.audio_buffers:
            # This should ideally not happen if create_session is called first
            logger.error("Audio buffer not initialized for session. Creating new buffer.", session_id=session_id)
            self.audio_buffers[session_id] = BytesIO()

        self.audio_buffers[session_id].write(audio_chunk)
        total_bytes = self.audio_buffers[session_id].tell()

        logger.debug("Accumulated audio chunk",
                    session_id=session_id,
                    chunk_size=len(audio_chunk),
                    total_bytes=total_bytes)

        return total_bytes

    def get_accumulated_audio(self, session_id: str) -> bytes:
        """Get accumulated audio data and reset buffer."""
        if session_id not in self.audio_buffers:
            logger.warning("Attempted to retrieve audio from non-existent buffer", session_id=session_id)
            return b''

        audio_buffer = self.audio_buffers[session_id]
        audio_buffer.seek(0) # Move to the beginning of the buffer
        audio_data = audio_buffer.read()

        # Reset buffer for next audio stream (important for continuous input)
        self.audio_buffers[session_id] = BytesIO()

        logger.debug("Retrieved accumulated audio",
                    session_id=session_id,
                    audio_size=len(audio_data))

        return audio_data

    def cleanup_inactive_sessions(self, max_inactive_time: int = 3600):
        """
        Clean up sessions that have been inactive for more than max_inactive_time seconds.

        Args:
            max_inactive_time (int): Maximum inactive time in seconds (default: 1 hour)
        """
        try:
            current_time = time.time()
            inactive_sessions = []

            # Iterate over a copy of keys to allow modification during iteration
            for session_id in list(self.sessions.keys()):
                session_data = self.sessions[session_id]
                if current_time - session_data['last_activity'] > max_inactive_time:
                    inactive_sessions.append(session_id)

            for session_id in inactive_sessions:
                self.cleanup_session(session_id) # Use the full cleanup method
                logger.info("Cleaned up inactive session", session_id=session_id)

            if inactive_sessions:
                logger.info("Finished cleaning up inactive sessions", count=len(inactive_sessions))
            else:
                logger.debug("No inactive sessions to clean up.")

        except Exception as e:
            logger.error("Error cleaning up inactive sessions", error=str(e))


# Global session manager instance
audio_session_manager = AudioSessionManager()


def handle_client_connection(sid: str, auth: Optional[Dict] = None):
    """
    Handle new WebSocket client connection.
    
    Args:
        sid (str): Session ID from Flask-SocketIO.
        auth (dict, optional): Authentication data from client, potentially containing customer info.
    
    Returns:
        dict: Connection status and session details.
    """
    logger.info("New WebSocket client connected", session_id=sid)
    
    try:
        customer_info = {}
        llm_config = {}
        if auth and isinstance(auth, dict):
            customer_info = {
                'customer_id': auth.get('customer_id'),
                'name': auth.get('name'),
                'email': auth.get('email'),
                'plan': auth.get('plan', 'free'),
                'preferences': auth.get('preferences', {})
            }
            # Allow client to suggest LLM config (e.g., for testing different temperatures)
            llm_config = auth.get('llm_config', {})
        
        # Create session in the manager
        conversation_id = audio_session_manager.create_session(sid, customer_info, llm_config)
        
        # Send connection confirmation to client
        return {
            'status': 'connected',
            'session_id': sid,
            'conversation_id': conversation_id,
            'message': 'Connected to AI Customer Success Agent. Send audio chunks.'
        }
        
    except Exception as e:
        logger.error("Error handling client connection", 
                    session_id=sid, error=str(e))
        return {
            'status': 'error',
            'message': f'Failed to establish connection: {str(e)}'
        }


def handle_client_disconnection(sid: str):
    """
    Handle WebSocket client disconnection.
    
    Args:
        sid (str): Session ID from Flask-SocketIO.
    """
    logger.info("WebSocket client disconnected", session_id=sid)
    
    try:
        # Clean up session resources
        audio_session_manager.cleanup_session(sid)
        
    except Exception as e:
        logger.error("Error handling client disconnection", 
                    session_id=sid, error=str(e))


def handle_incoming_audio_stream(sid: str, data: Dict[str, Any], socketio=None):
    """
    Handle incoming audio stream from WebSocket client.
    This is the main orchestration function for the audio conversation flow.
    
    Args:
        sid (str): Session ID from Flask-SocketIO.
        data (dict): Audio data and metadata from client. Expected keys:
                     'audio_data' (base64 encoded string), 'is_final' (bool),
                     'format' (str, e.g., 'wav', 'mp3').
        socketio: Flask-SocketIO instance for emitting responses.
    """
    logger.debug("Received audio stream chunk", session_id=sid, data_keys=data.keys())
    
    session = audio_session_manager.get_session(sid)
    if not session:
        logger.warning("Audio stream received for non-existent session", session_id=sid)
        if socketio:
            socketio.emit('error', {
                'message': 'Session not found. Please reconnect.'
            }, room=sid)
        return

    # Use a lock to prevent race conditions if multiple audio chunks arrive very rapidly
    with audio_session_manager.session_locks[sid]:
        try:
            # Check if already processing a previous full audio segment
            if session.get('is_processing', False):
                logger.debug("Audio stream received while previous request is still processing. Skipping.", session_id=sid)
                # Optionally, send a status update to the client
                if socketio:
                    socketio.emit('status', {
                        'message': 'Currently processing previous request. Please wait or speak after response.'
                    }, room=sid)
                return
            
            audio_session_manager.update_session_activity(sid)
            
            # Extract and decode audio data
            audio_chunk_b64 = data.get('audio_data')
            is_final = data.get('is_final', False)
            # Client should ideally send the format of the audio it's sending
            audio_format = data.get('format', session.get('audio_format', 'wav')) 
            
            if not audio_chunk_b64:
                logger.warning("Received empty audio_data in chunk", session_id=sid)
                if is_final: # If it's the final chunk and empty, it's an issue
                     if socketio:
                        socketio.emit('error', {'message': 'No audio data received for processing.'}, room=sid)
                     session['is_processing'] = False # Reset flag
                return

            try:
                audio_chunk_bytes = base64.b64decode(audio_chunk_b64)
            except Exception as e:
                logger.error("Failed to base64 decode audio chunk", session_id=sid, error=str(e))
                if socketio:
                    socketio.emit('error', {'message': 'Invalid audio data format.'}, room=sid)
                if is_final: session['is_processing'] = False # Reset flag
                return
            
            # Accumulate audio chunk
            audio_session_manager.accumulate_audio_chunk(sid, audio_chunk_bytes)
            
            # If this is the final chunk, trigger full processing
            if is_final:
                logger.info("Final audio chunk received. Initiating full processing.", session_id=sid)
                
                # Mark as processing to prevent new audio chunks from starting new processes
                session['is_processing'] = True 
                
                # Get accumulated audio
                complete_audio = audio_session_manager.get_accumulated_audio(sid)
                
                if not complete_audio:
                    logger.warning("No complete audio data to process after final chunk.", session_id=sid)
                    if socketio:
                        socketio.emit('error', {
                            'message': 'No complete audio data received for processing.'
                        }, room=sid)
                    session['is_processing'] = False # Reset flag
                    return
                
                # Start processing in separate thread to avoid blocking the WebSocket event loop
                processing_thread = threading.Thread(
                    target=_process_audio_conversation,
                    args=(sid, complete_audio, audio_format, socketio)
                )
                processing_thread.daemon = True # Allow thread to exit with main program
                processing_thread.start()
            
            else:
                # Send chunk received confirmation (optional, but good for client feedback)
                if socketio:
                    socketio.emit('audio_chunk_received', {
                        'chunk_size': len(audio_chunk_bytes),
                        'status': 'received',
                        'is_final': False
                    }, room=sid)
        
        except Exception as e:
            logger.error("Unhandled error in handle_incoming_audio_stream", 
                        session_id=sid, error=str(e))
            if socketio:
                socketio.emit('error', {
                    'message': 'An internal error occurred while receiving audio.'
                }, room=sid)
            # Ensure processing flag is reset if an error occurs before thread starts
            if session:
                session['is_processing'] = False


def _process_audio_conversation(sid: str, audio_data: bytes, audio_format: str, socketio=None):
    """
    Process complete audio conversation workflow in a separate thread.
    
    Args:
        sid (str): Session ID.
        audio_data (bytes): Complete audio data for STT.
        audio_format (str): Audio format (e.g., 'wav', 'mp3').
        socketio: Flask-SocketIO instance for emitting responses.
    """
    session = audio_session_manager.get_session(sid)
    if not session:
        logger.error("Session not found in _process_audio_conversation. This should not happen.", session_id=sid)
        return # Cannot proceed without session

    conversation_id = session['conversation_id']
    llm_session_config = session.get('llm_session_config', {})
    
    start_time = time.time()
    
    try:
        # Step 1: Speech-to-Text using Whisper STT
        logger.info("Starting STT conversion", session_id=sid, audio_size=len(audio_data))
        
        if socketio:
            socketio.emit('processing_status', {
                'stage': 'stt',
                'message': 'Converting speech to text...'
            }, room=sid)
        
        stt_start = time.time()
        stt_result = audio_session_manager.whisper_client.transcribe_audio(
            audio_data=audio_data,
            audio_format=audio_format,
            language='en'  # Whisper uses 'en' instead of 'en-US'
        )
        stt_duration = time.time() - stt_start
        
        transcribed_text = stt_result.get('text', '').strip()
        
        if not transcribed_text:
            logger.warning("STT returned empty transcription", session_id=sid)
            if socketio:
                socketio.emit('error', {
                    'message': 'I could not understand what you said. Please try speaking more clearly.'
                }, room=sid)
            return
        
        logger.info("STT conversion completed", 
                   session_id=sid,
                   transcribed_text_preview=transcribed_text[:100],
                   stt_duration=stt_duration)
        
        # Step 2: Generate LLM response
        if socketio:
            socketio.emit('processing_status', {
                'stage': 'llm',
                'message': 'Generating intelligent response...',
                'transcribed_text': transcribed_text
            }, room=sid)
        
        llm_start = time.time()
        # Use LLM config from session or defaults from config.py
        llm_response = audio_session_manager.llm_client.chat_completion(
            message=transcribed_text,
            conversation_id=conversation_id,
            temperature=llm_session_config.get('temperature', config['LLM_TEMPERATURE']),
            max_tokens=llm_session_config.get('max_tokens', config['LLM_MAX_TOKENS'])
        )
        llm_duration = time.time() - llm_start
        
        response_text = llm_response['response']
        
        logger.info("LLM response generated", 
                   session_id=sid,
                   response_text_preview=response_text[:100],
                   llm_duration=llm_duration)
        
        # Step 3: Convert response to streaming audio using Murf TTS
        if socketio:
            socketio.emit('processing_status', {
                'stage': 'tts',
                'message': 'Converting response to speech...',
                'response_text': response_text
            }, room=sid)
        
        tts_start = time.time()
        
        # Stream TTS audio back to client
        _stream_tts_audio(sid, response_text, socketio)
        
        tts_duration = time.time() - tts_start
        
        # Update session statistics
        total_duration = time.time() - start_time
        session['message_count'] += 1
        session['total_audio_duration'] += total_duration # This might be total processing time, not audio duration
        
        logger.info("Audio conversation completed", 
                   session_id=sid,
                   conversation_id=conversation_id,
                   total_duration=total_duration,
                   stt_duration=stt_duration,
                   llm_duration=llm_duration,
                   tts_duration=tts_duration)
        
        # Send final completion status
        if socketio:
            socketio.emit('conversation_completed', {
                'transcribed_text': transcribed_text,
                'response_text': response_text,
                'processing_time': {
                    'stt': stt_duration,
                    'llm': llm_duration,
                    'tts': tts_duration,
                    'total': total_duration
                }
            }, room=sid)
    
    except (MurfAPIError, LLMAPIError, WhisperAPIError) as e:
        logger.error("API error during conversation processing", 
                    session_id=sid, error=str(e), api_status_code=getattr(e, 'status_code', 'N/A'))
        if socketio:
            socketio.emit('error', {
                'message': f'Service error: {str(e)}. Please try again.'
            }, room=sid)
    
    except Exception as e:
        logger.critical("Unexpected error during conversation processing", 
                    session_id=sid, error=str(e), exc_info=True) # Log full traceback
        if socketio:
            socketio.emit('error', {
                'message': 'An unexpected internal error occurred. Our team has been notified.'
            }, room=sid)
    
    finally:
        # Reset processing flag, ensuring it's always reset
        with audio_session_manager.session_locks[sid]:
            session = audio_session_manager.get_session(sid)
            if session:
                session['is_processing'] = False
                logger.debug("Processing flag reset for session", session_id=sid)


def _stream_tts_audio(sid: str, text: str, socketio=None):
    """
    Stream TTS audio back to client using Murf streaming TTS.
    
    Args:
        sid (str): Session ID.
        text (str): Text to convert to speech.
        socketio: Flask-SocketIO instance for emitting audio chunks.
    """
    try:
        logger.info("Starting TTS streaming", session_id=sid, text_length=len(text))
        
        session = audio_session_manager.get_session(sid)
        if not session:
            logger.error("Session not found for TTS streaming. Cannot stream audio.", session_id=sid)
            return

        customer_info = session.get('customer_info', {})
        voice_preferences = customer_info.get('preferences', {}).get('voice', {})
        
        # Configure TTS parameters. Prioritize session preferences, then config.py defaults.
        tts_config = {
            'voice_id': voice_preferences.get('voice_id') or config['MURF_DEFAULT_VOICE_ID'],
            'speed': voice_preferences.get('speed', config['MURF_DEFAULT_SPEED']),
            'pitch': voice_preferences.get('pitch', config['MURF_DEFAULT_PITCH']),
            'volume': voice_preferences.get('volume', config['MURF_DEFAULT_VOLUME']),
            'audio_format': voice_preferences.get('audio_format') or config['MURF_DEFAULT_AUDIO_FORMAT'],
            'sample_rate': voice_preferences.get('sample_rate') or config['MURF_DEFAULT_SAMPLE_RATE'],
            'chunk_size': config['MURF_STREAM_CHUNK_SIZE']
        }
        
        if socketio:
            socketio.emit('audio_stream_start', {
                'message': 'Starting audio response stream',
                'format': tts_config['audio_format']
            }, room=sid)
        
        chunk_count = 0
        total_bytes = 0
        
        for audio_chunk in audio_session_manager.murf_client.text_to_speech_streaming(
            text=text,
            voice_id=tts_config['voice_id'],
            speed=tts_config['speed'],
            pitch=tts_config['pitch'],
            volume=tts_config['volume'],
            audio_format=tts_config['audio_format'],
            chunk_size=tts_config['chunk_size'] # Pass chunk_size to murf_client if it supports it
        ):
            if audio_chunk:
                chunk_count += 1
                total_bytes += len(audio_chunk)
                
                # Encode audio chunk as base64 for JSON transmission over WebSocket
                encoded_chunk = base64.b64encode(audio_chunk).decode('utf-8')
                
                # Send audio chunk to client
                if socketio:
                    socketio.emit('audio_chunk', {
                        'chunk_data': encoded_chunk,
                        'chunk_number': chunk_count,
                        'chunk_size': len(audio_chunk),
                        'format': tts_config['audio_format']
                    }, room=sid)
                
                # Small delay to prevent overwhelming the client, especially over slow networks
                time.sleep(0.005) # Reduced slightly from 0.01 for potentially faster streaming
        
        # Send stream completion
        if socketio:
            socketio.emit('audio_stream_complete', {
                'message': 'Audio response stream completed',
                'total_chunks': chunk_count,
                'total_bytes': total_bytes
            }, room=sid)
        
        logger.info("TTS streaming completed", 
                   session_id=sid,
                   chunk_count=chunk_count,
                   total_bytes=total_bytes)
    
    except Exception as e:
        logger.error("Error during TTS streaming", session_id=sid, error=str(e), exc_info=True)
        if socketio:
            socketio.emit('error', {
                'message': 'Error streaming audio response. Please try again.'
            }, room=sid)


def handle_get_conversation_history(sid: str, data: Dict[str, Any], socketio=None):
    """
    Handle request for conversation history.
    
    Args:
        sid (str): Session ID.
        data (dict): Request data (currently unused, but kept for consistency).
        socketio: Flask-SocketIO instance for emitting response.
    """
    logger.info("Received request for conversation history", session_id=sid)
    try:
        session = audio_session_manager.get_session(sid)
        if not session:
            logger.warning("Request for history from non-existent session", session_id=sid)
            if socketio:
                socketio.emit('error', {
                    'message': 'Session not found. Please reconnect.'
                }, room=sid)
            return
        
        conversation_id = session['conversation_id']
        conversation_context = audio_session_manager.llm_client.get_conversation(conversation_id)
        
        if conversation_context:
            # Extract conversation history (excluding system messages)
            history = []
            for msg in conversation_context.messages:
                if msg['role'] != 'system':
                    history.append({
                        'role': msg['role'],
                        'content': msg['content'],
                        'timestamp': msg.get('timestamp')
                    })
            
            if socketio:
                socketio.emit('conversation_history', {
                    'history': history,
                    'conversation_summary': conversation_context.get_conversation_summary()
                }, room=sid)
            logger.info("Conversation history sent", session_id=sid, message_count=len(history))
        else:
            logger.warning("Conversation context not found for session", session_id=sid, conversation_id=conversation_id)
            if socketio:
                socketio.emit('error', {
                    'message': 'Conversation history not found.'
                }, room=sid)
    
    except Exception as e:
        logger.error("Error retrieving conversation history", 
                    session_id=sid, error=str(e), exc_info=True)
        if socketio:
            socketio.emit('error', {
                'message': 'Error retrieving conversation history.'
            }, room=sid)


def handle_analyze_sentiment(sid: str, data: Dict[str, Any], socketio=None):
    """
    Handle request for sentiment analysis of current conversation.
    
    Args:
        sid (str): Session ID.
        data (dict): Request data (currently unused).
        socketio: Flask-SocketIO instance for emitting response.
    """
    logger.info("Received request for sentiment analysis", session_id=sid)
    try:
        session = audio_session_manager.get_session(sid)
        if not session:
            logger.warning("Request for sentiment from non-existent session", session_id=sid)
            if socketio:
                socketio.emit('error', {
                    'message': 'Session not found. Please reconnect.'
                }, room=sid)
            return
        
        conversation_id = session['conversation_id']
        
        # Analyze sentiment
        sentiment_result = audio_session_manager.llm_client.analyze_customer_sentiment(conversation_id)
        
        if socketio:
            socketio.emit('sentiment_analysis', sentiment_result, room=sid)
        
        logger.info("Sentiment analysis completed and sent", 
                   session_id=sid,
                   sentiment=sentiment_result.get('sentiment'))
    
    except LLMAPIError as e:
        logger.error("LLM API error during sentiment analysis", 
                    session_id=sid, error=str(e), api_status_code=getattr(e, 'status_code', 'N/A'))
        if socketio:
            socketio.emit('error', {
                'message': f'AI sentiment analysis error: {e.message}.'
            }, room=sid)
    except Exception as e:
        logger.error("Unexpected error analyzing sentiment", session_id=sid, error=str(e), exc_info=True)
        if socketio:
            socketio.emit('error', {
                'message': 'An unexpected error occurred during sentiment analysis.'
            }, room=sid)


def handle_generate_summary(sid: str, data: Dict[str, Any], socketio=None):
    """
    Handle request for conversation summary.
    
    Args:
        sid (str): Session ID.
        data (dict): Request data (currently unused).
        socketio: Flask-SocketIO instance for emitting response.
    """
    logger.info("Received request for conversation summary", session_id=sid)
    try:
        session = audio_session_manager.get_session(sid)
        if not session:
            logger.warning("Request for summary from non-existent session", session_id=sid)
            if socketio:
                socketio.emit('error', {
                    'message': 'Session not found. Please reconnect.'
                }, room=sid)
            return
        
        conversation_id = session['conversation_id']
        
        # Generate summary
        summary_result = audio_session_manager.llm_client.generate_summary(conversation_id)
        
        if socketio:
            socketio.emit('conversation_summary', summary_result, room=sid)
        
        logger.info("Conversation summary generated and sent", 
                   session_id=sid)
    
    except LLMAPIError as e:
        logger.error("LLM API error during summary generation", 
                    session_id=sid, error=str(e), api_status_code=getattr(e, 'status_code', 'N/A'))
        if socketio:
            socketio.emit('error', {
                'message': f'AI summary generation error: {e.message}.'
            }, room=sid)
    except Exception as e:
        logger.error("Unexpected error generating summary", session_id=sid, error=str(e), exc_info=True)
        if socketio:
            socketio.emit('error', {
                'message': 'An unexpected error occurred during summary generation.'
            }, room=sid)


def handle_text_input(sid: str, data: Dict[str, Any], socketio=None):
    """
    Handle incoming text input from WebSocket client.
    This processes text messages through the LLM without STT processing.
    
    Args:
        sid (str): Session ID from Flask-SocketIO.
        data (dict): Text input data from client. Expected keys:
                     'text' (str), 'response_format' (str: 'text', 'audio', 'both').
        socketio: Flask-SocketIO instance for emitting responses.
    """
    logger.info("Received text input", session_id=sid)
    
    session = audio_session_manager.get_session(sid)
    if not session:
        logger.warning("Text input received for non-existent session", session_id=sid)
        if socketio:
            socketio.emit('error', {
                'message': 'Session not found. Please reconnect.'
            }, room=sid)
        return

    # Use a lock to prevent race conditions
    with audio_session_manager.session_locks[sid]:
        try:
            # Check if already processing
            if session.get('is_processing', False):
                logger.debug("Text input received while previous request is still processing", session_id=sid)
                if socketio:
                    socketio.emit('status', {
                        'message': 'Currently processing previous request. Please wait.'
                    }, room=sid)
                return
            
            # Mark as processing
            session['is_processing'] = True
            audio_session_manager.update_session_activity(sid)
            
            # Extract and validate text input
            text_input = data.get('text', '').strip()
            response_format = data.get('response_format', 'text')  # 'text', 'audio', 'both'
            
            if not text_input:
                logger.warning("Received empty text input", session_id=sid)
                if socketio:
                    socketio.emit('error', {
                        'message': 'Empty text input received.'
                    }, room=sid)
                session['is_processing'] = False
                return
            
            # Validate text input length
            if len(text_input) > config['TEXT_INPUT_MAX_LENGTH']:
                logger.warning("Text input exceeds maximum length", 
                             session_id=sid, 
                             text_length=len(text_input),
                             max_length=config['TEXT_INPUT_MAX_LENGTH'])
                if socketio:
                    socketio.emit('error', {
                        'message': f'Text input too long. Maximum {config["TEXT_INPUT_MAX_LENGTH"]} characters allowed.'
                    }, room=sid)
                session['is_processing'] = False
                return
            
            # Rate limiting check (optional)
            if not _check_text_input_rate_limit(sid):
                logger.warning("Text input rate limit exceeded", session_id=sid)
                if socketio:
                    socketio.emit('error', {
                        'message': 'Rate limit exceeded. Please wait before sending another message.'
                    }, room=sid)
                session['is_processing'] = False
                return
            
            # Start processing in separate thread
            processing_thread = threading.Thread(
                target=_process_text_conversation,
                args=(sid, text_input, response_format, socketio)
            )
            processing_thread.daemon = True
            processing_thread.start()
            
            logger.info("Text input processing initiated", 
                       session_id=sid, 
                       text_preview=text_input[:100],
                       response_format=response_format)
        
        except Exception as e:
            logger.error("Error handling text input", 
                        session_id=sid, error=str(e), exc_info=True)
            if socketio:
                socketio.emit('error', {
                    'message': 'An internal error occurred while processing text input.'
                }, room=sid)
            # Reset processing flag
            if session:
                session['is_processing'] = False


def _process_text_conversation(sid: str, text_input: str, response_format: str, socketio=None):
    """
    Process text conversation workflow in a separate thread.
    
    Args:
        sid (str): Session ID.
        text_input (str): User's text input.
        response_format (str): Desired response format ('text', 'audio', 'both').
        socketio: Flask-SocketIO instance for emitting responses.
    """
    session = audio_session_manager.get_session(sid)
    if not session:
        logger.error("Session not found in _process_text_conversation", session_id=sid)
        return

    conversation_id = session['conversation_id']
    llm_session_config = session.get('llm_session_config', {})
    
    start_time = time.time()
    
    try:
        # Send processing status
        if socketio:
            socketio.emit('processing_status', {
                'stage': 'llm',
                'message': 'Generating intelligent response...',
                'input_text': text_input,
                'input_method': 'text'
            }, room=sid)
        
        # Generate LLM response
        llm_start = time.time()
        llm_response = audio_session_manager.llm_client.chat_completion(
            message=text_input,
            conversation_id=conversation_id,
            temperature=llm_session_config.get('temperature', config['LLM_TEMPERATURE']),
            max_tokens=llm_session_config.get('max_tokens', config['LLM_MAX_TOKENS'])
        )
        llm_duration = time.time() - llm_start
        
        response_text = llm_response['response']
        
        logger.info("LLM response generated for text input", 
                   session_id=sid,
                   response_text_preview=response_text[:100],
                   llm_duration=llm_duration)
        
        # Handle different response formats
        tts_duration = 0
        if response_format in ['text', 'both']:
            # Send text response
            if socketio:
                socketio.emit('text_response', {
                    'response_text': response_text,
                    'input_text': text_input,
                    'input_method': 'text'
                }, room=sid)
        
        if response_format in ['audio', 'both']:
            # Convert response to audio using TTS
            if socketio:
                socketio.emit('processing_status', {
                    'stage': 'tts',
                    'message': 'Converting response to speech...',
                    'response_text': response_text
                }, room=sid)
            
            tts_start = time.time()
            _stream_tts_audio(sid, response_text, socketio)
            tts_duration = time.time() - tts_start
        
        # Update session statistics
        total_duration = time.time() - start_time
        session['message_count'] += 1
        session['total_audio_duration'] += total_duration  # Track total processing time
        
        logger.info("Text conversation completed", 
                   session_id=sid,
                   conversation_id=conversation_id,
                   total_duration=total_duration,
                   llm_duration=llm_duration,
                   tts_duration=tts_duration,
                   response_format=response_format)
        
        # Send completion status
        if socketio:
            socketio.emit('conversation_completed', {
                'input_text': text_input,
                'response_text': response_text,
                'input_method': 'text',
                'response_format': response_format,
                'processing_time': {
                    'llm': llm_duration,
                    'tts': tts_duration if response_format in ['audio', 'both'] else 0,
                    'total': total_duration
                }
            }, room=sid)
    
    except (MurfAPIError, LLMAPIError) as e:
        logger.error("API error during text conversation processing", 
                    session_id=sid, error=str(e), api_status_code=getattr(e, 'status_code', 'N/A'))
        if socketio:
            socketio.emit('error', {
                'message': f'Service error: {str(e)}. Please try again.'
            }, room=sid)
    
    except Exception as e:
        logger.critical("Unexpected error during text conversation processing", 
                    session_id=sid, error=str(e), exc_info=True)
        if socketio:
            socketio.emit('error', {
                'message': 'An unexpected internal error occurred. Our team has been notified.'
            }, room=sid)
    
    finally:
        # Reset processing flag
        with audio_session_manager.session_locks[sid]:
            session = audio_session_manager.get_session(sid)
            if session:
                session['is_processing'] = False
                logger.debug("Processing flag reset for text session", session_id=sid)


def _check_text_input_rate_limit(sid: str) -> bool:
    """
    Check if the session has exceeded the text input rate limit.
    
    Args:
        sid (str): Session ID.
        
    Returns:
        bool: True if within rate limit, False if exceeded.
    """
    try:
        session = audio_session_manager.get_session(sid)
        if not session:
            return False
        
        current_time = time.time()
        
        # Initialize rate limiting data if not exists
        if 'text_input_timestamps' not in session:
            session['text_input_timestamps'] = []
        
        # Clean old timestamps (older than 1 minute)
        session['text_input_timestamps'] = [
            ts for ts in session['text_input_timestamps'] 
            if current_time - ts < 60
        ]
        
        # Check if rate limit exceeded
        if len(session['text_input_timestamps']) >= config['TEXT_INPUT_RATE_LIMIT']:
            return False
        
        # Add current timestamp
        session['text_input_timestamps'].append(current_time)
        return True
        
    except Exception as e:
        logger.error("Error checking text input rate limit", session_id=sid, error=str(e))
        return True  # Allow on error to avoid blocking legitimate requests


def get_active_sessions() -> List[Dict[str, Any]]:
    """
    Get information about all active audio sessions.
    
    Returns:
        List of active session information.
    """
    try:
        active_sessions = []
        # Iterate over a copy to avoid issues if sessions are cleaned up concurrently
        for session_id, session_data in list(audio_session_manager.sessions.items()):
            active_sessions.append({
                'session_id': session_id,
                'conversation_id': session_data['conversation_id'],
                'created_at': session_data['created_at'],
                'last_activity': session_data['last_activity'],
                'message_count': session_data['message_count'],
                'is_processing': session_data['is_processing'],
                'customer_info': session_data['customer_info'] # Include customer info
            })
        
        logger.debug("Retrieved active sessions list", count=len(active_sessions))
        return active_sessions
    
    except Exception as e:
        logger.error("Error retrieving active sessions list", error=str(e), exc_info=True)
        return []
