"""
Flask Application for AI-Powered Customer Success Agent

This is the main Flask application that serves as the server-side component
for the AI-powered customer success agent with real-time voice communication.

Features:
- Real-time WebSocket communication using Flask-SocketIO
- Audio streaming for voice conversations
- Integration with LLM (Azure GPT-4.1) and Murf TTS/STT services
- Session management with automatic cleanup
- Health monitoring and status endpoints
- Graceful shutdown handling

Dependencies:
- Flask: Web framework
- Flask-SocketIO: WebSocket support
- Flask-CORS: Cross-origin resource sharing
- python-dotenv: Environment variable management
- structlog: Structured logging
- threading: Background task management
- signal: Graceful shutdown handling
"""

import os
import signal
import sys
import atexit
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
from dotenv import load_dotenv
import structlog

# Load environment variables from .env file FIRST (before importing services)
# Ensure this path is correct relative to where app.py is executed
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env'))

# Import configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import get_config

# Load configuration
config = get_config()

# Import our service modules and WebSocket handler functions (after .env is loaded)
from services.llm import LLMAPIError
from services.murf import MurfAPIError
from services.websocket_handler import (
    handle_client_connection,
    handle_client_disconnection,
    handle_incoming_audio_stream,
    handle_text_input,
    handle_get_conversation_history,
    handle_analyze_sentiment,
    handle_generate_summary,
    get_active_sessions,
    audio_session_manager # The global instance of AudioSessionManager
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info, # For detailed exception info
        structlog.dev.ConsoleRenderer() # For development, use ConsoleRenderer
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Initialize logger for this module
logger = structlog.get_logger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Flask configuration from config.py
app.config['SECRET_KEY'] = config['FLASK_SECRET_KEY']
app.config['DEBUG'] = config['FLASK_DEBUG']

# Flask-SocketIO configuration from config.py
socketio_config = {
    'cors_allowed_origins': config['SOCKETIO_CORS_ORIGINS'].split(',') if isinstance(config['SOCKETIO_CORS_ORIGINS'], str) else [config['SOCKETIO_CORS_ORIGINS']],
    'async_mode': config['SOCKETIO_ASYNC_MODE'],
    'logger': app.config['DEBUG'],
    'engineio_logger': app.config['DEBUG'],
    'ping_timeout': config['SOCKETIO_PING_TIMEOUT'],
    'ping_interval': config['SOCKETIO_PING_INTERVAL'],
    'max_http_buffer_size': config['SOCKETIO_MAX_HTTP_BUFFER_SIZE']
}

# Initialize Flask-SocketIO
socketio = SocketIO(app, **socketio_config)

# Initialize Flask-CORS for HTTP endpoints (separate from SocketIO CORS)
CORS(app, origins=config['FLASK_CORS_ORIGINS'].split(',') if isinstance(config['FLASK_CORS_ORIGINS'], str) else [config['FLASK_CORS_ORIGINS']])

# Application state
app_state = {
    'start_time': datetime.now(),
    'total_connections': 0,
    'current_connections': 0,
    'total_conversations': 0, # Incremented only on successful session creation
    'is_shutting_down': False
}

# Global lock for thread-safe operations on app_state
app_state_lock = threading.Lock()


# ===== HTTP Routes =====

@app.route('/')
def index():
    """Root endpoint with basic application information."""
    logger.info("Root endpoint accessed")
    return jsonify({
        'name': config['APP_NAME'],
        'version': config['APP_VERSION'],
        'description': config['APP_DESCRIPTION'],
        'status': 'running' if not app_state['is_shutting_down'] else 'shutting_down',
        'supported_input_methods': config['SUPPORTED_INPUT_METHODS'],
        'default_input_method': config['DEFAULT_INPUT_METHOD'],
        'endpoints': {
            'health': '/health',
            'status': '/status',
            'sessions': '/sessions',
            'websocket': '/socket.io/' # Standard Socket.IO path
        },
        'websocket_events': {
            'input': ['audio_stream', 'text_input'],
            'requests': ['get_conversation_history', 'analyze_sentiment', 'generate_summary', 'ping'],
            'responses': ['text_response', 'audio_chunk', 'conversation_completed', 'processing_status']
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    logger.info("Health check requested")
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': (datetime.now() - app_state['start_time']).total_seconds(),
        'services': {
            'flask': 'healthy',
            'socketio': 'healthy',
            'session_manager': 'healthy',
            'llm_client': 'untested',
            'murf_client': 'untested'
        }
    }
    status_code = 200

    try:
        # Check if app is shutting down
        with app_state_lock:
            if app_state['is_shutting_down']:
                health_status['status'] = 'shutting_down'
                status_code = 503
                logger.warning("Health check: App is shutting down.")
                return jsonify(health_status), status_code

        # Test session manager
        try:
            active_sessions = get_active_sessions()
            health_status['active_sessions'] = len(active_sessions)
            health_status['services']['session_manager'] = 'healthy'
        except Exception as e:
            health_status['services']['session_manager'] = 'unhealthy'
            health_status['session_manager_error'] = str(e)
            health_status['status'] = 'unhealthy'
            status_code = 503
            logger.error("Health check: Session manager unhealthy", error=str(e))

        # Test LLM client (e.g., a simple dummy call)
        try:
            # Assuming LLM client has a simple method to check connectivity, or we can make a tiny prompt
            # For a real health check, you might make a very small, cheap request.
            # Example: audio_session_manager.llm_client.get_llm_response("ping", conversation_history=[])
            # Or a dedicated health check method if LLM API supports it.
            # For now, we'll assume successful initialization implies basic health.
            # A more robust check would be to call a dummy LLM prompt and assert response.
            # For this example, we'll just check if the client object exists.
            if audio_session_manager.llm_client:
                health_status['services']['llm_client'] = 'healthy'
            else:
                raise Exception("LLM client not initialized.")
        except Exception as e:
            health_status['services']['llm_client'] = 'unhealthy'
            health_status['llm_client_error'] = str(e)
            health_status['status'] = 'unhealthy'
            status_code = 503
            logger.error("Health check: LLM client unhealthy", error=str(e))

        # Test Murf client (e.g., a simple dummy call)
        try:
            # Similar to LLM, a dummy call or check for client existence.
            # Example: audio_session_manager.murf_client.get_available_voices(language='en')
            if audio_session_manager.murf_client:
                health_status['services']['murf_client'] = 'healthy'
            else:
                raise Exception("Murf client not initialized.")
        except Exception as e:
            health_status['services']['murf_client'] = 'unhealthy'
            health_status['murf_client_error'] = str(e)
            health_status['status'] = 'unhealthy'
            status_code = 503
            logger.error("Health check: Murf client unhealthy", error=str(e))

        return jsonify(health_status), status_code

    except Exception as e:
        logger.critical("Critical error during health check", error=str(e), exc_info=True)
        return jsonify({
            'status': 'critical_unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/status')
def status():
    """Detailed status endpoint with application metrics."""
    logger.info("Status endpoint requested")
    try:
        active_sessions = get_active_sessions()
        
        with app_state_lock:
            status_data = {
                'application': {
                    'name': config['APP_NAME'],
                    'version': config['APP_VERSION'],
                    'status': 'running' if not app_state['is_shutting_down'] else 'shutting_down',
                    'start_time': app_state['start_time'].isoformat(),
                    'uptime_seconds': (datetime.now() - app_state['start_time']).total_seconds()
                },
                'connections': {
                    'current': app_state['current_connections'],
                    'total': app_state['total_connections'],
                    'active_sessions': len(active_sessions)
                },
                'conversations': {
                    'total': app_state['total_conversations'],
                    'active': len(active_sessions)
                },
                'configuration': {
                    'debug_mode': app.config['DEBUG'],
                    'session_inactivity_timeout_seconds': config['SESSION_INACTIVITY_TIMEOUT_SECONDS'],
                    'session_cleanup_interval_seconds': config['SESSION_CLEANUP_INTERVAL_SECONDS'],
                    'socketio_async_mode': socketio_config['async_mode'],
                    'socketio_ping_timeout': socketio_config['ping_timeout'],
                    'socketio_ping_interval': socketio_config['ping_interval'],
                    'socketio_max_http_buffer_size': socketio_config['max_http_buffer_size'],
                    'text_input_max_length': config['TEXT_INPUT_MAX_LENGTH'],
                    'text_input_rate_limit': config['TEXT_INPUT_RATE_LIMIT'],
                    'audio_input_max_duration': config['AUDIO_INPUT_MAX_DURATION'],
                    'supported_input_methods': config['SUPPORTED_INPUT_METHODS'],
                    'default_input_method': config['DEFAULT_INPUT_METHOD']
                },
                'environment': {
                    'python_version': sys.version,
                    'flask_debug': app.config['DEBUG'],
                }
            }
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error("Status endpoint failed", error=str(e), exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve status',
            'details': str(e)
        }), 500


@app.route('/sessions')
def sessions():
    """Get information about active sessions."""
    logger.info("Sessions endpoint requested")
    try:
        active_sessions = get_active_sessions()
        
        return jsonify({
            'active_sessions': len(active_sessions),
            'sessions': active_sessions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error("Sessions endpoint failed", error=str(e), exc_info=True)
        return jsonify({
            'error': 'Failed to retrieve sessions',
            'details': str(e)
        }), 500


@app.route('/sessions/<session_id>', methods=['DELETE'])
def terminate_session(session_id: str):
    """Terminate a specific session."""
    logger.info("Session termination requested via HTTP", session_id=session_id)
    try:
        session = audio_session_manager.get_session(session_id)
        if not session:
            logger.warning("Attempted to terminate non-existent session", session_id=session_id)
            return jsonify({
                'error': 'Session not found',
                'session_id': session_id
            }), 404
        
        # Notify the client that the session is being terminated
        socketio.emit('server_force_disconnect', {
            'message': 'Your session has been terminated by the administrator.'
        }, room=session_id, namespace='/') # Emit to the specific session ID in the default namespace

        # Disconnect the WebSocket client from the server side
        # This will trigger the on_disconnect handler
        disconnect(sid=session_id, namespace='/')
        
        logger.info("Session termination initiated for client", session_id=session_id)

        # The actual cleanup will happen in handle_client_disconnection,
        # which is called by disconnect().
        # No need to call audio_session_manager.cleanup_session(session_id) directly here
        # as it would be redundant and could cause race conditions if the disconnect
        # event processes first.

        return jsonify({
            'message': 'Session termination initiated successfully',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error("Failed to terminate session via HTTP", session_id=session_id, error=str(e), exc_info=True)
        return jsonify({
            'error': 'Failed to terminate session',
            'details': str(e)
        }), 500


# ===== WebSocket Event Handlers =====

@socketio.on('connect')
def on_connect(auth):
    """Handle WebSocket client connection."""
    client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
    user_agent = request.headers.get('User-Agent', 'unknown')
    
    logger.info("WebSocket connection attempt",
                session_id=request.sid,
                client_ip=client_ip,
                user_agent=user_agent[:100])
    
    try:
        # Handle connection using websocket_handler
        connection_result = handle_client_connection(request.sid, auth)
        
        if connection_result['status'] == 'connected':
            # Update connection stats only on successful connection
            with app_state_lock:
                app_state['total_connections'] += 1
                app_state['current_connections'] += 1
                app_state['total_conversations'] += 1 # Increment total conversations here
            
            logger.info("WebSocket client connected successfully",
                        session_id=request.sid,
                        conversation_id=connection_result.get('conversation_id'))
            
            # Send welcome message
            emit('connection_established', connection_result)
            
            return True  # Accept the connection
        else:
            logger.warning("WebSocket connection rejected",
                          session_id=request.sid,
                          reason=connection_result.get('message', 'Unknown reason'))
            
            emit('connection_rejected', connection_result)
            return False  # Reject the connection
            
    except Exception as e:
        logger.error("Error handling WebSocket connection",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        
        emit('connection_error', {
            'message': 'Internal server error during connection. Please try again.'
        })
        return False


@socketio.on('disconnect')
def on_disconnect():
    """Handle WebSocket client disconnection."""
    logger.info("WebSocket client disconnecting", session_id=request.sid)
    
    try:
        # Update connection stats
        with app_state_lock:
            app_state['current_connections'] = max(0, app_state['current_connections'] - 1)
        
        # Handle disconnection using websocket_handler (will clean up session resources)
        handle_client_disconnection(request.sid)
        
        logger.info("WebSocket client disconnected successfully", session_id=request.sid)
        
    except Exception as e:
        logger.error("Error handling WebSocket disconnection",
                    session_id=request.sid,
                    error=str(e), exc_info=True)


@socketio.on('audio_stream')
def on_audio_stream(data):
    """Handle incoming audio stream from client."""
    logger.debug("Received 'audio_stream' event", session_id=request.sid, data_keys=data.keys())
    try:
        # Basic data validation (more detailed validation is in websocket_handler)
        if not isinstance(data, dict) or 'audio_data' not in data or 'is_final' not in data:
            logger.warning("Invalid or incomplete 'audio_stream' data format", session_id=request.sid, received_data=data)
            emit('error', {'message': 'Invalid audio stream data format. Missing required fields.'})
            return
        
        # Process audio stream using websocket_handler, passing the socketio instance
        # The websocket_handler will handle all subsequent emits for processing status, errors, etc.
        handle_incoming_audio_stream(request.sid, data, socketio)
        
    except Exception as e:
        logger.error("Error handling 'audio_stream' event",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        emit('error', {
            'message': 'Internal error processing audio stream. Please try again.'
        })


@socketio.on('get_conversation_history')
def on_get_conversation_history(data=None):
    """Handle request for conversation history."""
    logger.info("Received 'get_conversation_history' event", session_id=request.sid)
    try:
        handle_get_conversation_history(request.sid, data or {}, socketio)
        
    except Exception as e:
        logger.error("Error handling 'get_conversation_history' request",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        emit('error', {
            'message': 'Internal error retrieving conversation history. Please try again.'
        })


@socketio.on('analyze_sentiment')
def on_analyze_sentiment(data=None):
    """Handle request for sentiment analysis."""
    logger.info("Received 'analyze_sentiment' event", session_id=request.sid)
    try:
        handle_analyze_sentiment(request.sid, data or {}, socketio)
        
    except Exception as e:
        logger.error("Error handling 'analyze_sentiment' request",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        emit('error', {
            'message': 'Internal error analyzing sentiment. Please try again.'
        })


@socketio.on('generate_summary')
def on_generate_summary(data=None):
    """Handle request for conversation summary."""
    logger.info("Received 'generate_summary' event", session_id=request.sid)
    try:
        handle_generate_summary(request.sid, data or {}, socketio)
        
    except Exception as e:
        logger.error("Error handling 'generate_summary' request",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        emit('error', {
            'message': 'Internal error generating summary. Please try again.'
        })


@socketio.on('text_input')
def on_text_input(data):
    """Handle incoming text input from client."""
    logger.info("Received 'text_input' event", session_id=request.sid)
    try:
        # Basic data validation
        if not isinstance(data, dict) or 'text' not in data:
            logger.warning("Invalid 'text_input' data format", session_id=request.sid, received_data=data)
            emit('error', {'message': 'Invalid text input data format. Missing required fields.'})
            return
        
        # Process text input using websocket_handler
        handle_text_input(request.sid, data, socketio)
        
    except Exception as e:
        logger.error("Error handling 'text_input' event",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        emit('error', {
            'message': 'Internal error processing text input. Please try again.'
        })


@socketio.on('ping')
def on_ping(data=None):
    """Handle ping request for connection health check."""
    logger.debug("Received 'ping' event", session_id=request.sid)
    try:
        emit('pong', {
            'timestamp': datetime.now().isoformat(),
            'message': 'Connection is alive'
        })
        
    except Exception as e:
        logger.error("Error handling 'ping' request",
                    session_id=request.sid,
                    error=str(e), exc_info=True)
        emit('error', {
            'message': 'Error responding to ping.'
        })


# ===== Error Handlers =====

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.warning("HTTP 404 Not Found", path=request.path)
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found',
        'code': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error (HTTP 500)", error=str(error), exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred',
        'code': 500
    }), 500


@socketio.on_error_default
def default_error_handler(e):
    """Handle unhandled WebSocket errors."""
    logger.error("Unhandled WebSocket error caught by default handler",
                session_id=request.sid,
                error=str(e), exc_info=True)
    
    emit('error', {
        'message': 'A WebSocket error occurred. Please try again.',
        'type': 'websocket_error'
    })


# ===== Application Lifecycle Management =====

def start_background_services():
    """Start background services and threads."""
    try:
        # Start the session cleanup thread from the audio_session_manager
        audio_session_manager.start_background_cleanup_thread()
        logger.info("Background services started successfully")
        
    except Exception as e:
        logger.critical("Failed to start background services", error=str(e), exc_info=True)
        # Re-raise to prevent the app from starting if critical services fail
        raise


def stop_background_services():
    """Stop background services and threads."""
    try:
        logger.info("Stopping background services...")
        
        # Signal the session cleanup thread to stop
        audio_session_manager.stop_background_cleanup_thread()
        
        # Mark as shutting down
        with app_state_lock:
            app_state['is_shutting_down'] = True
        
        logger.info("Background services stopped successfully")
        
    except Exception as e:
        logger.error("Error stopping background services", error=str(e), exc_info=True)


def graceful_shutdown(signum, frame):
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    logger.info("Received shutdown signal", signal=signum)
    
    try:
        # Stop background services
        stop_background_services()
        
        # Clean up all active sessions
        active_sessions_to_cleanup = get_active_sessions() # Get a list of sessions
        logger.info("Initiating cleanup of active sessions during shutdown", count=len(active_sessions_to_cleanup))
        
        for session_info in active_sessions_to_cleanup:
            session_id = session_info['session_id']
            try:
                # Notify client of shutdown (optional, as disconnect might be immediate)
                socketio.emit('server_shutdown', {
                    'message': 'Server is shutting down. Please reconnect later.'
                }, room=session_id, namespace='/')
                
                # Disconnect the WebSocket client, which will trigger handle_client_disconnection
                # and clean up the session in audio_session_manager.
                disconnect(sid=session_id, namespace='/')
                logger.info("Disconnected client during shutdown", session_id=session_id)
                
            except Exception as e:
                logger.warning("Error disconnecting client or cleaning up session during shutdown",
                              session_id=session_id,
                              error=str(e), exc_info=True)
        
        logger.info("Graceful shutdown completed")
        
    except Exception as e:
        logger.critical("Critical error during graceful shutdown process", error=str(e), exc_info=True)
    
    finally:
        # Exit the application after cleanup
        sys.exit(0)


# Register cleanup functions for various shutdown scenarios
atexit.register(stop_background_services) # For normal program exit
signal.signal(signal.SIGTERM, graceful_shutdown) # For systemd/docker stop
signal.signal(signal.SIGINT, graceful_shutdown) # For Ctrl+C


# ===== Application Entry Point =====

if __name__ == '__main__':
    try:
        # Log application startup
        logger.info("Starting AI Customer Success Agent server",
                    debug=app.config['DEBUG'],
                    host=config['FLASK_HOST'],
                    port=config['FLASK_PORT'])
        
        # Start background services (like session cleanup)
        start_background_services()
        
        # Run the Flask-SocketIO application
        # use_reloader=False is crucial when using threading for background tasks
        # log_output=app.config['DEBUG'] controls logging from Werkzeug/Engine.IO
        socketio.run(
            app,
            host=config['FLASK_HOST'],
            port=config['FLASK_PORT'],
            debug=app.config['DEBUG'],
            use_reloader=False,
            log_output=app.config['DEBUG']
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        # Call graceful shutdown directly on KeyboardInterrupt
        graceful_shutdown(signal.SIGINT, None)
        
    except Exception as e:
        logger.critical("Failed to start application due to unhandled error", error=str(e), exc_info=True)
        # Ensure background services are stopped even if startup fails
        stop_background_services()
        sys.exit(1)

