#!/usr/bin/env python3
"""
Interactive CLI Tool for AI Customer Success Agent Backend Testing

This interactive CLI provides a beautiful menu-driven interface for testing:
- Service Testing: LLM, Murf API, WebSocket handlers
- HTTP Endpoint Testing: Health checks, status, sessions
- WebSocket Testing: Real-time communication testing
- Configuration Testing: Config validation and environment setup
- Integration Testing: End-to-end workflow testing
- Performance Testing: Load testing and benchmarking
- Error Handling Testing: Fault injection and recovery testing

Usage:
    python cli.py
"""

import base64
import json
import os
import sys
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
import warnings
import wave

# Third-party imports
try:
    import requests
    import socketio
    import structlog
    from dotenv import load_dotenv
    import numpy as np
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages:")
    print("pip install requests python-socketio structlog python-dotenv numpy")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import backend modules
from config import get_config
from services.llm import create_llm_service, LLMAPIError, ConversationContext
from services.murf import create_murf_client, MurfAPIError
from services.whisper import create_whisper_client, WhisperAPIError
from services.websocket_handler import (
    AudioSessionManager, 
    handle_client_connection,
    handle_client_disconnection,
    handle_incoming_audio_stream,
    handle_text_input,
    handle_get_conversation_history,
    handle_analyze_sentiment,
    handle_generate_summary,
    get_active_sessions
)

# Configure structured logging for CLI
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global configuration
config = get_config()

# CLI State
cli_state = {
    'test_results': {},
    'start_time': None,
    'total_tests': 0,
    'passed_tests': 0,
    'failed_tests': 0,
    'server_url': f"http://localhost:{config['FLASK_PORT']}",
    'current_session': None,
    'llm_client': None,
    'murf_client': None,
    'session_manager': None
}

# ANSI Color codes for beautiful output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the application banner"""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              AI Customer Success Agent Backend Testing CLI                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}

{Colors.BLUE}🚀 Welcome to the comprehensive testing suite for your AI Customer Success Agent!{Colors.RESET}
{Colors.BLUE}📊 Server URL: {Colors.YELLOW}{cli_state['server_url']}{Colors.RESET}
{Colors.BLUE}🕐 Session started: {Colors.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}
"""
    print(banner)

def print_header(title: str, color: str = Colors.CYAN):
    """Print a formatted header"""
    print(f"\n{color}{Colors.BOLD}")
    print("=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80)
    print(f"{Colors.RESET}")

def print_section(title: str, icon: str = "🔧"):
    """Print a formatted section title"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{icon} {title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'─' * (len(title) + 4)}{Colors.RESET}")

def print_success(message: str, duration: float = 0.0):
    """Print success message"""
    if duration > 0:
        print(f"{Colors.GREEN}✅ {message} {Colors.CYAN}({duration:.2f}s){Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def print_error(message: str, duration: float = 0.0):
    """Print error message"""
    if duration > 0:
        print(f"{Colors.RED}❌ {message} {Colors.CYAN}({duration:.2f}s){Colors.RESET}")
    else:
        print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")

def print_menu_option(number: int, title: str, description: str = ""):
    """Print a menu option"""
    if description:
        print(f"{Colors.CYAN}[{number}]{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET} - {Colors.YELLOW}{description}{Colors.RESET}")
    else:
        print(f"{Colors.CYAN}[{number}]{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET}")

def wait_for_input(prompt: str = "Press Enter to continue..."):
    """Wait for user input"""
    print(f"\n{Colors.YELLOW}{prompt}{Colors.RESET}")
    input()

def get_user_choice(prompt: str = "Enter your choice: ") -> str:
    """Get user input choice"""
    print(f"\n{Colors.CYAN}{prompt}{Colors.RESET}", end="")
    return input().strip()

def record_test_result(test_name: str, success: bool, message: str = "", duration: float = 0.0):
    """Record test result for reporting"""
    cli_state['total_tests'] += 1
    if success:
        cli_state['passed_tests'] += 1
        print_success(f"{test_name}: {message}", duration)
    else:
        cli_state['failed_tests'] += 1
        print_error(f"{test_name}: {message}", duration)
    
    cli_state['test_results'][test_name] = {
        'success': success,
        'message': message,
        'duration': duration,
        'timestamp': datetime.now().isoformat()
    }

def generate_test_audio(format_type: str = "wav", duration: float = 1.0) -> bytes:
    """Generate test audio data"""
    try:
        sample_rate = 22050
        samples = int(sample_rate * duration)
        frequency = 440  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, samples, False)
        wave_data = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Convert to 16-bit PCM
        wave_data = (wave_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(wave_data.tobytes())
        
        return buffer.getvalue()
    except Exception as e:
        print_warning(f"Could not generate audio: {e}")
        # Fallback: Generate simple binary data for testing
        return b'RIFF\x24\x08\x00\x00WAVE' + b'\x00' * 1000

def initialize_services():
    """Initialize all backend services"""
    print_section("Initializing Backend Services", "🔧")
    
    # Initialize LLM Client
    try:
        cli_state['llm_client'] = create_llm_service()
        print_success("LLM Client initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize LLM Client: {str(e)}")
    
    # Initialize Murf Client
    try:
        cli_state['murf_client'] = create_murf_client()
        print_success("Murf Client initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize Murf Client: {str(e)}")
    
    # Initialize Whisper Client
    try:
        cli_state['whisper_client'] = create_whisper_client()
        print_success("Whisper STT Client initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize Whisper Client: {str(e)}")
    
    # Initialize Session Manager
    try:
        cli_state['session_manager'] = AudioSessionManager()
        print_success("Session Manager initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize Session Manager: {str(e)}")

def test_llm_service():
    """Test OpenAI GPT Service"""
    print_section("Testing OpenAI GPT Service", "🧠")
    
    if not cli_state['llm_client']:
        print_error("OpenAI GPT Client not initialized")
        return
    
    # Test conversation creation
    print_info("Creating conversation...")
    start_time = time.time()
    try:
        customer_info = {
            'name': 'Test User',
            'email': 'test@example.com',
            'plan': 'premium'
        }
        conversation_id = cli_state['llm_client'].create_conversation(customer_info)
        duration = time.time() - start_time
        print_success(f"OpenAI GPT Create Conversation: Conversation ID: {conversation_id[:12]}...", duration)
    except Exception as e:
        print_error(f"OpenAI GPT Create Conversation: {str(e)}")
        return
    
    # Test 2: Chat completion
    print_info("Testing chat completion...")
    start_time = time.time()
    try:
        response = cli_state['llm_client'].chat_completion(
            message="Hello, I need help with my account.",
            conversation_id=conversation_id
        )
        duration = time.time() - start_time
        response_preview = response['response'][:50] + "..." if len(response['response']) > 50 else response['response']
        record_test_result("LLM Chat Completion", True, f"Response: {response_preview}", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("LLM Chat Completion", False, f"Error: {str(e)}", duration)
    
    # Test 3: Streaming chat completion
    print_info("Testing streaming chat completion...")
    start_time = time.time()
    try:
        full_response = ""
        chunk_count = 0
        for chunk in cli_state['llm_client'].chat_completion_streaming(
            message="Can you explain how to reset my password?",
            conversation_id=conversation_id
        ):
            if chunk.get('content'):
                full_response += chunk['content']
                chunk_count += 1
        duration = time.time() - start_time
        record_test_result("LLM Streaming Chat", True, f"Received {chunk_count} chunks", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("LLM Streaming Chat", False, f"Error: {str(e)}", duration)

def test_murf_service():
    """Test Murf TTS service functionality"""
    print_section("Testing Murf TTS Service", "🎤")
    
    if not cli_state['murf_client']:
        print_error("Murf Client not initialized")
        return
    
    # Test: Text-to-Speech
    print_info("Testing Text-to-Speech...")
    start_time = time.time()
    try:
        tts_result = cli_state['murf_client'].text_to_speech(
            text="Hello, this is a test message for the AI Customer Success Agent.",
            voice_id=config['MURF_DEFAULT_VOICE_ID']
        )
        duration = time.time() - start_time
        audio_size = len(tts_result.get('audio_data', b''))
        record_test_result("Murf Text-to-Speech", True, f"Generated {audio_size} bytes of audio", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Murf Text-to-Speech", False, f"Error: {str(e)}", duration)
    
    # Note: STT functionality has been moved to Whisper service
    print_info("Note: Speech-to-Text functionality is now handled by Whisper service")

def test_whisper_service():
    """Test Whisper STT service functionality"""
    print_section("Testing Whisper STT Service", "🎙️")
    
    if not cli_state['whisper_client']:
        print_error("Whisper Client not initialized")
        return
    
    # Test 1: Speech-to-Text with generated audio
    print_info("Testing Speech-to-Text with generated audio...")
    start_time = time.time()
    try:
        test_audio = generate_test_audio()
        stt_result = cli_state['whisper_client'].transcribe_audio(
            audio_data=test_audio,
            audio_format='wav',
            language='en'
        )
        duration = time.time() - start_time
        transcribed_text = stt_result.get('text', '')
        confidence = stt_result.get('confidence', 0.0)
        record_test_result("Whisper Speech-to-Text", True, 
                          f"Transcribed: '{transcribed_text}' (Confidence: {confidence:.2f})", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Whisper Speech-to-Text", False, f"Error: {str(e)}", duration)
    
    # Test 2: Test format support
    print_info("Testing supported formats...")
    start_time = time.time()
    try:
        supported_formats = cli_state['whisper_client'].get_supported_formats()
        max_file_size = cli_state['whisper_client'].get_max_file_size()
        duration = time.time() - start_time
        record_test_result("Whisper Format Support", True, 
                          f"Supports: {', '.join(supported_formats)} | Max size: {max_file_size/1024/1024:.1f}MB", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Whisper Format Support", False, f"Error: {str(e)}", duration)
    
    # Test 3: Test audio validation
    print_info("Testing audio validation...")
    start_time = time.time()
    try:
        # Test with invalid format
        try:
            cli_state['whisper_client'].transcribe_audio(b'invalid_audio', 'invalid_format')
            record_test_result("Whisper Audio Validation", False, "Should have failed with invalid format", time.time() - start_time)
        except WhisperAPIError as e:
            if "not supported" in str(e):
                record_test_result("Whisper Audio Validation", True, "Correctly rejected invalid format", time.time() - start_time)
            else:
                record_test_result("Whisper Audio Validation", False, f"Unexpected error: {str(e)}", time.time() - start_time)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Whisper Audio Validation", False, f"Error: {str(e)}", duration)

def test_session_manager():
    """Test session manager functionality"""
    print_section("Testing Session Manager", "📋")
    
    if not cli_state['session_manager']:
        print_error("Session Manager not initialized")
        return
    
    session_id = str(uuid.uuid4())
    
    # Test 1: Create session
    print_info("Creating session...")
    start_time = time.time()
    try:
        customer_info = {'name': 'Test Customer', 'email': 'test@example.com'}
        conversation_id = cli_state['session_manager'].create_session(session_id, customer_info)
        duration = time.time() - start_time
        record_test_result("Session Creation", True, f"Session created successfully", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Session Creation", False, f"Error: {str(e)}", duration)
        return
    
    # Test 2: Get session
    print_info("Retrieving session...")
    start_time = time.time()
    try:
        session = cli_state['session_manager'].get_session(session_id)
        duration = time.time() - start_time
        record_test_result("Session Retrieval", True, "Session retrieved successfully", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Session Retrieval", False, f"Error: {str(e)}", duration)
    
    # Test 3: Cleanup session
    print_info("Cleaning up session...")
    start_time = time.time()
    try:
        cli_state['session_manager'].cleanup_session(session_id)
        duration = time.time() - start_time
        record_test_result("Session Cleanup", True, "Session cleaned up successfully", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Session Cleanup", False, f"Error: {str(e)}", duration)

def test_http_endpoints():
    """Test HTTP endpoints"""
    print_section("Testing HTTP Endpoints", "🌐")
    
    session = requests.Session()
    timeout = 30
    
    endpoints = [
        ('Root Endpoint', '/', 'GET'),
        ('Health Check', '/health', 'GET'),
        ('Status Endpoint', '/status', 'GET'),
        ('Sessions Endpoint', '/sessions', 'GET'),
        ('404 Test', '/nonexistent', 'GET')
    ]
    
    for name, endpoint, method in endpoints:
        print_info(f"Testing {name}...")
        start_time = time.time()
        try:
            response = session.request(method, f"{cli_state['server_url']}{endpoint}", timeout=timeout)
            duration = time.time() - start_time
            
            if endpoint == '/nonexistent':
                if response.status_code == 404:
                    record_test_result(name, True, "404 returned correctly", duration)
                else:
                    record_test_result(name, False, f"Expected 404, got {response.status_code}", duration)
            else:
                if response.status_code == 200:
                    record_test_result(name, True, f"Status: {response.status_code}", duration)
                else:
                    record_test_result(name, False, f"Status: {response.status_code}", duration)
        except Exception as e:
            duration = time.time() - start_time
            record_test_result(name, False, f"Error: {str(e)}", duration)

def test_websocket_connection():
    """Test WebSocket functionality"""
    print_section("Testing WebSocket Connection", "🔌")
    
    try:
        client = socketio.Client()
        events_received = {}
        connection_established = False
        
        @client.event
        def connect():
            nonlocal connection_established
            connection_established = True
            print_info("WebSocket connected")
        
        @client.event
        def disconnect():
            nonlocal connection_established
            connection_established = False
            print_info("WebSocket disconnected")
        
        @client.event
        def text_response(data):
            events_received['text_response'] = data
            print_info(f"Received text response: {data.get('response_text', '')[:50]}...")
        
        @client.event
        def error(data):
            events_received['error'] = data
            print_warning(f"Received error: {data}")
        
        # Test 1: Connect
        print_info("Connecting to WebSocket...")
        start_time = time.time()
        try:
            client.connect(cli_state['server_url'])
            time.sleep(2)  # Give connection time to establish
            duration = time.time() - start_time
            if connection_established:
                record_test_result("WebSocket Connection", True, "Connected successfully", duration)
            else:
                record_test_result("WebSocket Connection", False, "Connection not established", duration)
                return
        except Exception as e:
            duration = time.time() - start_time
            record_test_result("WebSocket Connection", False, f"Error: {str(e)}", duration)
            return
        
        # Test 2: Text input
        print_info("Testing text input...")
        start_time = time.time()
        try:
            client.emit('text_input', {
                'text': 'Hello, I need help with my account.',
                'response_format': 'text'
            })
            time.sleep(5)  # Wait for response
            duration = time.time() - start_time
            if 'text_response' in events_received:
                record_test_result("Text Input", True, "Response received", duration)
            else:
                record_test_result("Text Input", False, "No response received", duration)
        except Exception as e:
            duration = time.time() - start_time
            record_test_result("Text Input", False, f"Error: {str(e)}", duration)
        
        # Test 3: Ping test
        print_info("Testing ping...")
        start_time = time.time()
        try:
            client.emit('ping', {'message': 'test ping'})
            time.sleep(2)
            duration = time.time() - start_time
            record_test_result("WebSocket Ping", True, "Ping sent successfully", duration)
        except Exception as e:
            duration = time.time() - start_time
            record_test_result("WebSocket Ping", False, f"Error: {str(e)}", duration)
        
        # Cleanup
        try:
            client.disconnect()
        except:
            pass
            
    except Exception as e:
        print_error(f"WebSocket test failed: {str(e)}")

def test_configuration():
    """Test configuration validation"""
    print_section("Testing Configuration", "⚙️")
    
    # Test 1: Load configuration
    print_info("Testing configuration loading...")
    start_time = time.time()
    try:
        config = get_config()
        duration = time.time() - start_time
        if config and isinstance(config, dict):
            record_test_result("Configuration Loading", True, f"Loaded {len(config)} config items", duration)
        else:
            record_test_result("Configuration Loading", False, "Config not loaded or invalid", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Configuration Loading", False, f"Error: {str(e)}", duration)
    
    # Test 2: Environment variables
    print_info("Checking environment variables...")
    start_time = time.time()
    try:
        required_vars = ['MURF_API_KEY', 'GITHUB_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        duration = time.time() - start_time
        if not missing_vars:
            record_test_result("Environment Variables", True, "All required vars present", duration)
        else:
            record_test_result("Environment Variables", False, f"Missing: {', '.join(missing_vars)}", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Environment Variables", False, f"Error: {str(e)}", duration)

def run_performance_test():
    """Run performance tests"""
    print_section("Running Performance Tests", "⚡")
    
    num_requests = 10
    num_threads = 5
    
    print_info(f"Running {num_requests} requests with {num_threads} threads...")
    
    start_time = time.time()
    try:
        def make_request():
            response = requests.get(f"{cli_state['server_url']}/health", timeout=30)
            return response.status_code == 200, response.elapsed.total_seconds()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        duration = time.time() - start_time
        successful = sum(1 for success, _ in results if success)
        avg_response_time = sum(rt for _, rt in results) / len(results)
        
        record_test_result("HTTP Performance", True, 
                         f"{successful}/{num_requests} successful, avg: {avg_response_time:.3f}s", 
                         duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("HTTP Performance", False, f"Error: {str(e)}", duration)

def print_test_summary():
    """Print comprehensive test summary"""
    print_header("📊 TEST SUMMARY", Colors.MAGENTA)
    
    total_time = time.time() - cli_state['start_time'] if cli_state['start_time'] else 0
    success_rate = (cli_state['passed_tests']/cli_state['total_tests']*100) if cli_state['total_tests'] > 0 else 0
    
    print(f"{Colors.BOLD}Total Tests:{Colors.RESET} {cli_state['total_tests']}")
    print(f"{Colors.GREEN}✅ Passed:{Colors.RESET} {cli_state['passed_tests']}")
    print(f"{Colors.RED}❌ Failed:{Colors.RESET} {cli_state['failed_tests']}")
    print(f"{Colors.BLUE}📈 Success Rate:{Colors.RESET} {success_rate:.1f}%")
    print(f"{Colors.BLUE}⏱️  Total Time:{Colors.RESET} {total_time:.2f}s")
    
    if cli_state['failed_tests'] > 0:
        print(f"\n{Colors.RED}❌ Failed Tests:{Colors.RESET}")
        for test_name, result in cli_state['test_results'].items():
            if not result['success']:
                print(f"  • {test_name}: {result['message']}")
    
    print(f"\n{Colors.CYAN}🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")

def test_handoff_analysis():
    """Test handoff analysis functionality"""
    print_section("Testing Handoff Analysis", "🤝")
    
    if not cli_state['llm_client']:
        print_error("LLM Client not initialized")
        return
    
    # Create a conversation for testing
    print_info("Creating conversation for handoff testing...")
    start_time = time.time()
    try:
        customer_info = {
            'name': 'Test Customer',
            'email': 'test@example.com',
            'plan': 'Premium',
            'issue_type': 'Support Request'
        }
        conversation_id = cli_state['llm_client'].create_conversation(customer_info=customer_info)
        duration = time.time() - start_time
        record_test_result("Handoff Conversation Setup", True, f"Conversation ID: {conversation_id[:8]}...", duration)
    except Exception as e:
        duration = time.time() - start_time
        record_test_result("Handoff Conversation Setup", False, f"Error: {str(e)}", duration)
        return
    
    # Test scenarios that should trigger handoff
    handoff_test_cases = [
        {
            'message': "I want a refund for my subscription, this is ridiculous!",
            'expected_handoff': True,
            'description': "Refund Request"
        },
        {
            'message': "I can't access my account and I've tried everything. This is urgent!",
            'expected_handoff': True,
            'description': "Account Access Issue"
        },
        {
            'message': "What are your business hours?",
            'expected_handoff': False,
            'description': "General Inquiry"
        },
        {
            'message': "I need to speak to a manager immediately. This service is terrible!",
            'expected_handoff': True,
            'description': "Escalation Request"
        }
    ]
    
    for i, test_case in enumerate(handoff_test_cases, 1):
        print_info(f"Testing handoff scenario {i}: {test_case['description']}")
        start_time = time.time()
        
        try:
            # Add some conversation context first
            cli_state['llm_client'].chat_completion(
                message="Hello, I need some help with my account.",
                conversation_id=conversation_id
            )
            
            # Now test the handoff analysis
            handoff_result = cli_state['llm_client'].analyze_handoff_need(
                conversation_id=conversation_id,
                user_message=test_case['message']
            )
            
            duration = time.time() - start_time
            
            # Validate the response structure
            required_keys = ['needs_handoff', 'confidence', 'reason', 'category', 'urgency', 'suggested_response']
            if not all(k in handoff_result for k in required_keys):
                record_test_result(f"Handoff Analysis {i}", False, "Missing required keys in response", duration)
                continue
            
            needs_handoff = handoff_result['needs_handoff']
            confidence = handoff_result['confidence']
            category = handoff_result['category']
            urgency = handoff_result['urgency']
            
            # Check if the analysis matches expectations
            if needs_handoff == test_case['expected_handoff']:
                record_test_result(
                    f"Handoff Analysis {i}", 
                    True, 
                    f"✓ {test_case['description']}: handoff={needs_handoff}, confidence={confidence:.2f}, category={category}, urgency={urgency}", 
                    duration
                )
            else:
                record_test_result(
                    f"Handoff Analysis {i}", 
                    False, 
                    f"✗ Expected handoff={test_case['expected_handoff']}, got {needs_handoff} (confidence={confidence:.2f})", 
                    duration
                )
            
            # Print detailed results
            print(f"    📋 Analysis Details:")
            print(f"       Needs Handoff: {needs_handoff}")
            print(f"       Confidence: {confidence:.2f}")
            print(f"       Reason: {handoff_result['reason']}")
            print(f"       Category: {category}")
            print(f"       Urgency: {urgency}")
            print(f"       Suggested Response: {handoff_result['suggested_response'][:80]}...")
            
        except Exception as e:
            duration = time.time() - start_time
            record_test_result(f"Handoff Analysis {i}", False, f"Error: {str(e)}", duration)

def service_testing_menu():
    """Service testing submenu"""
    while True:
        clear_screen()
        print_header("🔧 SERVICE TESTING", Colors.BLUE)
        
        print_menu_option(1, "Test LLM Service", "Azure GPT-4.1 API testing")
        print_menu_option(2, "Test Murf TTS Service", "Text-to-Speech API testing")
        print_menu_option(3, "Test Whisper Service", "STT API testing")
        print_menu_option(4, "Test Session Manager", "Session management testing")
        print_menu_option(5, "Test Handoff Analysis", "Human handoff detection testing")
        print_menu_option(6, "Test All Services", "Run all service tests")
        print_menu_option(0, "Back to Main Menu", "Return to main menu")
        
        choice = get_user_choice()
        
        if choice == '1':
            clear_screen()
            test_llm_service()
            wait_for_input()
        elif choice == '2':
            clear_screen()
            test_murf_service()
            wait_for_input()
        elif choice == '3':
            clear_screen()
            test_whisper_service()
            wait_for_input()
        elif choice == '4':
            clear_screen()
            test_session_manager()
            wait_for_input()
        elif choice == '5':
            clear_screen()
            test_handoff_analysis()
            wait_for_input()
        elif choice == '6':
            clear_screen()
            test_llm_service()
            test_murf_service()
            test_whisper_service()
            test_session_manager()
            test_handoff_analysis()
            wait_for_input()
        elif choice == '0':
            break
        else:
            print_warning("Invalid choice. Please try again.")
            time.sleep(1)

def network_testing_menu():
    """Network testing submenu"""
    while True:
        clear_screen()
        print_header("🌐 NETWORK TESTING", Colors.GREEN)
        
        print_menu_option(1, "Test HTTP Endpoints", "REST API endpoint testing")
        print_menu_option(2, "Test WebSocket Connection", "Real-time communication testing")
        print_menu_option(3, "Test Both", "Run all network tests")
        print_menu_option(0, "Back to Main Menu", "Return to main menu")
        
        choice = get_user_choice()
        
        if choice == '1':
            clear_screen()
            test_http_endpoints()
            wait_for_input()
        elif choice == '2':
            clear_screen()
            test_websocket_connection()
            wait_for_input()
        elif choice == '3':
            clear_screen()
            test_http_endpoints()
            test_websocket_connection()
            wait_for_input()
        elif choice == '0':
            break
        else:
            print_warning("Invalid choice. Please try again.")
            time.sleep(1)

def main_menu():
    """Main menu loop"""
    while True:
        clear_screen()
        print_banner()
        print_header("🎯 MAIN MENU", Colors.CYAN)
        
        print_menu_option(1, "Service Testing", "Test LLM, Murf, and Session Manager")
        print_menu_option(2, "Network Testing", "Test HTTP endpoints and WebSocket")
        print_menu_option(3, "Configuration Testing", "Validate config and environment")
        print_menu_option(4, "Performance Testing", "Load testing and benchmarking")
        print_menu_option(5, "Run All Tests", "Execute complete test suite")
        print_menu_option(6, "View Test Results", "Show current test summary")
        print_menu_option(7, "Reset Test Results", "Clear all test results")
        print_menu_option(8, "Settings", "Configure server URL and options")
        print_menu_option(0, "Exit", "Close the testing application")
        
        choice = get_user_choice()
        
        if choice == '1':
            service_testing_menu()
        elif choice == '2':
            network_testing_menu()
        elif choice == '3':
            clear_screen()
            test_configuration()
            wait_for_input()
        elif choice == '4':
            clear_screen()
            run_performance_test()
            wait_for_input()
        elif choice == '5':
            clear_screen()
            print_info("Running complete test suite...")
            cli_state['start_time'] = time.time()
            
            # Run all tests
            test_configuration()
            test_llm_service()
            test_murf_service()
            test_session_manager()
            test_http_endpoints()
            test_websocket_connection()
            run_performance_test()
            
            print_test_summary()
            wait_for_input()
        elif choice == '6':
            clear_screen()
            print_test_summary()
            wait_for_input()
        elif choice == '7':
            cli_state['test_results'] = {}
            cli_state['total_tests'] = 0
            cli_state['passed_tests'] = 0
            cli_state['failed_tests'] = 0
            print_success("Test results cleared!")
            time.sleep(1)
        elif choice == '8':
            clear_screen()
            print_header("⚙️ SETTINGS", Colors.YELLOW)
            print(f"Current server URL: {Colors.CYAN}{cli_state['server_url']}{Colors.RESET}")
            new_url = get_user_choice("Enter new server URL (or press Enter to keep current): ")
            if new_url:
                cli_state['server_url'] = new_url
                print_success(f"Server URL updated to: {new_url}")
            wait_for_input()
        elif choice == '0':
            clear_screen()
            print_header("👋 GOODBYE", Colors.MAGENTA)
            print(f"{Colors.CYAN}Thank you for using the AI Customer Success Agent Testing CLI!{Colors.RESET}")
            if cli_state['total_tests'] > 0:
                print(f"{Colors.YELLOW}Final Results: {cli_state['passed_tests']}/{cli_state['total_tests']} tests passed{Colors.RESET}")
            print(f"{Colors.GREEN}Happy testing! 🚀{Colors.RESET}")
            break
        else:
            print_warning("Invalid choice. Please try again.")
            time.sleep(1)

def main():
    """Main application function"""
    try:
        clear_screen()
        print_banner()
        print_section("Initializing System", "🔧")
        
        # Initialize services
        initialize_services()
        
        print_success("System initialized successfully!")
        print_info("Starting interactive testing interface...")
        time.sleep(2)
        
        # Start main menu loop
        main_menu()
        
    except KeyboardInterrupt:
        clear_screen()
        print_header("⚠️ INTERRUPTED", Colors.YELLOW)
        print(f"{Colors.YELLOW}Testing interrupted by user{Colors.RESET}")
        print(f"{Colors.CYAN}Goodbye! 👋{Colors.RESET}")
    except Exception as e:
        clear_screen()
        print_header("❌ ERROR", Colors.RED)
        print_error(f"Unexpected error: {str(e)}")
        print_warning("Please check your configuration and try again.")
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    main() 