import io from 'socket.io-client';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.callbacks = new Map();
    this.SOCKET_URL = 'http://localhost:5000';
  }

  connect() {
    if (this.socket && this.socket.connected) {
      return;
    }

    this.socket = io(this.SOCKET_URL, {
      transports: ['websocket', 'polling'], // Add polling as fallback
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      timeout: 20000, // Add timeout
    });

    this.socket.on('connect', () => {
      console.log('WebSocket Connected!');
      this._executeCallbacks('connect', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket Disconnected!', reason);
      this._executeCallbacks('disconnect', { connected: false, reason });
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket Connection Error:', error);
      this._executeCallbacks('error', { message: 'Connection failed', error });
    });

    this.socket.on('error', (error) => {
      console.log('WebSocket Error:', error);
      this._executeCallbacks('error', error);
    });

    // Register all the event listeners
    this.socket.on('processing_status', (data) => {
      this._executeCallbacks('processing_status', data);
    });

    this.socket.on('conversation_completed', (data) => {
      this._executeCallbacks('conversation_completed', data);
    });

    this.socket.on('audio_stream_start', (data) => {
      this._executeCallbacks('audio_stream_start', data);
    });

    this.socket.on('audio_chunk', (data) => {
      this._executeCallbacks('audio_chunk', data);
    });

    this.socket.on('audio_stream_complete', (data) => {
      this._executeCallbacks('audio_stream_complete', data);
    });

    this.socket.on('handoff_suggestion', (data) => {
      this._executeCallbacks('handoff_suggestion', data);
    });

    this.socket.on('human_handoff_initiated', (data) => {
      this._executeCallbacks('human_handoff_initiated', data);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.callbacks.clear();
  }

  on(event, callback) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, []);
    }
    this.callbacks.get(event).push(callback);
  }

  off(event, callback = null) {
    if (callback) {
      const callbacks = this.callbacks.get(event) || [];
      this.callbacks.set(event, callbacks.filter(cb => cb !== callback));
    } else {
      this.callbacks.delete(event);
    }
  }

  _executeCallbacks(event, data) {
    const callbacks = this.callbacks.get(event) || [];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error executing callback for event ${event}:`, error);
      }
    });
  }

  emit(event, data) {
    if (this.socket && this.socket.connected) {
      console.log(`Emitted ${event}:`, data);
      this.socket.emit(event, data);
    } else {
      console.error('Socket not connected. Cannot emit event:', event);
      this._executeCallbacks('error', { message: 'Socket not connected' });
    }
  }

  sendTextInput(text, responseFormat = 'text') {
    this.emit('text_input', {
      text: text,
      response_format: responseFormat
    });
  }

  sendAudioStream(audioBlob, responseFormat = 'audio') {
    const reader = new FileReader();
    
    reader.onloadend = () => {
      try {
        const base64Audio = reader.result.split(',')[1];
        
        // Ensure response format is correctly passed
        const audioData = {
          audio_data: base64Audio,
          is_final: true,
          response_format: responseFormat, // Make sure this is 'audio' not 'text'
          format: 'webm'
        };

        console.log('Sending audio_stream with response_format:', responseFormat);
        console.log('Audio data keys:', Object.keys(audioData));
        
        this.emit('audio_stream', audioData);

      } catch (error) {
        console.error('Error processing audio blob:', error);
        this._executeCallbacks('error', { message: 'Failed to process audio data', error: error.message });
      }
    };

    reader.readAsDataURL(audioBlob);
  }
}

// Export the class directly
export default WebSocketService; 