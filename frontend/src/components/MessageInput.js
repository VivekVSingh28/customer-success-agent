import React, { useState, useCallback, useEffect } from 'react';
import { useMicrophone } from '../hooks/useMicrophone';
import './MessageInput.css';

const MessageInput = ({ 
  onSendText, 
  onSendAudio, 
  inputMode = 'text', 
  onInputModeChange,
  currentResponseFormat = 'text',
  onResponseFormatChange,
  isProcessing = false,
  messages = [] // Add default empty array
}) => {
  const [textInput, setTextInput] = useState('');
  const [recordingState, setRecordingState] = useState('idle');
  const [recordingTime, setRecordingTime] = useState(0);

  const {
    startRecording,
    stopRecording,
    isRecording,
    recordingDuration,
    audioLevel
  } = useMicrophone();

  const handleVoiceRecord = useCallback(async () => {
    if (recordingState === 'idle') {
      setRecordingState('recording');
      setRecordingTime(0);
      await startRecording();
      console.log('Recording started');
    } else if (recordingState === 'recording') {
      const audioBlob = await stopRecording();
      setRecordingState('idle');
      setRecordingTime(0);
      
      if (audioBlob && audioBlob.size > 0) {
        if (audioBlob.size < 5000) {
          alert('No speech detected. Please try recording again and speak clearly.');
          return;
        }
        
        console.log('Sending voice recording, size:', audioBlob.size);
        
        // For voice input, always use audio response regardless of toggle
        const forceAudioResponse = 'audio';
        console.log('Forcing audio response for voice input:', forceAudioResponse);
        
        onSendAudio(audioBlob, forceAudioResponse);
      } else {
        alert('No audio recorded. Please try again and make sure to speak into the microphone.');
      }
    }
  }, [recordingState, startRecording, stopRecording, onSendAudio]);

  useEffect(() => {
    let silenceTimer;
    
    if (recordingState === 'recording') {
      const interval = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      return () => clearInterval(interval);
    }

    return () => {
      if (silenceTimer) {
        clearTimeout(silenceTimer);
      }
    };
  }, [recordingState]);

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (textInput.trim() && !isProcessing) {
      onSendText(textInput.trim(), currentResponseFormat);
      setTextInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit(e);
    }
  };

  // Add null check for messages
  const showInput = messages && messages.length > 0;

  if (!showInput) {
    return null;
  }

  return (
    <div className="message-input">
      <div className="input-controls">
        <div className="input-mode-toggle">
          <button 
            className={inputMode === 'text' ? 'active' : ''} 
            onClick={() => onInputModeChange && onInputModeChange('text')}
            type="button"
          >
            Text
          </button>
          <button 
            className={inputMode === 'voice' ? 'active' : ''} 
            onClick={() => onInputModeChange && onInputModeChange('voice')}
            type="button"
          >
            Voice
          </button>
        </div>

        <div className="response-format-selector">
          <label>Response:</label>
          <div className="response-format-toggle">
            <button 
              className={currentResponseFormat === 'text' ? 'active' : ''} 
              onClick={() => onResponseFormatChange && onResponseFormatChange('text')}
              type="button"
            >
              Text
            </button>
            <button 
              className={currentResponseFormat === 'audio' ? 'active' : ''} 
              onClick={() => onResponseFormatChange && onResponseFormatChange('audio')}
              type="button"
            >
              Audio
            </button>
          </div>
        </div>
      </div>

      {inputMode === 'text' ? (
        <form onSubmit={handleTextSubmit} className="text-input-form">
          <div className="text-input-container">
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              disabled={isProcessing}
              rows={1}
              className="text-input"
            />
            <button 
              type="submit" 
              disabled={!textInput.trim() || isProcessing}
              className="send-button"
            >
              Send
            </button>
          </div>
        </form>
      ) : (
        <div className="voice-input-container">
          <button
            onClick={handleVoiceRecord}
            disabled={isProcessing}
            className={`voice-button ${recordingState === 'recording' ? 'recording' : ''}`}
          >
            {recordingState === 'recording' ? 'Stop' : 'Record'}
          </button>
          
          {recordingState === 'recording' && (
            <div className="recording-info">
              <div className="recording-indicator">
                <div className="recording-dot"></div>
                <span>Recording... {recordingTime}s</span>
              </div>
              
              <div className="audio-visualizer">
                {Array.from({ length: 20 }, (_, i) => (
                  <div
                    key={i}
                    className="audio-bar"
                    style={{
                      height: `${Math.random() * 100}%`,
                      animationDelay: `${i * 0.1}s`
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MessageInput; 