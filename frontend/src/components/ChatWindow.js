import React, { useRef, useEffect } from 'react';
import AudioPlayer from './AudioPlayer';
import './ChatWindow.css';

const ChatWindow = ({ messages, isProcessing, processingStage, onMethodSelect, onRequestHumanAssistance }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing]);

  const formatText = (text) => {
    if (!text) return null;

    // Remove markdown-style formatting (** for bold)
    text = text.replace(/\*\*/g, '');

    // Split text into lines and process
    const lines = text.split('\n').filter(line => line.trim());
    const elements = [];
    
    lines.forEach((line, index) => {
      const trimmedLine = line.trim();
      
      // Check for numbered list items (### 1. or just 1.)
      const numberedMatch = trimmedLine.match(/^(#{1,3}\s*)?(\d+)\.\s*(.+)/);
      if (numberedMatch) {
        const title = numberedMatch[3];
        elements.push(
          <h4 key={`title-${index}`} style={{ 
            fontSize: '16px', 
            fontWeight: '600', 
            margin: '12px 0 8px 0',
            color: 'inherit'
          }}>
            {numberedMatch[2]}. {title}
          </h4>
        );
        return;
      }
      
      // Check for bullet points
      const bulletMatch = trimmedLine.match(/^[•-]\s*(.+)/);
      if (bulletMatch) {
        elements.push(
          <div key={`bullet-${index}`} style={{ 
            display: 'flex', 
            alignItems: 'flex-start', 
            margin: '4px 0',
            paddingLeft: '16px'
          }}>
            <span style={{ marginRight: '8px', fontSize: '14px' }}>•</span>
            <span>{bulletMatch[1]}</span>
          </div>
        );
        return;
      }
      
      // Regular paragraph
      if (trimmedLine) {
        elements.push(
          <p key={`p-${index}`} style={{ 
            margin: '8px 0',
            lineHeight: '1.4'
          }}>
            {trimmedLine}
          </p>
        );
      }
    });

    return elements.length > 0 ? elements : <p>{text}</p>;
  };

  const renderMessageContent = (message) => {
    return (
      <div className="message-content">
        {message.text && (
          <div className="message-text">
            {formatText(message.text)}
          </div>
        )}
        
        {/* Handoff suggestion UI */}
        {message.handoffSuggestion && (
          <div className="handoff-suggestion">
            <div className="handoff-info">
              <span className="handoff-category">{message.handoffSuggestion.category}</span>
              <span className="handoff-urgency">{message.handoffSuggestion.urgency}</span>
            </div>
            <div className="handoff-actions">
              <button 
                className="handoff-accept"
                onClick={() => onRequestHumanAssistance && onRequestHumanAssistance(message.handoffSuggestion)}
              >
                Connect to Human Agent
              </button>
              <button className="handoff-decline">
                Continue with AI
              </button>
            </div>
          </div>
        )}

        {message.audioSrc && (
          <div className="audio-message">
            <AudioPlayer 
              src={message.audioSrc} 
              shouldAutoPlay={message.shouldAutoPlay || false}
            />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="messages-container">
      {messages.length === 0 && !isProcessing && (
        <div className="welcome-container">
          <div className="welcome-content">
            <div className="welcome-icon">🧑🏻‍💻</div>
            <h2>Welcome</h2>
            <p className="welcome-subtitle">I'm here to help! You can interact with me through text or voice.</p>

            <div className="interaction-methods">
              <div
                className="method-card clickable"
                onClick={() => onMethodSelect('text')}
                role="button"
                tabIndex={0}
              >
                <span className="method-icon">💬</span>
                <h3>Text Chat</h3>
                <p>Type your questions or requests</p>
              </div>
              <div
                className="method-card clickable"
                onClick={() => onMethodSelect('voice')}
                role="button"
                tabIndex={0}
              >
                <span className="method-icon">🎙️</span>
                <h3>Voice Chat</h3>
                <p>Speak naturally with voice input</p>
              </div>
            </div>

            <div className="welcome-prompt">
              <span className="prompt-icon">💡</span>
              <p>Try asking about our products, services, or any support questions!</p>
            </div>
          </div>
        </div>
      )}
      {messages.map((message, index) => (
        <div key={index} className={`message ${message.sender.toLowerCase()}`}>
          <div className="message-header">
            <span className="message-sender">{message.sender}</span>
            <span className="message-time">{message.time}</span>
          </div>
          <div className="message-content">
            {message.text && (
              <div className="message-text">
                {formatText(message.text)}
              </div>
            )}
            {message.audioSrc && (
              <div className="audio-message">
                <AudioPlayer 
                  src={message.audioSrc} 
                  shouldAutoPlay={message.shouldAutoPlay || false}
                />
              </div>
            )}
          </div>
        </div>
      ))}
      {isProcessing && (
        <div className="message agent">
          <div className="message-content">
            <div className="processing-indicator">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <div className="processing-text">
                {processingStage || 'Processing...'}
              </div>
            </div>
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatWindow; 