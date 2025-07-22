import React, { useState, useEffect, useRef } from 'react';
import ChatWindow from './components/ChatWindow';
import MessageInput from './components/MessageInput';
import websocketService from './services/websocketService';
import './styles/App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [errorMessage, setErrorMessage] = useState('');
  const [audioBlobUrl, setAudioBlobUrl] = useState(null); // URL for the *agent's* audio response
  const [processingStatus, setProcessingStatus] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedInputMode, setSelectedInputMode] = useState('text'); // 'text' or 'voice'
  const [responseFormat, setResponseFormat] = useState('text'); // Set default to 'text'
  const [selectedMethod, setSelectedMethod] = useState(null);
  const [inputMode, setInputMode] = useState('text');
  const [processedEvents, setProcessedEvents] = useState(new Set()); // Add this to track processed events
  const [isConnected, setIsConnected] = useState(false);

  const websocket = useRef(null); // Use useRef for the WebSocket service instance

  // State to accumulate audio chunks from server (for agent's audio response)
  const [serverAudioChunks, setServerAudioChunks] = useState([]);

  const resetChat = () => {
    setMessages([]);
    setIsProcessing(false);
    setProcessingStatus(null);
    setSelectedMethod(null);
    setErrorMessage('');
    setAudioBlobUrl(null);
    setServerAudioChunks([]);
    setInputMode('text');
    setResponseFormat('text'); // Reset to text default
    setProcessedEvents(new Set()); // Clear processed events
  };

  useEffect(() => {
    // Clear any existing WebSocket instance first
    if (websocket.current) {
      websocket.current.disconnect();
      websocket.current = null;
    }

    websocket.current = new websocketService();
    
    // Remove all existing listeners first to prevent duplicates
    websocket.current.off('connect');
    websocket.current.off('disconnect');
    websocket.current.off('error');
    websocket.current.off('processing_status');
    websocket.current.off('conversation_completed');
    websocket.current.off('audio_stream_start');
    websocket.current.off('audio_chunk');
    websocket.current.off('audio_stream_complete');
    websocket.current.off('handoff_suggestion');
    websocket.current.off('human_handoff_initiated');

    // Then add listeners
    websocket.current.on('connect', handleConnect);
    websocket.current.on('disconnect', handleDisconnect);
    websocket.current.on('error', handleError);
    websocket.current.on('processing_status', handleProcessingStatus);
    websocket.current.on('conversation_completed', handleConversationCompleted);
    websocket.current.on('audio_stream_start', handleAudioStreamStart);
    websocket.current.on('audio_chunk', handleAudioChunk);
    websocket.current.on('audio_stream_complete', handleAudioStreamComplete);
    websocket.current.on('handoff_suggestion', handleHandoffSuggestion);
    websocket.current.on('human_handoff_initiated', handleHandoffInitiated);

    websocket.current.connect();

    return () => {
      if (websocket.current) {
        websocket.current.disconnect();
        websocket.current = null;
      }
    };
  }, []); // Empty dependency array to run only once

  // Add this useEffect to handle audio URL updates
  useEffect(() => {
    if (audioBlobUrl) {
      console.log('Audio blob URL created, updating latest agent message');
      setMessages(prev => {
        const updated = [...prev];
        // Find the last agent message and add audio to it
        for (let i = updated.length - 1; i >= 0; i--) {
          if (updated[i].sender === 'Agent' && !updated[i].audioSrc) {
            updated[i] = { ...updated[i], audioSrc: audioBlobUrl };
            break;
          }
        }
        return updated;
      });
    }
  }, [audioBlobUrl]);

  // Add this debug useEffect
  useEffect(() => {
    console.log('Current state:', {
      messagesCount: messages.length,
      isProcessing,
      audioBlobUrl: audioBlobUrl ? 'present' : 'null',
      serverAudioChunks: serverAudioChunks.length
    });
  }, [messages, isProcessing, audioBlobUrl, serverAudioChunks]);

  const handleConnect = () => {
    console.log('WebSocket Connected!');
    setIsConnected(true); // Make sure this is being called
    setErrorMessage('');
    websocket.current.connect();
    };

    const handleDisconnect = () => {
    console.log('WebSocket Disconnected!');
    setIsConnected(false);
    websocket.current.disconnect();
    };

    const handleError = (error) => {
    console.log('WebSocket Error:', error);
    
    // Log more details about the error
    if (error && typeof error === 'object') {
      console.log('Error details:', JSON.stringify(error, null, 2));
      console.log('Error message:', error.message);
      console.log('Error type:', typeof error);
      console.log('Error keys:', Object.keys(error));
    }
    
    setErrorMessage(error?.message || 'Connection error occurred');
  };

  // Deprecated or simplified: 'conversation_completed' now handles adding the agent's message entirely.
  // const handleTextResponseReady = (data) => {
  //   // This is for text-only responses or the text part of 'both'
  //   // Given 'conversation_completed' will send both text and audioSrc,
  //   // this handler is now simplified for cases where text is the primary/only output.
  //   // For 'both' format, conversation_completed will add the message.
  //   if (data.response_format === 'text') {
  //     setMessages((prevMessages) => [
  //       ...prevMessages,
  //       {
  //         sender: 'Agent',
  //         text: data.response_text,
  //         time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  //       }
  //     ]);
  //   }
  //   // No need to clear processingStatus here if 'conversation_completed' will do it
  // };

  const handleAudioStreamStart = (data) => {
    console.log('Audio stream started:', data);
    setAudioBlobUrl(null);
    setServerAudioChunks([]); // Reset chunks for new stream
    // Don't change isProcessing here - let it stay true
  };

  const handleAudioChunk = (data) => {
    console.log('Audio chunk received, size:', data.chunk_size);
    // Store the chunk data (base64 encoded) - make sure it's valid base64
    if (data.chunk_data && typeof data.chunk_data === 'string') {
      setServerAudioChunks(prev => {
        const newChunks = [...prev, data.chunk_data];
        console.log('Total chunks stored:', newChunks.length);
        return newChunks;
      });
    } else {
      console.warn('Invalid chunk data received:', data);
    }
  };

  const handleAudioStreamComplete = (data) => {
    console.log('Audio stream completed:', data);
    
    setTimeout(() => {
      setServerAudioChunks(currentChunks => {
        console.log('Assembling', currentChunks.length, 'audio chunks');
        
        if (currentChunks.length > 0) {
          try {
            // Convert each chunk to binary and combine
            const audioChunks = [];
            
            for (const chunk of currentChunks) {
              try {
                const binaryString = atob(chunk);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                  bytes[i] = binaryString.charCodeAt(i);
                }
                audioChunks.push(bytes);
              } catch (e) {
                console.warn('Skipping invalid chunk:', e.message);
              }
            }
            
            // Calculate total length
            const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            
            // Combine all chunks
            const combinedBytes = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of audioChunks) {
              combinedBytes.set(chunk, offset);
              offset += chunk.length;
            }
            
            // Create audio blob
            const audioBlob = new Blob([combinedBytes], { type: 'audio/mp3' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            console.log('Audio blob created from binary chunks, size:', audioBlob.size);
            setAudioBlobUrl(audioUrl);
            
          } catch (error) {
            console.error('Failed to process audio chunks:', error);
          }
        }
        
        return currentChunks;
      });
    }, 100);
  };

  const handleConversationCompleted = (data) => {
    console.log('💬 Received conversation_completed event:', data);

    // Create unique event identifier for deduplication
    const eventId = `${data.transcribed_text || data.input_text}-${Math.floor(data.processing_time?.total || data.processing_time || 0)}-${data.response_text.substring(0, 50)}`;
    
    if (processedEvents.has(eventId)) {
      console.log('Ignoring duplicate conversation_completed event');
      return;
    }
    processedEvents.add(eventId);

    // Clean up old processed events (keep only last 10)
    if (processedEvents.size > 10) {
      const eventsArray = Array.from(processedEvents);
      processedEvents.clear();
      eventsArray.slice(-5).forEach(event => processedEvents.add(event));
    }

    setIsProcessing(false);
    setProcessingStatus(null);

    // Handle audio response if present in conversation_completed
    if (data.audio_data && data.response_format === 'audio') {
      console.log('Creating audio blob from conversation_completed data');
      try {
        const binaryString = atob(data.audio_data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const audioBlob = new Blob([bytes], { type: 'audio/mp3' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioBlobUrl(audioUrl);
        
        console.log('Audio blob created from conversation_completed, size:', audioBlob.size);
      } catch (error) {
        console.error('Failed to create audio blob from conversation_completed:', error);
      }
    }

    // Add message to chat
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Add user message (if from voice input)
    if (data.transcribed_text) {
      setMessages(prev => [...prev, {
        sender: 'You',
        text: data.transcribed_text,
        time: timestamp,
        audioSrc: null
      }]);
    }
    
    // Add agent response
    const agentMessage = {
      sender: 'Agent',
      text: data.response_text,
      time: timestamp,
      audioSrc: null // Will be set by useEffect when audioBlobUrl changes
    };

    setMessages(prev => [...prev, agentMessage]);
  };


    const handleProcessingStatus = (data) => {
    console.log('Processing status:', data);
      setIsProcessing(true);
    let stageMessage = '';
    switch (data.stage) {
      case 'stt':
        stageMessage = 'Converting your speech to text...';
        break;
      case 'llm':
        stageMessage = 'Generating response...';
        break;
      case 'tts':
        stageMessage = 'Synthesizing audio response...';
        break;
      default:
        stageMessage = 'Processing...';
    }
    setProcessingStatus(stageMessage);
  };

  const handleSendText = (text, responseFormat) => {
    setIsProcessing(true);
    setProcessingStatus('Sending message...');
    // Add user message immediately for text input to update UI
    setMessages((prevMessages) => [
      ...prevMessages,
      { sender: 'You', text: text, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
    ]);
    
    console.log('DEBUG: Sending text with response format:', responseFormat); // Add this debug line
    websocket.current.sendTextInput(text, responseFormat);
  };

  const handleSendAudio = (audioBlob, responseFormat) => {
      setIsProcessing(true);
    setProcessingStatus('Uploading audio...');
    // User's transcribed text for voice input will be added later in handleConversationCompleted
    websocket.current.sendAudioStream(audioBlob, responseFormat);
  };

  const handleMethodSelect = (method) => {
    setSelectedMethod(method);
    setInputMode(method === 'voice' ? 'voice' : 'text'); // Set input mode based on selection
    // Add first system message to start the conversation
    setMessages([{
      sender: 'Agent',
      text: `Great! Let's chat using ${method}. How can I help you today?`,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);
  };

  const handleHandoffSuggestion = (data) => {
    console.log('🤝 Handoff suggestion received:', data);
    
    // Add a special message with handoff option
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        sender: 'Agent',
        text: data.suggested_response,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        handoffSuggestion: {
          reason: data.reason,
          category: data.category,
          urgency: data.urgency,
          confidence: data.confidence
        }
      }
    ]);
  };

  const handleHandoffInitiated = (data) => {
    console.log('Human handoff initiated:', data);
    
    setMessages((prevMessages) => [
      ...prevMessages,
      {
        sender: 'Agent',
        text: `✅ ${data.message}\n\n📋 Ticket ID: ${data.ticket_id}\n⏱️ Estimated wait time: ${data.estimated_wait_time}`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isHandoffConfirmation: true
      }
    ]);
  };

  const handleRequestHumanAssistance = (handoffData) => {
    websocket.current.emit('request_human_assistance', {
      reason: handoffData.reason,
      category: handoffData.category,
      urgency: handoffData.urgency
    });
  };

  const connectionIndicatorClass = connectionStatus === 'connected' ? 'connected' : 'disconnected';

  return (
    <div className="app">
      <header className="app-header">
        <h1 
          onClick={resetChat} 
          className="app-title"
        >
          AI Customer Success Agent
        </h1>
        <div className="connection-status">
          <div className={`status-dot ${isConnected ? 'online' : 'offline'}`}></div>
          <span>{isConnected ? 'Online' : 'Offline'}</span>
        </div>
      </header>
      {errorMessage && (
        <div className="error-banner">
          <p>{errorMessage}</p>
          <button className="error-close" onClick={() => setErrorMessage('')}>X</button>
        </div>
      )}
      <main className="app-main">
        <ChatWindow 
          messages={messages} 
          isProcessing={isProcessing} 
          processingStage={processingStatus}
          onMethodSelect={handleMethodSelect}
        />
        
        {/* Only show input area when there are messages */}
        {messages.length > 0 && (
          <div className="input-area">
        <MessageInput
          onSendText={handleSendText}
          onSendAudio={handleSendAudio}
              inputMode={inputMode}
              onInputModeChange={setInputMode}
              currentResponseFormat={responseFormat}
              onResponseFormatChange={setResponseFormat}
          isProcessing={isProcessing}
              messages={messages} // Make sure this line exists
        />
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 