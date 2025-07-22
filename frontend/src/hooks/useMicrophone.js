import { useState, useRef, useCallback } from 'react';

export const useMicrophone = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const recordingTimeoutRef = useRef(null);
  const durationIntervalRef = useRef(null);

  const stopRecording = useCallback(() => {
    return new Promise((resolve) => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.addEventListener('dataavailable', (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        }, { once: true });

        mediaRecorderRef.current.addEventListener('stop', () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          console.log('Recording stopped, blob size:', audioBlob.size);
          
          // Cleanup
          if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
          }
          
          audioChunksRef.current = [];
          setIsRecording(false);
          setRecordingDuration(0);
          setAudioLevel(0);

          // Clear timers
          if (recordingTimeoutRef.current) {
            clearTimeout(recordingTimeoutRef.current);
            recordingTimeoutRef.current = null;
          }
          
          if (durationIntervalRef.current) {
            clearInterval(durationIntervalRef.current);
            durationIntervalRef.current = null;
          }

          resolve(audioBlob);
        }, { once: true });

        mediaRecorderRef.current.stop();
      } else {
        resolve(null);
      }
    });
  }, []); // No dependencies needed

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      streamRef.current = stream;
      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.start();
      setIsRecording(true);

      // Set recording timeout to 10 seconds
      recordingTimeoutRef.current = setTimeout(() => {
        console.log('Recording stopped due to timeout (10 seconds)');
        stopRecording();
      }, 10000);

      // Update duration every second
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsRecording(false);
    }
  }, [stopRecording]); // Include stopRecording dependency

  return {
    startRecording,
    stopRecording,
    isRecording,
    recordingDuration,
    audioLevel
  };
}; 