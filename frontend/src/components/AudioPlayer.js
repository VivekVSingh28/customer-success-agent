import React, { useState, useEffect, useRef } from 'react';
import './AudioPlayer.css';

const AudioPlayer = ({ src, shouldAutoPlay = false }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  
  const audioRef = useRef(null);
  const hasAutoPlayed = useRef(false);
  const progressInterval = useRef(null);

  useEffect(() => {
    if (src && audioRef.current) {
      setIsLoading(true);
      audioRef.current.src = src;
      audioRef.current.load();
    }
  }, [src]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
      
      // Autoplay if requested
      if (shouldAutoPlay && !hasAutoPlayed.current) {
        hasAutoPlayed.current = true;
        // Small delay to ensure audio is ready
        setTimeout(() => {
          audio.play().catch(console.error);
        }, 100);
      }
    };

    const handlePlay = () => {
      setIsPlaying(true);
      // Start progress tracking
      progressInterval.current = setInterval(() => {
        if (audio && !audio.paused) {
          const current = audio.currentTime;
          const total = audio.duration;
          setCurrentTime(current);
          setProgress(total > 0 ? (current / total) * 100 : 0);
        }
      }, 100);
    };

    const handlePause = () => {
      setIsPlaying(false);
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
        progressInterval.current = null;
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      setProgress(0);
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
        progressInterval.current = null;
      }
    };

    const handleError = (e) => {
      console.error('Audio error:', e);
      setIsLoading(false);
      setIsPlaying(false);
    };

    const handleTimeUpdate = () => {
      if (audio && audio.duration) {
        const current = audio.currentTime;
        const total = audio.duration;
        setCurrentTime(current);
        setProgress(total > 0 ? (current / total) * 100 : 0);
      }
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);
    audio.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
      }
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [shouldAutoPlay]);

  const handlePlayPause = () => {
    const audio = audioRef.current;
    if (!audio || isLoading) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(console.error);
    }
  };

  const handleProgressClick = (e) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickProgress = clickX / rect.width;
    const seekTime = clickProgress * duration;
    
    audio.currentTime = seekTime;
    setCurrentTime(seekTime);
    setProgress(clickProgress * 100);
  };

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`audio-player ${isLoading ? 'loading' : ''}`}>
      <audio ref={audioRef} preload="metadata" />
      
      <button
        className="play-pause-btn"
        onClick={handlePlayPause}
        disabled={isLoading}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isLoading ? (
          <div className="loading-spinner"></div>
        ) : isPlaying ? (
          '⏸'
        ) : (
          '▶'
        )}
      </button>
      
      <div className="progress-container" onClick={handleProgressClick}>
        <div className="progress-track">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          />
          <div 
            className="progress-thumb" 
            style={{ left: `${progress}%` }}
          />
        </div>
      </div>
      
      <div className="time-display">
        {formatTime(currentTime)} / {formatTime(duration)}
      </div>
    </div>
  );
};

export default AudioPlayer; 