/**
 * Custom hook for AI video generation management
 * Handles video project creation, generation, and monitoring
 */

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { UserTier } from '../config/copilotkit.config';
import { VideoProject } from '../types/agent.types';

export interface VideoProjectRequest {
  concept: string;
  duration: number;
  style: string;
  metadata?: any;
}

export interface VideoGenerationOptions {
  optimization_level?: 'standard' | 'high_quality' | 'fast_render';
  use_apple_silicon?: boolean;
  priority?: 'low' | 'medium' | 'high';
  resume?: boolean;
}

export interface VideoGenerationResult {
  projectId: string;
  status: string;
  performanceImprovement?: number;
  efficiencyImprovement?: number;
  estimatedCompletion?: number;
}

export const useVideoGeneration = (userId: string, userTier: UserTier) => {
  const [projects, setProjects] = useState<VideoProject[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationProgress, setGenerationProgress] = useState<Record<string, number>>({});

  // Load existing projects
  const loadProjects = useCallback(async () => {
    try {
      const response = await axios.get(`/api/copilotkit/video/projects/${userId}`);
      setProjects(response.data || []);
    } catch (err) {
      console.warn('Failed to load video projects:', err);
      setProjects([]);
    }
  }, [userId]);

  // Create new video project
  const createProject = useCallback(async (request: VideoProjectRequest): Promise<VideoProject> => {
    if (userTier !== UserTier.ENTERPRISE) {
      throw new Error('Video generation requires Enterprise tier subscription');
    }

    try {
      const response = await axios.post('/api/copilotkit/video/create', {
        ...request,
        userId,
        userTier
      });

      const project: VideoProject = {
        id: response.data.projectId || `proj_${Date.now()}`,
        concept: request.concept,
        duration: request.duration,
        style: request.style,
        status: 'created',
        progress: 0,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        estimatedCompletion: Date.now() + (request.duration * 1000 * 0.5), // Estimate 0.5s processing per second of video
        assignedAgents: ['creative_agent', 'technical_agent', 'optimization_agent'],
        metadata: {
          resolution: '1920x1080',
          format: 'mp4',
          frameRate: 30,
          estimatedFileSize: Math.round(request.duration * 2.5), // ~2.5MB per second estimate
          appleSiliconOptimized: true,
          ...request.metadata
        }
      };

      setProjects(prev => [...prev, project]);
      return project;
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || 'Failed to create video project';
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  }, [userId, userTier]);

  // Start video generation
  const startGeneration = useCallback(async (projectId: string, options: VideoGenerationOptions = {}): Promise<VideoGenerationResult> => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await axios.post(`/api/copilotkit/video/generate/${projectId}`, {
        ...options,
        userId,
        userTier
      });

      // Update project status
      setProjects(prev => prev.map(project => 
        project.id === projectId 
          ? { ...project, status: 'processing' as const }
          : project
      ));

      setGenerationProgress(prev => ({ ...prev, [projectId]: 0 }));

      return {
        projectId,
        status: 'processing',
        performanceImprovement: response.data.performanceImprovement || 0,
        efficiencyImprovement: response.data.efficiencyImprovement || 0,
        estimatedCompletion: response.data.estimatedCompletion
      };
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || 'Failed to start video generation';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  }, [userId, userTier]);

  // Pause video generation
  const pauseGeneration = useCallback(async (projectId: string) => {
    try {
      await axios.post(`/api/copilotkit/video/pause/${projectId}`);
      
      setProjects(prev => prev.map(project => 
        project.id === projectId 
          ? { ...project, status: 'paused' as const }
          : project
      ));
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to pause video generation');
      throw err;
    }
  }, []);

  // Stop video generation
  const stopGeneration = useCallback(async (projectId: string) => {
    try {
      await axios.post(`/api/copilotkit/video/stop/${projectId}`);
      
      setProjects(prev => prev.map(project => 
        project.id === projectId 
          ? { ...project, status: 'stopped' as const }
          : project
      ));
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to stop video generation');
      throw err;
    }
  }, []);

  // Download completed video
  const downloadVideo = useCallback(async (projectId: string) => {
    try {
      const response = await axios.get(`/api/copilotkit/video/download/${projectId}`, {
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      const project = projects.find(p => p.id === projectId);
      link.download = `${project?.concept.replace(/\s+/g, '_').toLowerCase() || 'video'}_${projectId}.mp4`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to download video');
      throw err;
    }
  }, [projects]);

  // Get project status
  const getProjectStatus = useCallback(async (projectId: string): Promise<VideoProject | null> => {
    try {
      const response = await axios.get(`/api/copilotkit/video/status/${projectId}`);
      
      if (response.data) {
        const updatedProject: VideoProject = {
          ...response.data,
          id: projectId
        };

        setProjects(prev => prev.map(project => 
          project.id === projectId ? updatedProject : project
        ));

        return updatedProject;
      }
      
      return null;
    } catch (err) {
      console.warn('Failed to get project status:', err);
      return null;
    }
  }, []);

  // Monitor generation progress
  useEffect(() => {
    const processingProjects = projects.filter(p => p.status === 'processing');
    
    if (processingProjects.length > 0) {
      const interval = setInterval(async () => {
        for (const project of processingProjects) {
          try {
            const status = await getProjectStatus(project.id);
            if (status) {
              setGenerationProgress(prev => ({
                ...prev,
                [project.id]: status.progress
              }));
            }
          } catch (err) {
            console.warn(`Failed to update progress for project ${project.id}:`, err);
          }
        }
      }, 2000); // Update every 2 seconds

      return () => clearInterval(interval);
    }
    
    return undefined;
  }, [projects, getProjectStatus]);

  // Load projects on mount
  useEffect(() => {
    if (userId && userTier === UserTier.ENTERPRISE) {
      loadProjects();
    }
  }, [userId, userTier, loadProjects]);

  return {
    projects,
    createProject,
    startGeneration,
    pauseGeneration,
    stopGeneration,
    downloadVideo,
    getProjectStatus,
    isGenerating,
    generationProgress,
    error,
    reloadProjects: loadProjects
  };
};