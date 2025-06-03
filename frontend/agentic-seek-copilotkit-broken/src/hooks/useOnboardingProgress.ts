/**
 * Custom hook for managing onboarding progress and step completion
 * Handles persistence of user progress through the onboarding flow
 */

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

export interface OnboardingProgress {
  userId: string;
  currentStep: number;
  completedSteps: string[];
  progress: number;
  preferences: any;
  lastUpdated: string;
}

interface UserPreferences {
  primaryUseCase: string;
  experienceLevel: string;
  interestedFeatures: string[];
  preferredWorkflow: string;
  goals: string[];
  notificationPreferences: string[];
}

export const useOnboardingProgress = (userId: string) => {
  const [progress, setProgress] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load progress from localStorage and backend
  const loadProgress = useCallback(async () => {
    setIsLoading(true);
    try {
      // Try to load from localStorage first
      const localProgress = localStorage.getItem(`onboarding_progress_${userId}`);
      if (localProgress) {
        const parsed = JSON.parse(localProgress);
        setProgress(parsed.progress || 0);
        setCompletedSteps(parsed.completedSteps || []);
      }

      // Then sync with backend
      const response = await axios.get(`/api/copilotkit/onboarding/progress/${userId}`);
      if (response.data) {
        setProgress(response.data.progress || 0);
        setCompletedSteps(response.data.completedSteps || []);
      }
    } catch (err) {
      console.warn('Failed to load onboarding progress:', err);
      // Continue with localStorage data if backend fails
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  // Save progress to localStorage and backend
  const saveProgress = useCallback(async (newProgress: number, newCompletedSteps: string[]) => {
    const progressData = {
      userId,
      progress: newProgress,
      completedSteps: newCompletedSteps,
      lastUpdated: new Date().toISOString()
    };

    // Save to localStorage immediately
    localStorage.setItem(`onboarding_progress_${userId}`, JSON.stringify(progressData));

    // Save to backend
    try {
      await axios.post('/api/copilotkit/onboarding/progress', progressData);
    } catch (err) {
      console.warn('Failed to save onboarding progress to backend:', err);
      // Continue with localStorage for now
    }
  }, [userId]);

  // Update progress
  const updateProgress = useCallback(async (newProgress: number) => {
    setProgress(newProgress);
    await saveProgress(newProgress, completedSteps);
  }, [completedSteps, saveProgress]);

  // Mark step as complete
  const markStepComplete = useCallback(async (stepId: string) => {
    if (!completedSteps.includes(stepId)) {
      const newCompletedSteps = [...completedSteps, stepId];
      setCompletedSteps(newCompletedSteps);
      await saveProgress(progress, newCompletedSteps);
    }
  }, [completedSteps, progress, saveProgress]);

  // Reset progress
  const resetProgress = useCallback(async () => {
    setProgress(0);
    setCompletedSteps([]);
    localStorage.removeItem(`onboarding_progress_${userId}`);
    
    try {
      await axios.delete(`/api/copilotkit/onboarding/progress/${userId}`);
    } catch (err) {
      console.warn('Failed to reset onboarding progress:', err);
    }
  }, [userId]);

  // Get recommended next step based on preferences and completed steps
  const getRecommendedNextStep = useCallback((preferences: UserPreferences, completed: string[]) => {
    if (!preferences.primaryUseCase) {
      return 'Complete your profile setup to get personalized recommendations';
    }

    if (!completed.includes('profile')) {
      return 'Finish setting up your profile and preferences';
    }

    if (!completed.includes('tier_selection')) {
      return 'Select the tier that best fits your needs';
    }

    if (!completed.includes('features_tour')) {
      if (preferences.primaryUseCase === 'content_creation') {
        return 'Explore video generation and creative features';
      } else if (preferences.primaryUseCase === 'research_analysis') {
        return 'Try the research and analysis workflow tutorial';
      } else if (preferences.primaryUseCase === 'workflow_automation') {
        return 'Learn about workflow design and automation';
      }
      return 'Take the interactive feature tour';
    }

    if (preferences.experienceLevel === 'beginner' && !completed.includes('first_workflow')) {
      return 'Create your first simple workflow';
    }

    if (preferences.interestedFeatures.includes('video_generation') && !completed.includes('video_generation')) {
      return 'Try AI video generation (Enterprise tier required)';
    }

    return 'Explore advanced tutorials and start creating!';
  }, []);

  // Load progress on mount
  useEffect(() => {
    if (userId) {
      loadProgress();
    }
  }, [userId, loadProgress]);

  return {
    progress,
    completedSteps,
    isLoading,
    error,
    updateProgress,
    markStepComplete,
    resetProgress,
    getRecommendedNextStep,
    reload: loadProgress
  };
};