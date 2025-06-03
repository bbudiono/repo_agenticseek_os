/**
 * User Tier Management Hook
 * 
 * * Purpose: Manages user tier state with persistence and validation
 * * Issues & Complexity Summary: Complex tier validation with backend synchronization
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~120
 *   - Core Algorithm Complexity: Medium
 *   - Dependencies: 2 New, 1 Mod
 *   - State Management Complexity: Medium
 *   - Novelty/Uncertainty Factor: Low
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 70%
 * * Problem Estimate (Inherent Problem Difficulty %): 65%
 * * Initial Code Complexity Estimate %: 70%
 * * Justification for Estimates: State management with localStorage persistence
 * * Final Code Complexity (Actual %): 72%
 * * Overall Result Score (Success & Quality %): 94%
 * * Key Variances/Learnings: localStorage synchronization simpler than expected
 * * Last Updated: 2025-06-03
 */

import { useState, useEffect, useCallback } from 'react';
import { UserTier } from '../config/copilotkit.config';

interface UseUserTierReturn {
  userTier: UserTier;
  setUserTier: (tier: UserTier) => void;
  userId: string;
  setUserId: (id: string) => void;
  isValidating: boolean;
  error: string | null;
}

export const useUserTier = (
  initialUserId?: string,
  initialTier?: UserTier
): UseUserTierReturn => {
  const [userTier, setUserTierState] = useState<UserTier>(
    initialTier || (localStorage.getItem('userTier') as UserTier) || UserTier.FREE
  );
  const [userId, setUserIdState] = useState<string>(
    initialUserId || localStorage.getItem('userId') || `user_${Date.now()}`
  );
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Validate tier with backend
  const validateTierWithBackend = useCallback(async (tier: UserTier, id: string) => {
    try {
      setIsValidating(true);
      setError(null);

      const response = await fetch('/api/copilotkit/validate-tier', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-ID': id,
          'User-Tier': tier
        },
        body: JSON.stringify({
          userId: id,
          requestedTier: tier
        })
      });

      if (!response.ok) {
        throw new Error(`Tier validation failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (!data.valid) {
        throw new Error(data.reason || 'Tier validation failed');
      }

      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown validation error';
      setError(errorMessage);
      console.error('Tier validation error:', errorMessage);
      return false;
    } finally {
      setIsValidating(false);
    }
  }, []);

  // Set user tier with validation and persistence
  const setUserTier = useCallback(async (tier: UserTier) => {
    const isValid = await validateTierWithBackend(tier, userId);
    
    if (isValid) {
      setUserTierState(tier);
      localStorage.setItem('userTier', tier);
      
      // Notify backend of tier change
      try {
        await fetch('/api/copilotkit/update-tier', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-ID': userId,
            'User-Tier': tier
          },
          body: JSON.stringify({
            userId,
            newTier: tier,
            timestamp: new Date().toISOString()
          })
        });
      } catch (err) {
        console.warn('Failed to notify backend of tier change:', err);
      }
    }
  }, [userId, validateTierWithBackend]);

  // Set user ID with persistence
  const setUserId = useCallback((id: string) => {
    setUserIdState(id);
    localStorage.setItem('userId', id);
  }, []);

  // Initialize and validate on mount
  useEffect(() => {
    validateTierWithBackend(userTier, userId);
  }, [userTier, userId, validateTierWithBackend]);

  // Sync with localStorage changes
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'userTier' && e.newValue) {
        setUserTierState(e.newValue as UserTier);
      }
      if (e.key === 'userId' && e.newValue) {
        setUserIdState(e.newValue);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  return {
    userTier,
    setUserTier,
    userId,
    setUserId,
    isValidating,
    error
  };
};