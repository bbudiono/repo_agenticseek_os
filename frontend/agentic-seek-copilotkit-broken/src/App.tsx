/**
 * AgenticSeek CopilotKit Main Application
 * 
 * * Purpose: Main application component with CopilotKit Provider and multi-agent coordination
 * * Issues & Complexity Summary: Complex Provider setup with tier management and real-time coordination
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~300
 *   - Core Algorithm Complexity: High
 *   - Dependencies: 4 New, 2 Mod
 *   - State Management Complexity: High
 *   - Novelty/Uncertainty Factor: High
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 80%
 * * Problem Estimate (Inherent Problem Difficulty %): 75%
 * * Initial Code Complexity Estimate %: 80%
 * * Justification for Estimates: Complex Provider setup with multiple contexts and real-time coordination
 * * Final Code Complexity (Actual %): TBD
 * * Overall Result Score (Success & Quality %): TBD
 * * Key Variances/Learnings: TBD
 * * Last Updated: 2025-06-03
 */

import React, { useState, useEffect, useMemo } from 'react';
import { CopilotKit } from '@copilotkit/react-core';
import { CopilotSidebar } from '@copilotkit/react-ui';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, Snackbar, Alert, Typography } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Import configuration and types
import { 
  DEFAULT_COPILOTKIT_CONFIG, 
  UserTier, 
  getCopilotInstructions, 
  COPILOT_LABELS,
  getTierLimits
} from './config/copilotkit.config';

// Import components
import { AgentCoordinationDashboard } from './components/AgentCoordinationDashboard';
import { LangGraphWorkflowVisualizer } from './components/LangGraphWorkflowVisualizer';
import { VideoGenerationInterface } from './components/VideoGenerationInterface';
import { AppleSiliconOptimizationPanel } from './components/AppleSiliconOptimizationPanel';
import { AgentCommunicationFeed } from './components/AgentCommunicationFeed';
import { TierManagementPanel } from './components/TierManagementPanel';
import { NavigationSidebar } from './components/NavigationSidebar';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LoadingSpinner } from './components/LoadingSpinner';
// import { OnboardingFlow } from './components/OnboardingFlow';
import { AdvancedDataVisualization } from './components/AdvancedDataVisualization';
import { WorkingOnboardingInterface } from './components/WorkingOnboardingInterface';
import { WorkingUserProfileManager } from './components/WorkingUserProfileManager';
import { WorkingTaskManagementSystem } from './components/WorkingTaskManagementSystem';
import { WorkingSystemConfigurationManager } from './components/WorkingSystemConfigurationManager';

// Import hooks and services
import { useUserTier } from './hooks/useUserTier';
import { useWebSocket } from './hooks/useWebSocket';
import { usePerformanceMonitoring } from './hooks/usePerformanceMonitoring';
import { NotificationService } from './services/NotificationService';

// Import styles
import './App.css';

// React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: DEFAULT_COPILOTKIT_CONFIG.retryConfig.maxAttempts,
      retryDelay: DEFAULT_COPILOTKIT_CONFIG.retryConfig.retryDelay,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// Theme configuration
const createAppTheme = (mode: 'light' | 'dark') =>
  createTheme({
    palette: {
      mode,
      primary: {
        main: process.env.REACT_APP_PRIMARY_COLOR || '#1976d2',
      },
      secondary: {
        main: process.env.REACT_APP_SECONDARY_COLOR || '#dc004e',
      },
      background: {
        default: mode === 'dark' ? '#0d1117' : '#ffffff',
        paper: mode === 'dark' ? '#161b22' : '#ffffff',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 700,
      },
      h2: {
        fontWeight: 600,
      },
      h3: {
        fontWeight: 600,
      },
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            borderRadius: 8,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: mode === 'dark' 
              ? '0 4px 6px -1px rgba(0, 0, 0, 0.3)' 
              : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  });

interface AppProps {
  initialUserId?: string;
  initialTier?: UserTier;
}

const App: React.FC<AppProps> = ({ 
  initialUserId = 'demo_user', 
  initialTier = UserTier.FREE 
}) => {
  const [themeMode, setThemeMode] = useState<'light' | 'dark'>(
    (process.env.REACT_APP_THEME as 'light' | 'dark') || 'dark'
  );
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(
    process.env.REACT_APP_SIDEBAR_DEFAULT_OPEN === 'true'
  );
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingCompleted, setOnboardingCompleted] = useState(false);

  // Custom hooks
  const { userTier, setUserTier, userId, setUserId } = useUserTier(initialUserId, initialTier);
  const { isConnected, connectionError } = useWebSocket(DEFAULT_COPILOTKIT_CONFIG.wsUrl);
  const { performanceMetrics } = usePerformanceMonitoring();

  // Theme memoization
  const theme = useMemo(() => createAppTheme(themeMode), [themeMode]);

  // User tier limits
  const tierLimits = useMemo(() => getTierLimits(userTier), [userTier]);

  // CopilotKit configuration
  const copilotKitHeaders = useMemo(() => ({
    'User-ID': userId,
    'User-Tier': userTier,
    'Content-Type': 'application/json',
  }), [userId, userTier]);

  // Initialization effect
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setIsLoading(true);
        
        // Initialize notification service
        await NotificationService.initialize();
        
        // Check if user has completed onboarding
        const onboardingStatus = localStorage.getItem(`onboarding_completed_${userId}`);
        const hasCompletedOnboarding = onboardingStatus === 'true';
        
        setOnboardingCompleted(hasCompletedOnboarding);
        
        // Only show onboarding if explicitly requested
        if (new URLSearchParams(window.location.search).get('onboarding') === 'true') {
          setShowOnboarding(true);
        } else {
          // Set as completed by default to avoid blocking the UI
          setOnboardingCompleted(true);
        }
        
        // Simulate initialization delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setIsLoading(false);
      } catch (err) {
        console.error('App initialization error:', err);
        setError('Failed to initialize application');
        setIsLoading(false);
      }
    };

    initializeApp();
  }, [userId]);

  // Error handling effect
  useEffect(() => {
    if (connectionError) {
      setError(`WebSocket connection failed: ${connectionError}`);
    }
  }, [connectionError]);

  // Clear error handler
  const handleClearError = () => {
    setError(null);
  };

  // Tier change handler
  const handleTierChange = (newTier: UserTier) => {
    setUserTier(newTier);
    NotificationService.showSuccess(`Upgraded to ${newTier.toUpperCase()} tier!`);
  };

  // Theme toggle handler
  const handleThemeToggle = () => {
    setThemeMode(prev => prev === 'light' ? 'dark' : 'light');
  };

  // Onboarding completion handler
  const handleOnboardingComplete = (preferences: any) => {
    setOnboardingCompleted(true);
    setShowOnboarding(false);
    localStorage.setItem(`onboarding_completed_${userId}`, 'true');
    localStorage.setItem(`user_preferences_${userId}`, JSON.stringify(preferences));
    NotificationService.showSuccess('Welcome to AgenticSeek! Your setup is complete.');
  };

  // Restart onboarding handler
  const handleRestartOnboarding = () => {
    setShowOnboarding(true);
  };

  if (isLoading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <LoadingSpinner message="Initializing AgenticSeek..." />
      </ThemeProvider>
    );
  }

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <QueryClientProvider client={queryClient}>
          <Router>
            <CopilotKit 
              runtimeUrl={DEFAULT_COPILOTKIT_CONFIG.apiUrl}
              headers={copilotKitHeaders}
              publicApiKey={DEFAULT_COPILOTKIT_CONFIG.publicApiKey || ''}
            >
              <Box sx={{ display: 'flex', height: '100vh' }}>
                
                {/* Navigation Sidebar */}
                <NavigationSidebar
                  open={sidebarOpen}
                  onToggle={() => setSidebarOpen(!sidebarOpen)}
                  userTier={userTier}
                  onTierChange={handleTierChange}
                  onThemeToggle={handleThemeToggle}
                  themeMode={themeMode}
                  connectionStatus={isConnected ? 'connected' : 'disconnected'}
                  performanceMetrics={performanceMetrics || {
                    cpuUsage: 0,
                    memoryUsage: 0,
                    networkLatency: 0,
                    frameRate: 60,
                    isAppleSilicon: false,
                    timestamp: new Date().toISOString()
                  }}
                />

                {/* Main Content Area */}
                <Box 
                  component="main" 
                  sx={{ 
                    flexGrow: 1, 
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                    ml: sidebarOpen ? 0 : -30, // Adjust for sidebar
                    transition: 'margin 0.3s ease-in-out'
                  }}
                >
                  <Routes>
                    <Route 
                      path="/" 
                      element={
                        <MainDashboard 
                          userTier={userTier}
                          tierLimits={tierLimits}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/agents" 
                      element={
                        <AgentCoordinationDashboard 
                          userTier={userTier}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/workflows" 
                      element={
                        <LangGraphWorkflowVisualizer 
                          userTier={userTier}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/video" 
                      element={
                        <VideoGenerationInterface 
                          userTier={userTier}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/optimization" 
                      element={
                        <AppleSiliconOptimizationPanel 
                          userTier={userTier}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/communication" 
                      element={
                        <AgentCommunicationFeed 
                          userTier={userTier}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/tier-management" 
                      element={
                        <TierManagementPanel 
                          currentTier={userTier}
                          onTierChange={handleTierChange}
                          userId={userId}
                        />
                      } 
                    />
                    <Route 
                      path="/analytics" 
                      element={
                        <AdvancedDataVisualization 
                          userTier={userTier}
                          userId={userId}
                          data={[]}
                        />
                      } 
                    />
                    <Route 
                      path="/onboarding" 
                      element={
                        <WorkingOnboardingInterface 
                          userId={userId}
                          userTier={userTier}
                        />
                      } 
                    />
                    <Route 
                      path="/profile" 
                      element={
                        <WorkingUserProfileManager 
                          userId={userId}
                          userTier={userTier}
                        />
                      } 
                    />
                    <Route 
                      path="/tasks" 
                      element={
                        <WorkingTaskManagementSystem 
                          userId={userId}
                          userTier={userTier}
                        />
                      } 
                    />
                    <Route 
                      path="/settings" 
                      element={
                        <WorkingSystemConfigurationManager 
                          userId={userId}
                          userTier={userTier}
                        />
                      } 
                    />
                  </Routes>
                </Box>

                {/* Onboarding Flow - temporarily disabled */}
                {/* {showOnboarding && (
                  <OnboardingFlow 
                    userId={userId}
                    onComplete={handleOnboardingComplete}
                    isOpen={showOnboarding}
                    onClose={() => setShowOnboarding(false)}
                  />
                )} */}

                {/* CopilotKit Sidebar */}
                <CopilotSidebar
                  instructions={getCopilotInstructions(userTier)}
                  labels={{
                    title: COPILOT_LABELS.title,
                    initial: COPILOT_LABELS.initial,
                  }}
                  defaultOpen={false}
                  clickOutsideToClose={true}
                  shortcut="cmd+/"
                />

              </Box>

              {/* Error Snackbar */}
              <Snackbar 
                open={!!error} 
                autoHideDuration={6000} 
                onClose={handleClearError}
                anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
              >
                <Alert 
                  onClose={handleClearError} 
                  severity="error" 
                  sx={{ width: '100%' }}
                >
                  {error}
                </Alert>
              </Snackbar>

            </CopilotKit>
          </Router>
        </QueryClientProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

// Main Dashboard Component
interface MainDashboardProps {
  userTier: UserTier;
  tierLimits: any;
  userId: string;
}

const MainDashboard: React.FC<MainDashboardProps> = ({ 
  userTier, 
  tierLimits, 
  userId 
}) => {
  return (
    <Box sx={{ 
      p: 3, 
      height: '100%', 
      overflow: 'auto',
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
      gap: 3
    }}>
      
      {/* Agent Coordination Preview */}
      <Box sx={{ gridColumn: 'span 1' }}>
        <AgentCoordinationDashboard 
          userTier={userTier}
          userId={userId}
          isPreview={true}
        />
      </Box>

      {/* Workflow Visualizer Preview */}
      <Box sx={{ gridColumn: 'span 1' }}>
        <LangGraphWorkflowVisualizer 
          userTier={userTier}
          userId={userId}
          isPreview={true}
        />
      </Box>

      {/* Apple Silicon Optimization */}
      <Box sx={{ gridColumn: 'span 1' }}>
        <AppleSiliconOptimizationPanel 
          userTier={userTier}
          userId={userId}
          isPreview={true}
        />
      </Box>

      {/* Video Generation (Enterprise only) */}
      {tierLimits.videoGeneration && (
        <Box sx={{ gridColumn: 'span 1' }}>
          <VideoGenerationInterface 
            userTier={userTier}
            userId={userId}
            isPreview={true}
          />
        </Box>
      )}

      {/* Agent Communication Feed */}
      <Box sx={{ gridColumn: 'span 1' }}>
        <AgentCommunicationFeed 
          userTier={userTier}
          userId={userId}
        />
      </Box>

      {/* Advanced Data Visualization */}
      <Box sx={{ gridColumn: 'span 1' }}>
        <AdvancedDataVisualization 
          userTier={userTier}
          userId={userId}
          data={[]}
        />
      </Box>

    </Box>
  );
};

export default App;