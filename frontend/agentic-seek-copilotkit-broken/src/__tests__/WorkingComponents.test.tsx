/**
 * Comprehensive Test Suite for Working Components
 * Tests real functionality without mock data
 * 
 * Purpose: Verify all working components render and function correctly
 * Test Approach: Integration testing with real API calls where possible
 * Coverage: All critical user interactions and error states
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Test utilities
const theme = createTheme();

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <BrowserRouter>
    <ThemeProvider theme={theme}>
      {children}
    </ThemeProvider>
  </BrowserRouter>
);

describe('Working Components Integration Tests', () => {
  describe('Component Rendering Tests', () => {
    test('should render main navigation elements', async () => {
      // Import components dynamically to avoid build-time errors
      const { NavigationSidebar } = await import('../components/NavigationSidebar');
      
      render(
        <TestWrapper>
          <NavigationSidebar />
        </TestWrapper>
      );

      // Verify core navigation elements exist
      expect(screen.getByText(/agentic seek/i)).toBeInTheDocument();
      
      // Check for navigation items
      const dashboardLink = screen.getByText(/dashboard/i);
      expect(dashboardLink).toBeInTheDocument();
      
      // Verify accessibility
      expect(dashboardLink).toHaveAttribute('role', 'button');
    });

    test('should render agent coordination dashboard', async () => {
      const { AgentCoordinationDashboard } = await import('../components/AgentCoordinationDashboard');
      
      render(
        <TestWrapper>
          <AgentCoordinationDashboard />
        </TestWrapper>
      );

      // Check for main dashboard elements
      expect(screen.getByText(/agent coordination/i)).toBeInTheDocument();
      
      // Verify agent status display
      const agentSection = screen.getByText(/active agents/i);
      expect(agentSection).toBeInTheDocument();
    });

    test('should render workflow visualizer', async () => {
      const { LangGraphWorkflowVisualizer } = await import('../components/LangGraphWorkflowVisualizer');
      
      render(
        <TestWrapper>
          <LangGraphWorkflowVisualizer />
        </TestWrapper>
      );

      // Check for workflow elements
      expect(screen.getByText(/workflow/i)).toBeInTheDocument();
      
      // Verify control buttons exist
      const playButton = screen.getByRole('button', { name: /play/i });
      expect(playButton).toBeInTheDocument();
    });
  });

  describe('User Interaction Tests', () => {
    test('should handle navigation clicks', async () => {
      const { NavigationSidebar } = await import('../components/NavigationSidebar');
      
      render(
        <TestWrapper>
          <NavigationSidebar />
        </TestWrapper>
      );

      const dashboardLink = screen.getByText(/dashboard/i);
      fireEvent.click(dashboardLink);
      
      // Verify navigation state changes
      await waitFor(() => {
        expect(dashboardLink).toHaveClass('Mui-selected');
      });
    });

    test('should handle task creation', async () => {
      const { WorkingTaskManagementSystem } = await import('../components/WorkingTaskManagementSystem');
      
      render(
        <TestWrapper>
          <WorkingTaskManagementSystem />
        </TestWrapper>
      );

      // Find and click create task button
      const createButton = screen.getByText(/create task/i);
      fireEvent.click(createButton);
      
      // Verify task creation form appears
      await waitFor(() => {
        expect(screen.getByText(/new task/i)).toBeInTheDocument();
      });
    });

    test('should handle user profile updates', async () => {
      const { WorkingUserProfileManager } = await import('../components/WorkingUserProfileManager');
      
      render(
        <TestWrapper>
          <WorkingUserProfileManager />
        </TestWrapper>
      );

      // Find edit button
      const editButton = screen.getByText(/edit profile/i);
      fireEvent.click(editButton);
      
      // Verify profile form appears
      await waitFor(() => {
        expect(screen.getByText(/save/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling Tests', () => {
    test('should display error states gracefully', async () => {
      const { ErrorBoundary } = await import('../components/ErrorBoundary');
      
      // Component that throws error for testing
      const ErrorComponent = () => {
        throw new Error('Test error');
      };
      
      render(
        <TestWrapper>
          <ErrorBoundary>
            <ErrorComponent />
          </ErrorBoundary>
        </TestWrapper>
      );

      // Verify error boundary catches and displays error
      await waitFor(() => {
        expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      });
    });

    test('should handle loading states', async () => {
      const { LoadingSpinner } = await import('../components/LoadingSpinner');
      
      render(
        <TestWrapper>
          <LoadingSpinner message="Loading data..." />
        </TestWrapper>
      );

      // Verify loading spinner and message
      expect(screen.getByText(/loading data/i)).toBeInTheDocument();
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });

  describe('Data Visualization Tests', () => {
    test('should render charts with real data structure', async () => {
      const { AdvancedDataVisualization } = await import('../components/AdvancedDataVisualization');
      
      render(
        <TestWrapper>
          <AdvancedDataVisualization />
        </TestWrapper>
      );

      // Check for chart containers
      await waitFor(() => {
        expect(screen.getByText(/performance metrics/i)).toBeInTheDocument();
      });
      
      // Verify chart controls exist
      const chartControls = screen.getByText(/chart type/i);
      expect(chartControls).toBeInTheDocument();
    });

    test('should handle chart interactions', async () => {
      const { AdvancedDataVisualization } = await import('../components/AdvancedDataVisualization');
      
      render(
        <TestWrapper>
          <AdvancedDataVisualization />
        </TestWrapper>
      );

      // Find chart type selector
      const chartTypeButton = screen.getByText(/line chart/i);
      fireEvent.click(chartTypeButton);
      
      // Verify chart type changes
      await waitFor(() => {
        expect(screen.getByText(/area chart/i)).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Features Tests', () => {
    test('should initialize WebSocket connections', async () => {
      const { AgentCommunicationFeed } = await import('../components/AgentCommunicationFeed');
      
      render(
        <TestWrapper>
          <AgentCommunicationFeed />
        </TestWrapper>
      );

      // Check for connection status indicator
      await waitFor(() => {
        const statusIndicator = screen.getByText(/connected|connecting|disconnected/i);
        expect(statusIndicator).toBeInTheDocument();
      });
    });

    test('should display performance monitoring', async () => {
      const { AppleSiliconOptimizationPanel } = await import('../components/AppleSiliconOptimizationPanel');
      
      render(
        <TestWrapper>
          <AppleSiliconOptimizationPanel />
        </TestWrapper>
      );

      // Check for performance metrics
      await waitFor(() => {
        expect(screen.getByText(/cpu usage|memory usage|gpu usage/i)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility Tests', () => {
    test('should have proper ARIA labels', async () => {
      const { NavigationSidebar } = await import('../components/NavigationSidebar');
      
      render(
        <TestWrapper>
          <NavigationSidebar />
        </TestWrapper>
      );

      // Check for navigation aria-label
      const nav = screen.getByRole('navigation');
      expect(nav).toHaveAttribute('aria-label');
    });

    test('should support keyboard navigation', async () => {
      const { WorkingTaskManagementSystem } = await import('../components/WorkingTaskManagementSystem');
      
      render(
        <TestWrapper>
          <WorkingTaskManagementSystem />
        </TestWrapper>
      );

      // Find focusable elements
      const createButton = screen.getByText(/create task/i);
      createButton.focus();
      
      expect(document.activeElement).toBe(createButton);
    });
  });

  describe('Performance Tests', () => {
    test('should render components within performance budget', async () => {
      const startTime = performance.now();
      
      const { AgentCoordinationDashboard } = await import('../components/AgentCoordinationDashboard');
      
      render(
        <TestWrapper>
          <AgentCoordinationDashboard />
        </TestWrapper>
      );

      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Ensure render time is under 500ms
      expect(renderTime).toBeLessThan(500);
    });
  });
});

describe('Integration Flow Tests', () => {
  test('should complete end-to-end user workflow', async () => {
    // Test a complete user journey through the application
    const { NavigationSidebar } = await import('../components/NavigationSidebar');
    const { AgentCoordinationDashboard } = await import('../components/AgentCoordinationDashboard');
    
    render(
      <TestWrapper>
        <NavigationSidebar />
        <AgentCoordinationDashboard />
      </TestWrapper>
    );

    // Step 1: Navigate to dashboard
    const dashboardLink = screen.getByText(/dashboard/i);
    fireEvent.click(dashboardLink);
    
    // Step 2: Verify dashboard loads
    await waitFor(() => {
      expect(screen.getByText(/agent coordination/i)).toBeInTheDocument();
    });

    // Step 3: Interact with agent controls
    const agentButton = screen.getByText(/add agent/i);
    fireEvent.click(agentButton);
    
    // Step 4: Verify agent creation interface
    await waitFor(() => {
      expect(screen.getByText(/select agent type/i)).toBeInTheDocument();
    });
  });
});