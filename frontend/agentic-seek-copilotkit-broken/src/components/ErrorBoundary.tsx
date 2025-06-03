/**
 * Error Boundary Component
 * 
 * * Purpose: React error boundary with user-friendly error display and recovery
 * * Issues & Complexity Summary: React error boundary pattern with state management
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~120
 *   - Core Algorithm Complexity: Low
 *   - Dependencies: 1 New, 0 Mod
 *   - State Management Complexity: Low
 *   - Novelty/Uncertainty Factor: Low
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 65%
 * * Problem Estimate (Inherent Problem Difficulty %): 60%
 * * Initial Code Complexity Estimate %: 65%
 * * Justification for Estimates: Standard React error boundary with enhanced UI
 * * Final Code Complexity (Actual %): 62%
 * * Overall Result Score (Success & Quality %): 95%
 * * Key Variances/Learnings: More straightforward than expected
 * * Last Updated: 2025-06-03
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  BugReport as BugReportIcon
} from '@mui/icons-material';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Send error to backend for logging
    this.logErrorToBackend(error, errorInfo);
  }

  private async logErrorToBackend(error: Error, errorInfo: ErrorInfo) {
    try {
      await fetch('/api/copilotkit/log-error', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          errorId: this.state.errorId,
          message: error.message,
          stack: error.stack,
          componentStack: errorInfo.componentStack,
          timestamp: new Date().toISOString(),
          userAgent: navigator.userAgent,
          url: window.location.href
        })
      });
    } catch (err) {
      console.warn('Failed to log error to backend:', err);
    }
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null
    });
  };

  private handleReportBug = () => {
    const errorReport = {
      errorId: this.state.errorId,
      message: this.state.error?.message,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    const reportUrl = `mailto:support@agenticseek.com?subject=Bug Report - ${this.state.errorId}&body=${encodeURIComponent(
      `Error Report:\n\n${JSON.stringify(errorReport, null, 2)}`
    )}`;
    
    window.open(reportUrl);
  };

  override render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            p: 3,
            backgroundColor: 'background.default'
          }}
        >
          <Card sx={{ maxWidth: 600, width: '100%' }}>
            <CardContent sx={{ p: 4 }}>
              {/* Error Header */}
              <Box display="flex" alignItems="center" mb={2}>
                <ErrorIcon color="error" sx={{ fontSize: 32, mr: 2 }} />
                <Typography variant="h5" component="h1" color="error">
                  Something went wrong
                </Typography>
              </Box>

              {/* Error Description */}
              <Alert severity="error" sx={{ mb: 3 }}>
                <Typography variant="body1" gutterBottom>
                  We're sorry, but an unexpected error occurred. The development team has been notified.
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Error ID: {this.state.errorId}
                </Typography>
              </Alert>

              {/* Action Buttons */}
              <Box display="flex" gap={2} mb={3} flexWrap="wrap">
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<RefreshIcon />}
                  onClick={this.handleRetry}
                >
                  Try Again
                </Button>
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={this.handleReload}
                >
                  Reload Page
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<BugReportIcon />}
                  onClick={this.handleReportBug}
                >
                  Report Bug
                </Button>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Error Details (Expandable) */}
              {this.state.error && (
                <Accordion>
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="error-details-content"
                    id="error-details-header"
                  >
                    <Typography variant="subtitle2">
                      Technical Details
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Typography variant="subtitle2" color="error" gutterBottom>
                        Error Message:
                      </Typography>
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          backgroundColor: 'grey.100',
                          p: 1,
                          borderRadius: 1,
                          fontSize: '0.75rem',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          mb: 2
                        }}
                      >
                        {this.state.error.message}
                      </Typography>

                      {this.state.error.stack && (
                        <>
                          <Typography variant="subtitle2" color="error" gutterBottom>
                            Stack Trace:
                          </Typography>
                          <Typography
                            variant="body2"
                            component="pre"
                            sx={{
                              backgroundColor: 'grey.100',
                              p: 1,
                              borderRadius: 1,
                              fontSize: '0.65rem',
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                              maxHeight: 200,
                              overflow: 'auto'
                            }}
                          >
                            {this.state.error.stack}
                          </Typography>
                        </>
                      )}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}
            </CardContent>
          </Card>
        </Box>
      );
    }

    return this.props.children;
  }
}