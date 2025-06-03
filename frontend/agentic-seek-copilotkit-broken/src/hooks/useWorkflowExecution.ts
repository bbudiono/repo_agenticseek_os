/**
 * Custom hook for workflow execution and monitoring
 * Handles LangGraph workflow execution with real-time updates
 */

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { UserTier } from '../config/copilotkit.config';
import { WorkflowNode, CoordinationInstance, WorkflowConnection, CoordinationStatus } from '../types/agent.types';

export interface WorkflowExecutionRequest {
  workflowId: string;
  inputData: any;
  executionMode: 'sequential' | 'parallel' | 'adaptive';
  priority: number;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
}

export interface WorkflowModification {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  modification: {
    type: string;
    reasoning: string;
  };
}

export const useWorkflowExecution = (userId: string, userTier: UserTier) => {
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [activeExecutions, setActiveExecutions] = useState<CoordinationInstance[]>([]);

  // Execute workflow
  const executeWorkflow = useCallback(async (request: WorkflowExecutionRequest): Promise<CoordinationInstance> => {
    setIsExecuting(true);
    setError(null);

    try {
      const response = await axios.post('/api/copilotkit/workflow/execute', {
        ...request,
        userId,
        userTier
      });

      const execution: CoordinationInstance = {
        id: response.data.executionId,
        status: CoordinationStatus.ACTIVE,
        taskDescription: `Workflow execution: ${request.workflowId}`,
        assignedAgents: [],
        progress: 0,
        currentStage: 'initializing',
        startTime: Date.now(),
        estimatedCompletion: Date.now() + (response.data.estimatedDuration || 60000),
        workflowGraph: request.nodes,
        communicationLog: [],
        performanceSnapshot: {
          responseTime: 0,
          accuracy: 0,
          uptime: 1,
          throughput: 0,
          errorRate: 0,
          resourceUtilization: {
            cpu: 0,
            memory: 0,
            bandwidth: 0
          },
          qualityScore: 0,
          successRate: 0,
          efficiency: 0,
          overallHealth: 1
        }
      };

      setActiveExecutions(prev => [...prev, execution]);
      return execution;
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || 'Failed to execute workflow';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsExecuting(false);
    }
  }, [userId, userTier]);

  // Pause workflow execution
  const pauseWorkflow = useCallback(async (executionId: string) => {
    try {
      await axios.post(`/api/copilotkit/workflow/pause/${executionId}`);
      
      setActiveExecutions(prev => 
        prev.map(exec => 
          exec.id === executionId 
            ? { ...exec, status: CoordinationStatus.PAUSED }
            : exec
        )
      );
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to pause workflow');
      throw err;
    }
  }, []);

  // Stop workflow execution
  const stopWorkflow = useCallback(async (executionId: string) => {
    try {
      await axios.post(`/api/copilotkit/workflow/stop/${executionId}`);
      
      setActiveExecutions(prev => 
        prev.map(exec => 
          exec.id === executionId 
            ? { ...exec, status: CoordinationStatus.CANCELLED }
            : exec
        )
      );
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to stop workflow');
      throw err;
    }
  }, []);

  // Modify workflow structure
  const modifyWorkflow = useCallback(async (workflowId: string, modification: WorkflowModification) => {
    try {
      const response = await axios.post(`/api/copilotkit/workflow/modify/${workflowId}`, {
        ...modification,
        userId,
        userTier
      });

      return response.data;
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || 'Failed to modify workflow';
      setError(errorMessage);
      throw new Error(errorMessage);
    }
  }, [userId, userTier]);

  // Get workflow status
  const getWorkflowStatus = useCallback(async (executionId: string): Promise<CoordinationInstance | null> => {
    try {
      const response = await axios.get(`/api/copilotkit/workflow/status/${executionId}`);
      
      if (response.data) {
        const updatedExecution: CoordinationInstance = {
          id: executionId,
          status: response.data.status,
          taskDescription: response.data.taskDescription || `Workflow ${executionId}`,
          assignedAgents: response.data.assignedAgents || [],
          progress: response.data.progress,
          currentStage: response.data.currentStage,
          startTime: response.data.startTime,
          estimatedCompletion: response.data.estimatedCompletion,
          workflowGraph: response.data.workflowGraph || [],
          communicationLog: response.data.communicationLog || [],
          performanceSnapshot: response.data.performanceSnapshot || {
            responseTime: 0,
            accuracy: 0,
            uptime: 1,
            throughput: 0,
            errorRate: 0,
            resourceUtilization: { cpu: 0, memory: 0, bandwidth: 0 },
            qualityScore: 0,
            successRate: 0,
            efficiency: 0,
            overallHealth: 1
          }
        };

        setActiveExecutions(prev => 
          prev.map(exec => 
            exec.id === executionId ? updatedExecution : exec
          )
        );

        return updatedExecution;
      }
      
      return null;
    } catch (err) {
      console.warn('Failed to get workflow status:', err);
      return null;
    }
  }, []);

  // Clean up completed executions
  useEffect(() => {
    const cleanup = setInterval(() => {
      setActiveExecutions(prev => 
        prev.filter(exec => 
          exec.status === 'active' || 
          exec.status === 'paused' ||
          (exec.status === 'completed' && Date.now() - exec.startTime < 300000) // Keep completed for 5 minutes
        )
      );
    }, 60000); // Clean up every minute

    return () => clearInterval(cleanup);
  }, []);

  return {
    executeWorkflow,
    pauseWorkflow,
    stopWorkflow,
    modifyWorkflow,
    getWorkflowStatus,
    isExecuting,
    executionProgress,
    activeExecutions,
    error
  };
};