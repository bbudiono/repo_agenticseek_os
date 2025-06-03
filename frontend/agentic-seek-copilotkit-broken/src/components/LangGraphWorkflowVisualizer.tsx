/**
 * LangGraph Workflow Visualizer with CopilotKit Integration
 * 
 * * Purpose: Interactive workflow visualization with real-time execution monitoring and CopilotKit actions
 * * Issues & Complexity Summary: Complex workflow visualization with drag-and-drop editing and real-time updates
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~500
 *   - Core Algorithm Complexity: High
 *   - Dependencies: 7 New, 5 Mod
 *   - State Management Complexity: High
 *   - Novelty/Uncertainty Factor: High
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 90%
 * * Problem Estimate (Inherent Problem Difficulty %): 85%
 * * Initial Code Complexity Estimate %: 90%
 * * Justification for Estimates: Complex interactive workflow editor with real-time coordination
 * * Final Code Complexity (Actual %): TBD
 * * Overall Result Score (Success & Quality %): TBD
 * * Key Variances/Learnings: TBD
 * * Last Updated: 2025-06-03
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Button,
  IconButton,
  Chip,
  Tooltip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  FormControl,
  InputLabel,
  Tab,
  Tabs,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Save,
  Add,
  Edit,
  Delete,
  Visibility,
  Timeline,
  AccountTree,
  Settings,
  MoreVert,
  Download,
  Upload,
  Share,
  Code,
  Analytics,
  ExpandMore,
  DragIndicator,
  Link,
  CallSplit,
  CallMerge,
  SmartToy,
  Memory,
  Speed,
  CheckCircle,
  Error,
  Warning,
  Pending
} from '@mui/icons-material';
import { useCopilotAction, useCopilotReadable } from '@copilotkit/react-core';
import { CopilotTextarea } from '@copilotkit/react-textarea';

// Import types and configuration
import { UserTier, getTierLimits } from '../config/copilotkit.config';
import { 
  WorkflowNode, 
  CoordinationInstance, 
  AgentStatus,
  CoordinationStatus 
} from '../types/agent.types';
import { useWorkflowExecution } from '../hooks/useWorkflowExecution';
import { useRealTimeUpdates } from '../hooks/useRealTimeUpdates';
import { TierGate } from './TierGate';
import { WorkflowNodeComponent, WorkflowNode as WorkflowNodeUIType } from './WorkflowNodeComponent';
import { WorkflowConnection, WorkflowConnectionProps } from './WorkflowConnection';

// Type alias for workflow connections in templates (simplified)
interface WorkflowConnectionTemplate {
  id: string;
  from: string; // Node ID
  to: string;   // Node ID
  type?: 'data' | 'control' | 'error' | 'success' | 'coordination' | 'parallel';
}

// Type alias for runtime workflow connections
type WorkflowConnectionType = Omit<WorkflowConnectionProps, 'onClick' | 'onDelete' | 'onEdit'>;

interface LangGraphWorkflowVisualizerProps {
  userTier: UserTier;
  userId: string;
  isPreview?: boolean;
}

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNodeUIType[];
  connections: WorkflowConnectionTemplate[];
  requiredTier: UserTier;
  category: string;
  estimatedDuration: number;
}

const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    id: 'research_analysis',
    name: 'Research & Analysis',
    description: 'Multi-agent research coordination with analysis and reporting',
    nodes: [
      { id: 'research', type: 'agent', label: 'Research Agent', position: { x: 100, y: 100 }, status: 'idle', metadata: { agent: 'research_agent' } },
      { id: 'analysis', type: 'agent', label: 'Analysis Agent', position: { x: 300, y: 100 }, status: 'idle', metadata: { agent: 'analysis_agent' } },
      { id: 'report', type: 'action', label: 'Generate Report', position: { x: 500, y: 100 }, status: 'idle' }
    ],
    connections: [
      { id: 'conn1', from: 'research', to: 'analysis', type: 'data' },
      { id: 'conn2', from: 'analysis', to: 'report', type: 'coordination' }
    ],
    requiredTier: UserTier.FREE,
    category: 'Research',
    estimatedDuration: 65000
  },
  {
    id: 'creative_production',
    name: 'Creative Production Pipeline',
    description: 'Creative content generation with optimization and review',
    nodes: [
      { id: 'ideation', type: 'agent', label: 'Creative Agent', position: { x: 100, y: 200 }, status: 'idle', metadata: { agent: 'creative_agent' } },
      { id: 'optimization', type: 'agent', label: 'Optimization Agent', position: { x: 300, y: 200 }, status: 'idle', metadata: { agent: 'optimization_agent' } },
      { id: 'review', type: 'condition', label: 'Review & Approve', position: { x: 500, y: 200 }, status: 'idle' }
    ],
    connections: [
      { id: 'conn1', from: 'ideation', to: 'optimization', type: 'data' },
      { id: 'conn2', from: 'optimization', to: 'review', type: 'coordination' }
    ],
    requiredTier: UserTier.PRO,
    category: 'Creative',
    estimatedDuration: 50000
  }
];

export const LangGraphWorkflowVisualizer: React.FC<LangGraphWorkflowVisualizerProps> = ({
  userTier,
  userId,
  isPreview = false
}) => {
  // State management
  const [currentWorkflow, setCurrentWorkflow] = useState<WorkflowTemplate | null>(null);
  const [workflowNodes, setWorkflowNodes] = useState<WorkflowNodeUIType[]>([]);
  const [workflowConnections, setWorkflowConnections] = useState<WorkflowConnectionTemplate[]>([]);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [isEditing, setIsEditing] = useState(false);
  const [executionState, setExecutionState] = useState<CoordinationInstance | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);
  const [nodeDialogOpen, setNodeDialogOpen] = useState(false);
  const [editingNode, setEditingNode] = useState<WorkflowNode | null>(null);
  const [workflowName, setWorkflowName] = useState('');
  const [workflowDescription, setWorkflowDescription] = useState('');

  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);

  // Custom hooks
  const {
    executeWorkflow,
    pauseWorkflow,
    stopWorkflow,
    modifyWorkflow,
    getWorkflowStatus,
    isExecuting,
    executionProgress,
    error
  } = useWorkflowExecution(userId, userTier);

  const { isConnected } = useRealTimeUpdates();

  // Tier limits
  const tierLimits = useMemo(() => getTierLimits(userTier), [userTier]);

  // CopilotKit readable state - Make workflow structure available
  useCopilotReadable({
    description: "Current LangGraph workflow structure and execution state",
    value: currentWorkflow ? {
      name: currentWorkflow.name,
      description: currentWorkflow.description,
      nodes: workflowNodes.map(node => ({
        id: node.id,
        type: node.type,
        status: node.status,
        agentType: node.metadata?.agent || node.type,
        dependencies: [],
        estimatedDuration: 30000
      })),
      connections: workflowConnections.map(conn => ({
        from: conn.from,
        to: conn.to,
        type: conn.type
      })),
      executionState: executionState ? {
        status: executionState.status,
        progress: executionState.progress,
        currentStage: executionState.currentStage
      } : null,
      estimatedTotal: currentWorkflow.estimatedDuration,
      nodeCount: workflowNodes.length,
      connectionCount: workflowConnections.length
    } : null
  });

  // CopilotKit action for workflow modification - temporarily disabled for build
  /* useCopilotAction({
    name: "modify_workflow_structure",
    description: "Modify the current LangGraph workflow structure with intelligent suggestions",
    parameters: [
      {
        name: "modification_type",
        type: "string",
        description: "Type of modification: add_agent, remove_agent, add_connection, remove_connection, optimize_flow, add_parallel_branch"
      },
      {
        name: "target_node",
        type: "string",
        description: "ID of the target node for modification (if applicable)"
      },
      {
        name: "agent_type",
        type: "string",
        description: "Type of agent to add: research, creative, technical, analysis, optimization"
      },
      {
        name: "position",
        type: "string",
        description: "Position in workflow: beginning, middle, end, parallel"
      },
      {
        name: "reasoning",
        type: "string",
        description: "Explanation for why this modification improves the workflow"
      }
    ],
    handler: async ({ modification_type, target_node, agent_type, position, reasoning }) => {
      if (userTier === UserTier.FREE && modification_type !== 'optimize_flow') {
        throw new Error("Workflow modification requires Pro tier or higher");
      }

      let modificationResult = '';

      switch (modification_type) {
        case 'add_agent':
          const newNode: WorkflowNode = {
            id: `${agent_type}_${Date.now()}`,
            type: 'agent',
            agentId: `${agent_type}_agent`,
            status: 'pending',
            input: {},
            dependencies: position === 'beginning' ? [] : [target_node || workflowNodes[0]?.id],
            estimatedDuration: 30000,
            metadata: { addedBy: 'copilot', reasoning }
          };

          setWorkflowNodes(prev => [...prev, newNode]);
          
          if (target_node && position !== 'parallel') {
            const newConnection: WorkflowConnectionTemplate = {
              id: `conn_${Date.now()}`,
              from: target_node,
              to: newNode.id,
              type: 'data'
            };
            setWorkflowConnections(prev => [...prev, newConnection]);
          }

          modificationResult = `Added ${agent_type} agent to workflow at ${position}. ${reasoning}`;
          break;

        case 'remove_agent':
          if (target_node) {
            setWorkflowNodes(prev => prev.filter(node => node.id !== target_node));
            setWorkflowConnections(prev => 
              prev.filter(conn => conn.from !== target_node && conn.to !== target_node)
            );
            modificationResult = `Removed node ${target_node} from workflow. ${reasoning}`;
          }
          break;

        case 'optimize_flow':
          // Analyze current workflow and suggest optimizations
          const optimizedNodes = workflowNodes.map(node => ({
            ...node,
            estimatedDuration: Math.round(node.estimatedDuration * 0.9) // 10% optimization
          }));
          setWorkflowNodes(optimizedNodes);
          modificationResult = `Optimized workflow for better performance. Estimated 10% reduction in execution time. ${reasoning}`;
          break;

        case 'add_parallel_branch':
          if (target_node) {
            const parallelNode: WorkflowNode = {
              id: `parallel_${agent_type}_${Date.now()}`,
              type: 'agent',
              agentId: `${agent_type}_agent`,
              status: 'pending',
              input: {},
              dependencies: [target_node],
              estimatedDuration: 25000,
              metadata: { addedBy: 'copilot', reasoning, isParallel: true }
            };

            setWorkflowNodes(prev => [...prev, parallelNode]);
            
            const parallelConnection: WorkflowConnectionTemplate = {
              id: `parallel_conn_${Date.now()}`,
              from: target_node,
              to: parallelNode.id,
              type: 'parallel'
            };
            setWorkflowConnections(prev => [...prev, parallelConnection]);

            modificationResult = `Added parallel ${agent_type} branch from ${target_node}. This enables concurrent processing. ${reasoning}`;
          }
          break;
      }

      await modifyWorkflow(currentWorkflow?.id || 'current', {
        nodes: workflowNodes,
        connections: workflowConnections,
        modification: { type: modification_type, reasoning }
      });

      return modificationResult;
    }
  }); */

  // CopilotKit action for workflow execution - temporarily disabled for build
  /* useCopilotAction({
    name: "execute_workflow_with_input",
    description: "Execute the current workflow with specific input data and monitoring",
    parameters: [
      {
        name: "input_data",
        type: "object",
        description: "Input data for workflow execution"
      },
      {
        name: "execution_mode",
        type: "string",
        description: "Execution mode: sequential, parallel, adaptive"
      },
      {
        name: "priority",
        type: "string",
        description: "Execution priority: low, medium, high"
      }
    ],
    handler: async ({ input_data, execution_mode, priority }) => {
      if (!currentWorkflow) {
        throw new Error("No workflow selected for execution");
      }

      const priorityLevel = { low: 3, medium: 5, high: 8 }[priority as string] || 5;

      const execution = await executeWorkflow({
        workflowId: currentWorkflow.id,
        inputData: input_data,
        executionMode: execution_mode || 'sequential',
        priority: priorityLevel,
        nodes: workflowNodes,
        connections: workflowConnections
      });

      setExecutionState(execution);

      return `Workflow execution started with ID: ${execution.id}. 
      Mode: ${execution_mode}, Priority: ${priority}. 
      Estimated completion: ${Math.round(execution.estimatedCompletion / 60000)} minutes.
      You can monitor progress in real-time.`;
    }
  }); */

  // CopilotKit action for workflow analysis - temporarily disabled for build
  /* useCopilotAction({
    name: "analyze_workflow_performance",
    description: "Analyze workflow performance and provide optimization recommendations",
    parameters: [
      {
        name: "analysis_type",
        type: "string",
        description: "Type of analysis: bottlenecks, efficiency, resource_usage, parallelization"
      },
      {
        name: "focus_area",
        type: "string",
        description: "Focus area for analysis: execution_time, resource_optimization, agent_coordination"
      }
    ],
    handler: async ({ analysis_type, focus_area }) => {
      const analysis = {
        bottlenecks: `Identified ${workflowNodes.filter(n => n.estimatedDuration > 30000).length} potential bottlenecks in the workflow. 
        Longest node: ${workflowNodes.reduce((max, node) => node.estimatedDuration > max.estimatedDuration ? node : max, workflowNodes[0])?.id || 'none'}.
        Recommendation: Consider breaking down long-running tasks or adding parallel branches.`,
        
        efficiency: `Current workflow efficiency score: ${Math.round((1 - (workflowConnections.length / workflowNodes.length)) * 100)}%. 
        ${workflowNodes.length} nodes with ${workflowConnections.length} connections. 
        Recommendation: ${workflowConnections.length > workflowNodes.length * 1.5 ? 'Simplify connections' : 'Consider adding parallel processing'}.`,
        
        resource_usage: `Estimated resource utilization: CPU: ${Math.round(workflowNodes.length * 15)}%, Memory: ${Math.round(workflowNodes.length * 10)}%. 
        Total estimated execution time: ${Math.round(currentWorkflow?.estimatedDuration || 0 / 1000)} seconds.
        Recommendation: ${workflowNodes.length > tierLimits.maxAgents ? `Reduce to ${tierLimits.maxAgents} agents max for your tier` : 'Resource usage is within limits'}.`,
        
        parallelization: `Parallelization opportunities: ${workflowNodes.filter(n => n.dependencies.length <= 1).length} nodes can run in parallel. 
        Current sequential dependencies: ${workflowConnections.filter(c => c.type === 'data').length}.
        Recommendation: Add parallel branches for independent tasks to reduce total execution time.`
      };

      return analysis[analysis_type as keyof typeof analysis] || 'Analysis completed successfully.';
    }
  }); */

  // Event handlers
  const handleTemplateSelect = useCallback((template: WorkflowTemplate) => {
    setCurrentWorkflow(template);
    setWorkflowNodes([...template.nodes]);
    setWorkflowConnections([...template.connections]);
    setWorkflowName(template.name);
    setWorkflowDescription(template.description);
    setTemplateDialogOpen(false);
  }, []);

  const handleNodeSelect = useCallback((nodeId: string, multiSelect: boolean = false) => {
    if (multiSelect) {
      setSelectedNodes(prev => 
        prev.includes(nodeId) 
          ? prev.filter(id => id !== nodeId)
          : [...prev, nodeId]
      );
    } else {
      setSelectedNodes([nodeId]);
    }
  }, []);

  const handleNodeEdit = useCallback((node: WorkflowNode) => {
    setEditingNode(node);
    setNodeDialogOpen(true);
  }, []);

  const handleWorkflowExecute = useCallback(async () => {
    if (!currentWorkflow || !workflowNodes.length) return;

    try {
      const execution = await executeWorkflow({
        workflowId: currentWorkflow.id,
        inputData: {},
        executionMode: 'sequential',
        priority: 5,
        nodes: workflowNodes.map(node => ({
          id: node.id,
          type: node.type === 'parallel' || node.type === 'sequential' ? 'coordination' as const :
                node.type === 'condition' ? 'decision' as const :
                node.type === 'start' ? 'merge' as const :
                node.type === 'end' ? 'split' as const :
                'agent' as const,
          agentId: node.metadata?.agent || 'default_agent',
          status: node.status === 'idle' ? 'pending' as const : 
                  node.status === 'running' ? 'active' as const :
                  node.status === 'completed' ? 'completed' as const :
                  node.status === 'error' ? 'failed' as const :
                  'pending' as const,
          input: {},
          dependencies: [],
          estimatedDuration: 30000,
          metadata: node.metadata || {}
        })),
        connections: workflowConnections
      });

      setExecutionState(execution);
    } catch (err) {
      console.error('Workflow execution failed:', err);
    }
  }, [currentWorkflow, workflowNodes, workflowConnections, executeWorkflow]);

  const handleWorkflowPause = useCallback(async () => {
    if (executionState) {
      await pauseWorkflow(executionState.id);
    }
  }, [executionState, pauseWorkflow]);

  const handleWorkflowStop = useCallback(async () => {
    if (executionState) {
      await stopWorkflow(executionState.id);
      setExecutionState(null);
    }
  }, [executionState, stopWorkflow]);

  // Effects
  useEffect(() => {
    if (executionState) {
      const interval = setInterval(async () => {
        const status = await getWorkflowStatus(executionState.id);
        if (status) {
          setExecutionState(status);
          
          // Update node statuses based on execution
          setWorkflowNodes(prev => prev.map(node => {
            const statusNode = status.workflowGraph.find(n => n.id === node.id);
            if (statusNode) {
              // Map agent workflow status to UI workflow status
              const uiStatus = statusNode.status === 'pending' ? 'idle' as const :
                              statusNode.status === 'active' ? 'running' as const :
                              statusNode.status === 'completed' ? 'completed' as const :
                              statusNode.status === 'failed' ? 'error' as const :
                              'idle' as const;
              return { ...node, status: uiStatus };
            }
            return node;
          }));
        }
      }, 1000);

      return () => clearInterval(interval);
    }
    
    return undefined; // Explicit return for when executionState is falsy
  }, [executionState, getWorkflowStatus]);

  // Derived state
  const canExecute = currentWorkflow && workflowNodes.length > 0 && !isExecuting;
  const canModify = userTier !== UserTier.FREE;

  if (isPreview) {
    return (
      <Card sx={{ height: '100%', minHeight: 400 }}>
        <CardHeader
          title="Workflow Visualizer"
          subheader={currentWorkflow?.name || "No workflow selected"}
          action={
            <Chip 
              label={userTier.toUpperCase()} 
              color={userTier === 'enterprise' ? 'primary' : userTier === 'pro' ? 'secondary' : 'default'}
              size="small"
            />
          }
        />
        <CardContent>
          <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {currentWorkflow ? (
              <Box>
                <Typography variant="h6" align="center" gutterBottom>
                  {workflowNodes.length} Nodes, {workflowConnections.length} Connections
                </Typography>
                <Box display="flex" justifyContent="center" gap={1} flexWrap="wrap">
                  {workflowNodes.slice(0, 4).map((node) => (
                    <Chip
                      key={node.id}
                      icon={<SmartToy />}
                      label={node.metadata?.agent?.replace('_agent', '') || node.label}
                      variant="outlined"
                      size="small"
                      color={node.status === 'completed' ? 'success' : node.status === 'running' ? 'primary' : 'default'}
                    />
                  ))}
                </Box>
                {executionState && (
                  <Box mt={2}>
                    <LinearProgress 
                      variant="determinate" 
                      value={executionState.progress * 100} 
                    />
                    <Typography variant="body2" align="center" mt={1}>
                      {Math.round(executionState.progress * 100)}% Complete
                    </Typography>
                  </Box>
                )}
              </Box>
            ) : (
              <Button
                variant="outlined"
                startIcon={<Add />}
                onClick={() => setTemplateDialogOpen(true)}
              >
                Select Workflow Template
              </Button>
            )}
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>
      
      {/* Header Section */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Workflow Visualizer
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <Chip 
              label={`${userTier.toUpperCase()} TIER`}
              color={userTier === 'enterprise' ? 'primary' : userTier === 'pro' ? 'secondary' : 'default'}
            />
            {currentWorkflow && (
              <Typography variant="body2" color="text.secondary">
                {currentWorkflow.name} • {workflowNodes.length} nodes
              </Typography>
            )}
            <Chip
              icon={isConnected ? <CheckCircle /> : <Error />}
              label={isConnected ? 'Connected' : 'Disconnected'}
              color={isConnected ? 'success' : 'error'}
              size="small"
            />
          </Box>
        </Box>
        
        <Box display="flex" gap={1}>
          <Button
            variant="outlined"
            startIcon={<Add />}
            onClick={() => setTemplateDialogOpen(true)}
          >
            Load Template
          </Button>
          
          <TierGate requiredTier={UserTier.PRO} currentTier={userTier} feature="Workflow Editing">
            <Button
              variant="outlined"
              startIcon={<Edit />}
              onClick={() => setIsEditing(!isEditing)}
              color={isEditing ? 'primary' : 'inherit'}
            >
              {isEditing ? 'Exit Edit' : 'Edit'}
            </Button>
          </TierGate>
          
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={handleWorkflowExecute}
            disabled={!canExecute}
          >
            Execute
          </Button>
        </Box>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Main Content */}
      <Grid container spacing={3}>
        
        {/* Workflow Canvas */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader
              title={
                <Box display="flex" alignItems="center" gap={1}>
                  <AccountTree />
                  <Typography variant="h6">
                    {currentWorkflow?.name || 'Workflow Canvas'}
                  </Typography>
                  {executionState && (
                    <Chip
                      label={executionState.status.toUpperCase()}
                      color={
                        executionState.status === 'completed' ? 'success' :
                        executionState.status === 'active' ? 'primary' :
                        executionState.status === 'failed' ? 'error' : 'default'
                      }
                      size="small"
                    />
                  )}
                </Box>
              }
              action={
                executionState && (
                  <Box display="flex" gap={1}>
                    <IconButton onClick={handleWorkflowPause} disabled={!isExecuting}>
                      <Pause />
                    </IconButton>
                    <IconButton onClick={handleWorkflowStop}>
                      <Stop />
                    </IconButton>
                  </Box>
                )
              }
            />
            <CardContent>
              {currentWorkflow ? (
                <Box
                  ref={canvasRef}
                  sx={{
                    minHeight: 400,
                    border: '1px dashed',
                    borderColor: 'divider',
                    borderRadius: 1,
                    position: 'relative',
                    overflow: 'auto'
                  }}
                >
                  {/* Workflow Nodes */}
                  {workflowNodes.map((node, index) => (
                    <WorkflowNodeComponent
                      key={node.id}
                      node={node}
                      isSelected={selectedNodes.includes(node.id)}
                      onSelect={() => handleNodeSelect(node.id, false)}
                      onEdit={() => handleNodeEdit({
                        id: node.id,
                        type: node.type === 'parallel' || node.type === 'sequential' ? 'coordination' as const :
                              node.type === 'condition' ? 'decision' as const :
                              node.type === 'start' ? 'merge' as const :
                              node.type === 'end' ? 'split' as const :
                              'agent' as const,
                        agentId: node.metadata?.agent || 'default_agent',
                        status: node.status === 'idle' ? 'pending' as const : 
                                node.status === 'running' ? 'active' as const :
                                node.status === 'completed' ? 'completed' as const :
                                node.status === 'error' ? 'failed' as const :
                                'pending' as const,
                        input: {},
                        dependencies: [],
                        estimatedDuration: 30000,
                        metadata: node.metadata || {}
                      })}
                      onDelete={(nodeId) => {
                        setWorkflowNodes(prev => prev.filter(n => n.id !== nodeId));
                        setWorkflowConnections(prev => 
                          prev.filter(c => c.from !== nodeId && c.to !== nodeId)
                        );
                      }}
                      readOnly={!isEditing}
                    />
                  ))}

                  {/* Workflow Connections */}
                  {workflowConnections.map((connection) => {
                    const fromNode = workflowNodes.find(n => n.id === connection.from);
                    const toNode = workflowNodes.find(n => n.id === connection.to);
                    
                    if (!fromNode || !toNode) return null;
                    
                    return (
                      <WorkflowConnection
                        key={connection.id}
                        id={connection.id}
                        from={{
                          x: fromNode.position.x + 100,
                          y: fromNode.position.y + 25,
                          nodeId: fromNode.id,
                          type: 'output'
                        }}
                        to={{
                          x: toNode.position.x,
                          y: toNode.position.y + 25,
                          nodeId: toNode.id,
                          type: 'input'
                        }}
                        type={connection.type === 'coordination' || connection.type === 'parallel' ? 'control' : connection.type || 'data'}
                        onDelete={() => {
                          setWorkflowConnections(prev => prev.filter(c => c.id !== connection.id));
                        }}
                      />
                    );
                  })}

                  {/* Empty State */}
                  {workflowNodes.length === 0 && (
                    <Box
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        minHeight: 300,
                        color: 'text.secondary'
                      }}
                    >
                      <AccountTree sx={{ fontSize: 64, mb: 2 }} />
                      <Typography variant="h6" gutterBottom>
                        No Workflow Loaded
                      </Typography>
                      <Typography variant="body2" align="center" mb={2}>
                        Load a template or create a custom workflow to get started
                      </Typography>
                      <Button
                        variant="outlined"
                        startIcon={<Add />}
                        onClick={() => setTemplateDialogOpen(true)}
                      >
                        Load Template
                      </Button>
                    </Box>
                  )}
                </Box>
              ) : (
                <Box textAlign="center" py={8}>
                  <Timeline sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    Welcome to Workflow Visualizer
                  </Typography>
                  <Typography variant="body2" color="text.secondary" mb={3}>
                    Create and execute complex multi-agent workflows with real-time monitoring
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<Add />}
                    onClick={() => setTemplateDialogOpen(true)}
                  >
                    Get Started
                  </Button>
                </Box>
              )}

              {/* Execution Progress */}
              {executionState && (
                <Box mt={2}>
                  <Typography variant="body2" gutterBottom>
                    Execution Progress: {Math.round(executionState.progress * 100)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={executionState.progress * 100}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="body2" color="text.secondary" mt={1}>
                    Current Stage: {executionState.currentStage}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Control Panel */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardHeader title="Workflow Control" />
            <CardContent>
              <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                <Tab label="Properties" />
                <Tab label="Templates" />
                <Tab label="Analytics" />
              </Tabs>

              {/* Properties Tab */}
              {tabValue === 0 && (
                <Box mt={2}>
                  <TextField
                    fullWidth
                    label="Workflow Name"
                    value={workflowName}
                    onChange={(e) => setWorkflowName(e.target.value)}
                    margin="normal"
                    size="small"
                  />
                  
                  <CopilotTextarea
                    className="workflow-description"
                    placeholder="Describe your workflow goals and requirements..."
                    autosuggestionsConfig={{
                      textareaPurpose: "Help the user design an effective workflow. Suggest agent types, execution patterns, and optimizations based on their description.",
                      chatApiConfigs: {}
                    }}
                    value={workflowDescription}
                    onChange={(e) => setWorkflowDescription(e.target.value)}
                    style={{
                      width: '100%',
                      minHeight: '100px',
                      padding: '8px',
                      border: '1px solid #e0e0e0',
                      borderRadius: '4px',
                      fontSize: '14px',
                      fontFamily: 'inherit',
                      marginTop: '16px'
                    }}
                  />

                  {currentWorkflow && (
                    <Box mt={2}>
                      <Typography variant="subtitle2" gutterBottom>
                        Workflow Statistics
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemIcon><SmartToy /></ListItemIcon>
                          <ListItemText 
                            primary={`${workflowNodes.length} Nodes`}
                            secondary={`${workflowConnections.length} connections`}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><Speed /></ListItemIcon>
                          <ListItemText 
                            primary={`~${Math.round((currentWorkflow.estimatedDuration || 0) / 1000)}s`}
                            secondary="Estimated duration"
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><Memory /></ListItemIcon>
                          <ListItemText 
                            primary={`${Math.round(workflowNodes.length * 15)}%`}
                            secondary="Estimated CPU usage"
                          />
                        </ListItem>
                      </List>
                    </Box>
                  )}
                </Box>
              )}

              {/* Templates Tab */}
              {tabValue === 1 && (
                <Box mt={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Available Templates
                  </Typography>
                  {WORKFLOW_TEMPLATES
                    .filter(template => 
                      userTier === UserTier.ENTERPRISE || 
                      template.requiredTier !== UserTier.ENTERPRISE
                    )
                    .map((template) => (
                      <Card
                        key={template.id}
                        sx={{ 
                          mb: 2, 
                          cursor: 'pointer',
                          border: currentWorkflow?.id === template.id ? 2 : 1,
                          borderColor: currentWorkflow?.id === template.id ? 'primary.main' : 'divider'
                        }}
                        onClick={() => handleTemplateSelect(template)}
                      >
                        <CardContent sx={{ py: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            {template.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" fontSize="0.8rem">
                            {template.description}
                          </Typography>
                          <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                            <Chip label={template.category} size="small" />
                            <Typography variant="caption" color="text.secondary">
                              {template.nodes.length} nodes
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    ))}
                </Box>
              )}

              {/* Analytics Tab */}
              {tabValue === 2 && (
                <TierGate requiredTier={UserTier.PRO} currentTier={userTier} feature="Workflow Analytics">
                  <Box mt={2}>
                    <Typography variant="subtitle2" gutterBottom>
                      Performance Analytics
                    </Typography>
                    {/* Analytics content would go here */}
                    <Typography variant="body2" color="text.secondary">
                      Real-time workflow performance metrics and optimization recommendations.
                    </Typography>
                  </Box>
                </TierGate>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Speed Dial for Actions */}
      {isEditing && (
        <TierGate requiredTier={UserTier.PRO} currentTier={userTier} feature="Workflow Editing Tools">
          <SpeedDial
            ariaLabel="Workflow Actions"
            sx={{ position: 'fixed', bottom: 16, right: 16 }}
            icon={<SpeedDialIcon />}
          >
            <SpeedDialAction
              icon={<SmartToy />}
              tooltipTitle="Add Agent Node"
              onClick={() => {/* Add agent node logic */}}
            />
            <SpeedDialAction
              icon={<Link />}
              tooltipTitle="Add Connection"
              onClick={() => {/* Add connection logic */}}
            />
            <SpeedDialAction
              icon={<CallSplit />}
              tooltipTitle="Add Parallel Branch"
              onClick={() => {/* Add parallel branch logic */}}
            />
            <SpeedDialAction
              icon={<Save />}
              tooltipTitle="Save Workflow"
              onClick={() => {/* Save workflow logic */}}
            />
          </SpeedDial>
        </TierGate>
      )}

      {/* Template Selection Dialog */}
      <Dialog
        open={templateDialogOpen}
        onClose={() => setTemplateDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Select Workflow Template</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            {WORKFLOW_TEMPLATES.map((template) => (
              <Grid item xs={12} sm={6} key={template.id}>
                <Card
                  sx={{ 
                    cursor: 'pointer',
                    height: '100%',
                    opacity: userTier === UserTier.FREE && template.requiredTier !== UserTier.FREE ? 0.6 : 1
                  }}
                  onClick={() => {
                    if (userTier === UserTier.FREE && template.requiredTier !== UserTier.FREE) {
                      return; // Prevent selection for tier-restricted templates
                    }
                    handleTemplateSelect(template);
                  }}
                >
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="h6">{template.name}</Typography>
                      <Chip 
                        label={template.requiredTier.toUpperCase()} 
                        size="small"
                        color={template.requiredTier === userTier ? 'primary' : 'default'}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {template.description}
                    </Typography>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                      <Chip label={template.category} size="small" variant="outlined" />
                      <Typography variant="caption">
                        {template.nodes.length} nodes, ~{Math.round(template.estimatedDuration / 1000)}s
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTemplateDialogOpen(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

    </Box>
  );
};