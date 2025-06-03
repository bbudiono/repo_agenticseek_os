/**
 * Agent System TypeScript Definitions
 * 
 * * Purpose: Type definitions for agent coordination and management system
 * * Issues & Complexity Summary: Comprehensive type system for multi-agent coordination
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~200
 *   - Core Algorithm Complexity: Medium
 *   - Dependencies: 1 New, 0 Mod
 *   - State Management Complexity: Medium
 *   - Novelty/Uncertainty Factor: Low
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 60%
 * * Problem Estimate (Inherent Problem Difficulty %): 55%
 * * Initial Code Complexity Estimate %: 60%
 * * Justification for Estimates: Well-defined type system with clear hierarchies
 * * Final Code Complexity (Actual %): TBD
 * * Overall Result Score (Success & Quality %): TBD
 * * Key Variances/Learnings: TBD
 * * Last Updated: 2025-06-03
 */

export enum AgentType {
  RESEARCH = 'research',
  CREATIVE = 'creative',
  TECHNICAL = 'technical',
  ANALYSIS = 'analysis',
  VIDEO = 'video',
  OPTIMIZATION = 'optimization',
  COMMUNICATION = 'communication',
  COORDINATION = 'coordination'
}

export enum AgentStatus {
  IDLE = 'idle',
  ACTIVE = 'active',
  BUSY = 'busy',
  PAUSED = 'paused',
  ERROR = 'error',
  OFFLINE = 'offline'
}

export enum CoordinationStatus {
  PENDING = 'pending',
  INITIALIZING = 'initializing',
  ACTIVE = 'active',
  PAUSED = 'paused',
  COMPLETING = 'completing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum TaskPriority {
  LOW = 1,
  MEDIUM = 5,
  HIGH = 8,
  CRITICAL = 10
}

export interface AgentCapability {
  id: string;
  name: string;
  description: string;
  proficiencyLevel: number; // 0-1
  requiredResources: string[];
  estimatedProcessingTime: number; // milliseconds
}

export interface PerformanceMetrics {
  responseTime: number; // milliseconds
  accuracy: number; // 0-1
  uptime: number; // 0-1
  throughput: number; // tasks per hour
  errorRate: number; // 0-1
  resourceUtilization: {
    cpu: number; // 0-1
    memory: number; // 0-1
    bandwidth: number; // 0-1
  };
  qualityScore: number; // 0-1
  successRate: number; // 0-1
  efficiency: number; // 0-1
  overallHealth: number; // 0-1
  bottlenecks?: string[];
}

export interface AgentStatusInfo {
  id: string;
  type: AgentType;
  name: string;
  status: AgentStatus;
  currentTask?: string;
  taskProgress?: number; // 0-1
  capabilities: AgentCapability[];
  performanceMetrics: PerformanceMetrics;
  workload: number; // 0-1
  estimatedAvailability?: number; // timestamp
  lastUpdate: number; // timestamp
  configuration: AgentConfiguration;
  metadata: Record<string, any>;
}

export interface AgentConfiguration {
  maxConcurrentTasks: number;
  specializations: string[];
  resourceLimits: {
    maxMemory: number; // MB
    maxCpuUsage: number; // 0-1
    maxExecutionTime: number; // milliseconds
  };
  optimization: {
    enableAppleSilicon: boolean;
    useNeuralEngine: boolean;
    useGpuAcceleration: boolean;
    priority: 'performance' | 'efficiency' | 'balanced';
  };
  communication: {
    enableInternalComms: boolean;
    enableExternalComms: boolean;
    communicationProtocol: string;
  };
}

export interface CoordinationRequest {
  id: string;
  taskDescription: string;
  requiredCapabilities: string[];
  priority: TaskPriority;
  maxAgents: number;
  coordinationMode: 'auto' | 'manual';
  deadline?: number; // timestamp
  constraints?: {
    budget?: number;
    resources?: string[];
    excludeAgents?: string[];
    preferredAgents?: string[];
  };
  metadata: Record<string, any>;
}

export interface CoordinationInstance {
  id: string;
  status: CoordinationStatus;
  taskDescription: string;
  assignedAgents: AgentStatusInfo[];
  progress: number; // 0-1
  currentStage: string;
  estimatedCompletion: number; // timestamp
  startTime: number; // timestamp
  endTime?: number; // timestamp
  result?: CoordinationResult;
  workflowGraph: WorkflowNode[];
  communicationLog: CommunicationMessage[];
  performanceSnapshot: PerformanceMetrics;
}

export interface CoordinationResult {
  success: boolean;
  output: any;
  qualityScore: number; // 0-1
  executionTime: number; // milliseconds
  resourcesUsed: {
    totalCpuTime: number;
    totalMemoryUsed: number;
    totalBandwidthUsed: number;
  };
  agentContributions: AgentContribution[];
  lessonsLearned: string[];
  recommendations: string[];
}

export interface AgentContribution {
  agentId: string;
  tasksPrimary: number;
  tasksSupporting: number;
  qualityScore: number; // 0-1
  timeContributed: number; // milliseconds
  outputGenerated: any;
  feedback: string;
}

export interface WorkflowNode {
  id: string;
  type: 'agent' | 'coordination' | 'decision' | 'merge' | 'split';
  agentId?: string;
  status: 'pending' | 'active' | 'completed' | 'failed';
  input: any;
  output?: any;
  dependencies: string[];
  estimatedDuration: number; // milliseconds
  actualDuration?: number; // milliseconds
  metadata: Record<string, any>;
}

export interface WorkflowConnection {
  id: string;
  from: string; // source node ID
  to: string; // target node ID
  type?: 'data' | 'control' | 'error' | 'success' | 'coordination' | 'parallel';
  status?: 'active' | 'inactive' | 'error' | 'success';
  animated?: boolean;
  label?: string;
  metadata?: Record<string, any>;
}

export interface CommunicationMessage {
  id: string;
  timestamp: number;
  fromAgent: string;
  toAgent: string | 'broadcast';
  type: 'coordination' | 'data' | 'status' | 'error' | 'internal';
  content: any;
  priority: TaskPriority;
  acknowledged: boolean;
  responseRequired: boolean;
  metadata: Record<string, any>;
}

export interface AgentTemplate {
  id: string;
  name: string;
  type: AgentType;
  description: string;
  defaultCapabilities: AgentCapability[];
  defaultConfiguration: AgentConfiguration;
  requiredTier: 'free' | 'pro' | 'enterprise';
  estimatedSetupTime: number; // milliseconds
  resourceRequirements: {
    minMemory: number; // MB
    minCpu: number; // cores
    specialHardware?: string[];
  };
}

export interface CoordinationTemplate {
  id: string;
  name: string;
  description: string;
  agentTypes: AgentType[];
  workflowPattern: WorkflowNode[];
  estimatedDuration: number; // milliseconds
  requiredTier: 'free' | 'pro' | 'enterprise';
  useCases: string[];
  successMetrics: string[];
}

// Video-specific types (Enterprise tier)
export interface VideoProject {
  id: string;
  concept: string;
  duration: number; // seconds
  style: string;
  status: 'created' | 'processing' | 'rendering' | 'completed' | 'failed' | 'paused' | 'stopped';
  progress: number; // 0-1
  previewUrl?: string;
  finalUrl?: string;
  metadata: {
    resolution: string;
    format: string;
    frameRate: number;
    estimatedFileSize: number; // MB
  };
  assignedAgents: string[];
  estimatedCompletion?: number; // timestamp
  createdAt: number; // timestamp
  updatedAt: number; // timestamp
}

// Apple Silicon optimization types
export interface HardwareMetrics {
  neuralEngineUsage: number; // 0-1
  gpuUsage: number; // 0-1
  cpuPerformanceCores: number;
  cpuEfficiencyCores: number;
  memoryPressure: 'normal' | 'warning' | 'urgent' | 'critical';
  thermalState: 'normal' | 'fair' | 'serious' | 'critical';
  powerState: 'normal' | 'low_power' | 'high_performance';
  timestamp: number;
}

export interface OptimizationSettings {
  level: 'basic' | 'advanced' | 'maximum';
  useNeuralEngine: boolean;
  useGpuAcceleration: boolean;
  cpuPriority: 'efficiency' | 'performance' | 'balanced';
  memoryOptimization: boolean;
  powerManagement: 'efficiency' | 'performance' | 'balanced';
  thermalManagement: boolean;
}

// Utility types
export type AgentEvent = {
  type: 'status_change' | 'task_assigned' | 'task_completed' | 'error' | 'communication';
  agentId: string;
  timestamp: number;
  data: any;
};

export type CoordinationEvent = {
  type: 'started' | 'paused' | 'resumed' | 'completed' | 'failed' | 'agent_added' | 'agent_removed';
  coordinationId: string;
  timestamp: number;
  data: any;
};

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    timestamp: number;
    requestId: string;
    executionTime: number;
  };
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

// Hook return types
export interface UseAgentCoordinationReturn {
  activeAgents: AgentStatusInfo[];
  availableAgents: AgentTemplate[];
  currentCoordination: CoordinationInstance | null;
  coordinationHistory: CoordinationInstance[];
  performanceMetrics: PerformanceMetrics;
  startCoordination: (request: Partial<CoordinationRequest>) => Promise<CoordinationInstance>;
  stopCoordination: (coordinationId: string) => Promise<void>;
  pauseCoordination: (coordinationId: string) => Promise<void>;
  resumeCoordination: (coordinationId: string) => Promise<void>;
  addAgent: (type: AgentType, reason?: string) => Promise<{ agent: AgentStatusInfo }>;
  removeAgent: (agentId: string, reason?: string) => Promise<void>;
  isLoading: boolean;
  error: string | null;
}

export interface UseRealTimeUpdatesReturn {
  isConnected: boolean;
  connectionError: string | null;
  lastUpdate: number;
  subscribe: (eventType: string, callback: (data: any) => void) => () => void;
  unsubscribe: (eventType: string) => void;
}

// Component prop types
export interface ComponentBaseProps {
  userTier: 'free' | 'pro' | 'enterprise';
  userId: string;
  isPreview?: boolean;
}