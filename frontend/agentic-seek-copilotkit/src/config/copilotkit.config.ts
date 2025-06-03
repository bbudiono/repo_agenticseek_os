/**
 * CopilotKit Configuration for AgenticSeek Multi-Agent System
 * 
 * * Purpose: Central configuration for CopilotKit integration with tier-aware features
 * * Issues & Complexity Summary: Complex tier management and real-time coordination configuration
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~150
 *   - Core Algorithm Complexity: Medium
 *   - Dependencies: 2 New, 1 Mod
 *   - State Management Complexity: Medium
 *   - Novelty/Uncertainty Factor: Medium
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 70%
 * * Problem Estimate (Inherent Problem Difficulty %): 65%
 * * Initial Code Complexity Estimate %: 70%
 * * Justification for Estimates: Comprehensive configuration with tier validation and feature gates
 * * Final Code Complexity (Actual %): TBD
 * * Overall Result Score (Success & Quality %): TBD
 * * Key Variances/Learnings: TBD
 * * Last Updated: 2025-06-03
 */

export interface CopilotKitConfig {
  apiUrl: string;
  wsUrl: string;
  publicApiKey?: string;
  enableRealTimeUpdates: boolean;
  enableTierValidation: boolean;
  defaultTier: UserTier;
  retryConfig: RetryConfig;
  performanceConfig: PerformanceConfig;
  featureFlags: FeatureFlags;
}

export interface RetryConfig {
  maxAttempts: number;
  retryDelay: number;
  exponentialBackoff: boolean;
  maxRetryDelay: number;
}

export interface PerformanceConfig {
  updateInterval: number;
  metricsBufferSize: number;
  chartUpdateInterval: number;
  virtualScrollThreshold: number;
  lazyLoadingThreshold: number;
}

export interface FeatureFlags {
  enableVideoGeneration: boolean;
  enableAppleSiliconOptimization: boolean;
  enableAdvancedWorkflows: boolean;
  enableAgentCommunicationFeed: boolean;
  enableRealTimeMetrics: boolean;
  enablePerformanceMonitoring: boolean;
  enableErrorReporting: boolean;
  enableAnimations: boolean;
}

export enum UserTier {
  FREE = 'free',
  PRO = 'pro',
  ENTERPRISE = 'enterprise'
}

export interface TierLimits {
  maxAgents: number;
  maxConcurrentWorkflows: number;
  videoGeneration: boolean;
  advancedOptimization: boolean;
  internalCommunications: boolean;
  realTimeMetrics: boolean;
  customWorkflows: boolean;
  apiRateLimit: number;
  storageLimit: number;
  supportLevel: 'community' | 'email' | 'priority';
}

export const TIER_LIMITS: Record<UserTier, TierLimits> = {
  [UserTier.FREE]: {
    maxAgents: 2,
    maxConcurrentWorkflows: 1,
    videoGeneration: false,
    advancedOptimization: false,
    internalCommunications: false,
    realTimeMetrics: false,
    customWorkflows: false,
    apiRateLimit: 100, // requests per hour
    storageLimit: 100, // MB
    supportLevel: 'community'
  },
  [UserTier.PRO]: {
    maxAgents: 5,
    maxConcurrentWorkflows: 3,
    videoGeneration: false,
    advancedOptimization: true,
    internalCommunications: false,
    realTimeMetrics: true,
    customWorkflows: true,
    apiRateLimit: 1000, // requests per hour
    storageLimit: 1000, // MB
    supportLevel: 'email'
  },
  [UserTier.ENTERPRISE]: {
    maxAgents: 20,
    maxConcurrentWorkflows: 10,
    videoGeneration: true,
    advancedOptimization: true,
    internalCommunications: true,
    realTimeMetrics: true,
    customWorkflows: true,
    apiRateLimit: 10000, // requests per hour
    storageLimit: 10000, // MB
    supportLevel: 'priority'
  }
};

export const DEFAULT_COPILOTKIT_CONFIG: CopilotKitConfig = {
  apiUrl: process.env.REACT_APP_COPILOTKIT_API_URL || 'http://localhost:8000/api/copilotkit',
  wsUrl: process.env.REACT_APP_COPILOTKIT_WS_URL || 'ws://localhost:8000/api/copilotkit/ws',
  ...(process.env.REACT_APP_COPILOTKIT_PUBLIC_API_KEY && { publicApiKey: process.env.REACT_APP_COPILOTKIT_PUBLIC_API_KEY }),
  enableRealTimeUpdates: process.env.REACT_APP_ENABLE_REAL_TIME_METRICS === 'true',
  enableTierValidation: process.env.REACT_APP_ENABLE_TIER_VALIDATION === 'true',
  defaultTier: (process.env.REACT_APP_DEFAULT_TIER as UserTier) || UserTier.FREE,
  retryConfig: {
    maxAttempts: 3,
    retryDelay: 1000,
    exponentialBackoff: true,
    maxRetryDelay: 10000
  },
  performanceConfig: {
    updateInterval: parseInt(process.env.REACT_APP_AGENT_UPDATE_INTERVAL || '5000'),
    metricsBufferSize: parseInt(process.env.REACT_APP_METRICS_BUFFER_SIZE || '100'),
    chartUpdateInterval: parseInt(process.env.REACT_APP_CHART_UPDATE_INTERVAL || '1000'),
    virtualScrollThreshold: parseInt(process.env.REACT_APP_VIRTUALIZATION_THRESHOLD || '50'),
    lazyLoadingThreshold: parseInt(process.env.REACT_APP_LAZY_LOADING_THRESHOLD || '10')
  },
  featureFlags: {
    enableVideoGeneration: process.env.REACT_APP_ENABLE_VIDEO_GENERATION === 'true',
    enableAppleSiliconOptimization: process.env.REACT_APP_ENABLE_APPLE_SILICON_OPTIMIZATION === 'true',
    enableAdvancedWorkflows: process.env.REACT_APP_ENABLE_ADVANCED_WORKFLOWS === 'true',
    enableAgentCommunicationFeed: process.env.REACT_APP_ENABLE_AGENT_COMMUNICATION_FEED === 'true',
    enableRealTimeMetrics: process.env.REACT_APP_ENABLE_REAL_TIME_METRICS === 'true',
    enablePerformanceMonitoring: process.env.REACT_APP_ENABLE_PERFORMANCE_MONITORING === 'true',
    enableErrorReporting: process.env.REACT_APP_ENABLE_ERROR_REPORTING === 'true',
    enableAnimations: process.env.REACT_APP_ENABLE_ANIMATIONS === 'true'
  }
};

export const COPILOT_INSTRUCTIONS = {
  base: `You are coordinating a multi-agent AI system called AgenticSeek. Help users accomplish complex tasks by coordinating specialized agents including research, creative, technical, and video generation agents. Always respect tier limitations and provide clear guidance on available features.`,
  
  free: `The user has FREE tier access with limited features:
  - Maximum 2 agents
  - 1 concurrent workflow
  - Basic optimization only
  - Community support
  Help them make the most of these limitations while suggesting Pro/Enterprise upgrades when appropriate.`,
  
  pro: `The user has PRO tier access with enhanced features:
  - Maximum 5 agents
  - 3 concurrent workflows
  - Advanced optimization
  - Custom workflows
  - Real-time metrics
  - Email support
  Guide them to leverage these advanced capabilities effectively.`,
  
  enterprise: `The user has ENTERPRISE tier access with full features:
  - Maximum 20 agents
  - 10 concurrent workflows
  - Video generation
  - Advanced optimization
  - Internal communications
  - Custom workflows
  - Priority support
  Help them maximize the full potential of the multi-agent system.`
};

export const COPILOT_LABELS = {
  title: "AgenticSeek Multi-Agent Coordinator",
  initial: "How can I help coordinate your AI agents today?",
  placeholder: "Describe a task for your agents to coordinate on...",
  
  // Action-specific labels
  agentCoordination: "Agent Coordination",
  workflowManagement: "Workflow Management", 
  videoGeneration: "Video Generation",
  hardwareOptimization: "Hardware Optimization",
  performanceMonitoring: "Performance Monitoring",
  
  // Error messages
  tierLimitReached: "This feature requires a higher tier subscription",
  maxAgentsReached: "Maximum agent limit reached for your tier",
  upgradeRequired: "Upgrade to access this feature",
  
  // Success messages
  agentsCoordinated: "Agents successfully coordinated",
  workflowExecuted: "Workflow execution started",
  optimizationApplied: "Hardware optimization applied",
  
  // Tier-specific prompts
  upgradeToProPrompt: "Upgrade to Pro for more agents and advanced workflows",
  upgradeToEnterprisePrompt: "Upgrade to Enterprise for video generation and unlimited access"
};

export const ACTION_CONFIGURATIONS = {
  coordinateAgents: {
    name: "coordinate_agents",
    description: "Coordinate multiple AI agents for complex task execution",
    parameters: [
      {
        name: "task_description",
        type: "string" as const,
        description: "Description of the task to coordinate across agents",
        required: true
      },
      {
        name: "agent_preferences",
        type: "object" as const,
        description: "Preferred agent types and configurations",
        required: false
      },
      {
        name: "priority_level",
        type: "number" as const,
        description: "Task priority from 1-10",
        required: false
      }
    ]
  },
  
  modifyWorkflow: {
    name: "modify_workflow",
    description: "Modify LangGraph workflow structure (Pro tier and above)",
    parameters: [
      {
        name: "modification_type",
        type: "string" as const,
        description: "Type of modification: add_agent, remove_agent, change_flow",
        required: true
      },
      {
        name: "details",
        type: "object" as const,
        description: "Modification details",
        required: true
      }
    ]
  },
  
  generateVideo: {
    name: "generate_video_content",
    description: "Generate video content using specialized AI agents (Enterprise tier required)",
    parameters: [
      {
        name: "concept",
        type: "string" as const,
        description: "Video concept and description",
        required: true
      },
      {
        name: "duration",
        type: "number" as const,
        description: "Video duration in seconds",
        required: true
      },
      {
        name: "style",
        type: "string" as const,
        description: "Visual style preference",
        required: false
      }
    ]
  },
  
  optimizeHardware: {
    name: "optimize_apple_silicon",
    description: "Optimize Apple Silicon hardware for current workload",
    parameters: [
      {
        name: "optimization_type",
        type: "string" as const,
        description: "Type of optimization: performance, efficiency, or balanced",
        required: true
      },
      {
        name: "workload_focus",
        type: "string" as const,
        description: "Focus area: agent_coordination, video_generation, or general",
        required: false
      }
    ]
  }
};

// Utility functions
export const getTierLimits = (tier: UserTier | string | null | undefined): TierLimits => {
  // Handle invalid inputs by falling back to FREE tier
  if (!tier || !TIER_LIMITS[tier as UserTier]) {
    return TIER_LIMITS[UserTier.FREE];
  }
  return TIER_LIMITS[tier as UserTier];
};

export const canAccessFeature = (tier: UserTier, feature: keyof TierLimits): boolean => {
  const limits = getTierLimits(tier);
  return Boolean(limits[feature]);
};

export const getUpgradeMessage = (currentTier: UserTier, requiredFeature: string): string => {
  switch (currentTier) {
    case UserTier.FREE:
      return `${requiredFeature} requires Pro or Enterprise tier. Upgrade to unlock advanced features.`;
    case UserTier.PRO:
      return `${requiredFeature} requires Enterprise tier. Upgrade for full access to all features.`;
    default:
      return `You have access to all features.`;
  }
};

export const getCopilotInstructions = (tier: UserTier): string => {
  return `${COPILOT_INSTRUCTIONS.base}\n\n${COPILOT_INSTRUCTIONS[tier]}`;
};

export const formatTierDisplay = (tier: UserTier): string => {
  return tier.toUpperCase();
};

export const isFeatureEnabled = (featureFlag: keyof FeatureFlags): boolean => {
  return DEFAULT_COPILOTKIT_CONFIG.featureFlags[featureFlag];
};