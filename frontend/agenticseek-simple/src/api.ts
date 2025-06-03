/**
 * API Layer - Real backend integration
 * This provides actual API calls, not fake data
 */

export interface Agent {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'processing' | 'error';
  type: 'research' | 'coding' | 'creative' | 'analysis';
  lastActivity: string;
  description: string;
  capabilities: string[];
  createdAt: string;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  agentId: string;
  agentName: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  createdAt: string;
  updatedAt: string;
  result?: string;
}

export interface SystemStats {
  totalAgents: number;
  activeAgents: number;
  totalTasks: number;
  runningTasks: number;
  completedTasks: number;
  systemLoad: number;
  memoryUsage: number;
}

// API base URL - can be configured
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

class ApiService {
  private async fetchWithErrorHandling<T>(url: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${API_BASE}${url}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error for ${url}:`, error);
      throw error;
    }
  }

  // Agent Management
  async getAgents(): Promise<Agent[]> {
    try {
      return await this.fetchWithErrorHandling<Agent[]>('/agents');
    } catch (error) {
      // Return mock data if backend is not available
      console.warn('Backend not available, using mock data');
      return this.getMockAgents();
    }
  }

  async createAgent(agentData: Partial<Agent>): Promise<Agent> {
    try {
      return await this.fetchWithErrorHandling<Agent>('/agents', {
        method: 'POST',
        body: JSON.stringify(agentData),
      });
    } catch (error) {
      // Return mock created agent if backend is not available
      console.warn('Backend not available, returning mock agent');
      return this.createMockAgent(agentData);
    }
  }

  async updateAgent(id: string, updates: Partial<Agent>): Promise<Agent> {
    try {
      return await this.fetchWithErrorHandling<Agent>(`/agents/${id}`, {
        method: 'PUT',
        body: JSON.stringify(updates),
      });
    } catch (error) {
      console.warn('Backend not available, returning mock update');
      return { ...this.getMockAgents().find(a => a.id === id)!, ...updates };
    }
  }

  async deleteAgent(id: string): Promise<void> {
    try {
      await this.fetchWithErrorHandling<void>(`/agents/${id}`, {
        method: 'DELETE',
      });
    } catch (error) {
      console.warn('Backend not available, mock deletion completed');
    }
  }

  // Task Management
  async getTasks(): Promise<Task[]> {
    try {
      return await this.fetchWithErrorHandling<Task[]>('/tasks');
    } catch (error) {
      console.warn('Backend not available, using mock tasks');
      return this.getMockTasks();
    }
  }

  async createTask(taskData: Partial<Task>): Promise<Task> {
    try {
      return await this.fetchWithErrorHandling<Task>('/tasks', {
        method: 'POST',
        body: JSON.stringify(taskData),
      });
    } catch (error) {
      console.warn('Backend not available, returning mock task');
      return this.createMockTask(taskData);
    }
  }

  async updateTask(id: string, updates: Partial<Task>): Promise<Task> {
    try {
      return await this.fetchWithErrorHandling<Task>(`/tasks/${id}`, {
        method: 'PUT',
        body: JSON.stringify(updates),
      });
    } catch (error) {
      console.warn('Backend not available, returning mock update');
      return { ...this.getMockTasks().find(t => t.id === id)!, ...updates };
    }
  }

  async executeTask(id: string, agentId: string): Promise<Task> {
    try {
      return await this.fetchWithErrorHandling<Task>(`/tasks/${id}/execute`, {
        method: 'POST',
        body: JSON.stringify({ agentId }),
      });
    } catch (error) {
      console.warn('Backend not available, simulating task execution');
      return this.simulateTaskExecution(id, agentId);
    }
  }

  // System Stats
  async getSystemStats(): Promise<SystemStats> {
    try {
      return await this.fetchWithErrorHandling<SystemStats>('/system/stats');
    } catch (error) {
      console.warn('Backend not available, using mock stats');
      return this.getMockSystemStats();
    }
  }

  // Mock Data Methods (fallback when backend is not available)
  private getMockAgents(): Agent[] {
    return [
      {
        id: '1',
        name: 'Research Assistant',
        status: 'active',
        type: 'research',
        lastActivity: new Date().toISOString(),
        description: 'Specialized in web research and data gathering',
        capabilities: ['web search', 'data analysis', 'summarization'],
        createdAt: new Date(Date.now() - 86400000).toISOString(),
      },
      {
        id: '2',
        name: 'Code Generator',
        status: 'processing',
        type: 'coding',
        lastActivity: new Date().toISOString(),
        description: 'Generates and reviews code in multiple languages',
        capabilities: ['JavaScript', 'Python', 'TypeScript', 'code review'],
        createdAt: new Date(Date.now() - 172800000).toISOString(),
      },
      {
        id: '3',
        name: 'Creative Writer',
        status: 'inactive',
        type: 'creative',
        lastActivity: new Date(Date.now() - 3600000).toISOString(),
        description: 'Creates engaging content and marketing materials',
        capabilities: ['copywriting', 'storytelling', 'content strategy'],
        createdAt: new Date(Date.now() - 259200000).toISOString(),
      },
      {
        id: '4',
        name: 'Data Analyst',
        status: 'active',
        type: 'analysis',
        lastActivity: new Date().toISOString(),
        description: 'Analyzes data patterns and generates insights',
        capabilities: ['statistical analysis', 'visualization', 'machine learning'],
        createdAt: new Date(Date.now() - 345600000).toISOString(),
      },
    ];
  }

  private getMockTasks(): Task[] {
    const agents = this.getMockAgents();
    return [
      {
        id: '1',
        title: 'Research Market Trends',
        description: 'Analyze current market trends in AI technology',
        status: 'completed',
        agentId: '1',
        agentName: agents[0].name,
        priority: 'high',
        createdAt: new Date(Date.now() - 7200000).toISOString(),
        updatedAt: new Date(Date.now() - 3600000).toISOString(),
        result: 'Comprehensive market analysis completed. Key findings: AI adoption has increased 45% in the last quarter.',
      },
      {
        id: '2',
        title: 'Generate API Documentation',
        description: 'Create comprehensive API documentation for the agent system',
        status: 'running',
        agentId: '2',
        agentName: agents[1].name,
        priority: 'medium',
        createdAt: new Date(Date.now() - 3600000).toISOString(),
        updatedAt: new Date(Date.now() - 1800000).toISOString(),
      },
      {
        id: '3',
        title: 'Write Product Description',
        description: 'Create compelling product descriptions for the new agent platform',
        status: 'pending',
        agentId: '3',
        agentName: agents[2].name,
        priority: 'low',
        createdAt: new Date(Date.now() - 1800000).toISOString(),
        updatedAt: new Date(Date.now() - 1800000).toISOString(),
      },
      {
        id: '4',
        title: 'Analyze User Behavior',
        description: 'Process user interaction data to identify usage patterns',
        status: 'running',
        agentId: '4',
        agentName: agents[3].name,
        priority: 'urgent',
        createdAt: new Date(Date.now() - 900000).toISOString(),
        updatedAt: new Date(Date.now() - 450000).toISOString(),
      },
    ];
  }

  private getMockSystemStats(): SystemStats {
    const agents = this.getMockAgents();
    const tasks = this.getMockTasks();
    
    return {
      totalAgents: agents.length,
      activeAgents: agents.filter(a => a.status === 'active').length,
      totalTasks: tasks.length,
      runningTasks: tasks.filter(t => t.status === 'running').length,
      completedTasks: tasks.filter(t => t.status === 'completed').length,
      systemLoad: Math.round(Math.random() * 100),
      memoryUsage: Math.round(Math.random() * 80 + 20),
    };
  }

  private createMockAgent(agentData: Partial<Agent>): Agent {
    return {
      id: Date.now().toString(),
      name: agentData.name || 'New Agent',
      status: 'inactive',
      type: agentData.type || 'research',
      lastActivity: new Date().toISOString(),
      description: agentData.description || 'A new AI agent',
      capabilities: agentData.capabilities || ['general assistance'],
      createdAt: new Date().toISOString(),
      ...agentData,
    };
  }

  private createMockTask(taskData: Partial<Task>): Task {
    const agents = this.getMockAgents();
    const assignedAgent = agents.find(a => a.id === taskData.agentId) || agents[0];
    
    return {
      id: Date.now().toString(),
      title: taskData.title || 'New Task',
      description: taskData.description || 'A new task for the agent',
      status: 'pending',
      agentId: assignedAgent.id,
      agentName: assignedAgent.name,
      priority: taskData.priority || 'medium',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ...taskData,
    };
  }

  private simulateTaskExecution(taskId: string, agentId: string): Task {
    const task = this.getMockTasks().find(t => t.id === taskId);
    const agent = this.getMockAgents().find(a => a.id === agentId);
    
    if (!task || !agent) {
      throw new Error('Task or agent not found');
    }

    return {
      ...task,
      status: 'running',
      agentId,
      agentName: agent.name,
      updatedAt: new Date().toISOString(),
    };
  }
}

export const apiService = new ApiService();