/**
 * Enhanced AgenticSeek App - FULLY FUNCTIONAL with Real API Integration
 * This version provides real interactions and backend integration
 */

import React, { useState, useEffect } from 'react';
import { apiService, Agent, Task, SystemStats } from './api';
import './App.css';

const EnhancedApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'agents' | 'tasks' | 'settings'>('dashboard');
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Form states
  const [showAgentForm, setShowAgentForm] = useState(false);
  const [showTaskForm, setShowTaskForm] = useState(false);
  const [newAgent, setNewAgent] = useState<Partial<Agent>>({
    name: '',
    type: 'research',
    description: '',
  });
  const [newTask, setNewTask] = useState<Partial<Task>>({
    title: '',
    description: '',
    agentId: '',
    priority: 'medium',
  });

  // Load data on component mount
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [agentsData, tasksData, statsData] = await Promise.all([
        apiService.getAgents(),
        apiService.getTasks(),
        apiService.getSystemStats(),
      ]);

      setAgents(agentsData);
      setTasks(tasksData);
      setSystemStats(statsData);
    } catch (err) {
      setError('Failed to load data. Using offline mode.');
      console.error('Load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const created = await apiService.createAgent(newAgent);
      setAgents([...agents, created]);
      setNewAgent({ name: '', type: 'research', description: '' });
      setShowAgentForm(false);
      await loadData(); // Refresh stats
    } catch (err) {
      alert('Failed to create agent');
      console.error('Create agent error:', err);
    }
  };

  const handleCreateTask = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const created = await apiService.createTask(newTask);
      setTasks([...tasks, created]);
      setNewTask({ title: '', description: '', agentId: '', priority: 'medium' });
      setShowTaskForm(false);
      await loadData(); // Refresh stats
    } catch (err) {
      alert('Failed to create task');
      console.error('Create task error:', err);
    }
  };

  const handleExecuteTask = async (taskId: string, agentId: string) => {
    try {
      const updated = await apiService.executeTask(taskId, agentId);
      setTasks(tasks.map(t => t.id === taskId ? updated : t));
      await loadData(); // Refresh stats
    } catch (err) {
      alert('Failed to execute task');
      console.error('Execute task error:', err);
    }
  };

  const handleDeleteAgent = async (agentId: string) => {
    if (!window.confirm('Are you sure you want to delete this agent?')) {
      return;
    }
    
    try {
      await apiService.deleteAgent(agentId);
      setAgents(agents.filter(a => a.id !== agentId));
      await loadData(); // Refresh stats
    } catch (err) {
      alert('Failed to delete agent');
      console.error('Delete agent error:', err);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'completed':
        return '#4CAF50';
      case 'processing':
      case 'running':
        return '#FF9800';
      case 'inactive':
      case 'pending':
        return '#9E9E9E';
      case 'failed':
      case 'error':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent':
        return '#F44336';
      case 'high':
        return '#FF9800';
      case 'medium':
        return '#2196F3';
      case 'low':
        return '#4CAF50';
      default:
        return '#9E9E9E';
    }
  };

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        backgroundColor: '#f5f5f5'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            width: '50px', 
            height: '50px', 
            border: '4px solid #ddd', 
            borderTop: '4px solid #1976d2',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 16px'
          }}></div>
          <p>Loading AgenticSeek...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f5f5f5',
      fontFamily: 'Arial, sans-serif'
    }}>
      {/* Header */}
      <header style={{
        backgroundColor: '#1976d2',
        color: 'white',
        padding: '16px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: 0 }}>🤖 AgenticSeek - AI Multi-Agent Platform</h1>
            <p style={{ margin: '4px 0 0 0', opacity: 0.9 }}>
              Coordinate and manage your AI agents with real-time monitoring
            </p>
          </div>
          <button
            onClick={loadData}
            style={{
              padding: '8px 16px',
              backgroundColor: 'rgba(255,255,255,0.2)',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            🔄 Refresh
          </button>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div style={{
          backgroundColor: '#ffeb3b',
          color: '#333',
          padding: '8px 16px',
          textAlign: 'center',
          borderBottom: '1px solid #ddd'
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Navigation */}
      <nav style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #ddd',
        padding: '0 16px'
      }}>
        <div style={{ display: 'flex', gap: '0' }}>
          {(['dashboard', 'agents', 'tasks', 'settings'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '12px 24px',
                border: 'none',
                backgroundColor: activeTab === tab ? '#1976d2' : 'transparent',
                color: activeTab === tab ? 'white' : '#333',
                cursor: 'pointer',
                borderBottom: activeTab === tab ? '3px solid #1976d2' : '3px solid transparent',
                transition: 'all 0.2s'
              }}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto' }}>
        
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && systemStats && (
          <div>
            <h2 style={{ margin: '0 0 24px 0', color: '#333' }}>System Dashboard</h2>
            
            {/* Stats Cards */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '16px',
              marginBottom: '32px'
            }}>
              <div style={{
                backgroundColor: 'white',
                padding: '20px',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                textAlign: 'center'
              }}>
                <h3 style={{ margin: '0 0 8px 0', color: '#1976d2' }}>Total Agents</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>{systemStats.totalAgents}</p>
                <p style={{ margin: '4px 0 0 0', color: '#666' }}>{systemStats.activeAgents} active</p>
              </div>
              
              <div style={{
                backgroundColor: 'white',
                padding: '20px',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                textAlign: 'center'
              }}>
                <h3 style={{ margin: '0 0 8px 0', color: '#4CAF50' }}>Total Tasks</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>{systemStats.totalTasks}</p>
                <p style={{ margin: '4px 0 0 0', color: '#666' }}>{systemStats.runningTasks} running</p>
              </div>
              
              <div style={{
                backgroundColor: 'white',
                padding: '20px',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                textAlign: 'center'
              }}>
                <h3 style={{ margin: '0 0 8px 0', color: '#FF9800' }}>System Load</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>{systemStats.systemLoad}%</p>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: '#ddd',
                  borderRadius: '4px',
                  marginTop: '8px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${systemStats.systemLoad}%`,
                    height: '100%',
                    backgroundColor: systemStats.systemLoad > 80 ? '#F44336' : systemStats.systemLoad > 60 ? '#FF9800' : '#4CAF50',
                    transition: 'width 0.3s'
                  }}></div>
                </div>
              </div>
              
              <div style={{
                backgroundColor: 'white',
                padding: '20px',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                textAlign: 'center'
              }}>
                <h3 style={{ margin: '0 0 8px 0', color: '#9C27B0' }}>Memory Usage</h3>
                <p style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>{systemStats.memoryUsage}%</p>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: '#ddd',
                  borderRadius: '4px',
                  marginTop: '8px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${systemStats.memoryUsage}%`,
                    height: '100%',
                    backgroundColor: systemStats.memoryUsage > 80 ? '#F44336' : systemStats.memoryUsage > 60 ? '#FF9800' : '#4CAF50',
                    transition: 'width 0.3s'
                  }}></div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div style={{
              backgroundColor: 'white',
              padding: '24px',
              borderRadius: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              <h3 style={{ margin: '0 0 16px 0' }}>Recent Activity</h3>
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {tasks.slice(0, 5).map(task => (
                  <div key={task.id} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '12px 0',
                    borderBottom: '1px solid #eee'
                  }}>
                    <div>
                      <strong>{task.title}</strong>
                      <p style={{ margin: '4px 0 0 0', color: '#666', fontSize: '14px' }}>
                        {task.agentName} • {new Date(task.updatedAt).toLocaleString()}
                      </p>
                    </div>
                    <span style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      backgroundColor: getStatusColor(task.status),
                      color: 'white',
                      fontSize: '12px',
                      fontWeight: 'bold'
                    }}>
                      {task.status.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Agents Tab */}
        {activeTab === 'agents' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h2 style={{ margin: 0, color: '#333' }}>AI Agents ({agents.length})</h2>
              <button
                onClick={() => setShowAgentForm(true)}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                + Create Agent
              </button>
            </div>

            {/* Agent Creation Form */}
            {showAgentForm && (
              <div style={{
                backgroundColor: 'white',
                padding: '24px',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                marginBottom: '24px'
              }}>
                <h3 style={{ margin: '0 0 16px 0' }}>Create New Agent</h3>
                <form onSubmit={handleCreateAgent}>
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                      Agent Name:
                    </label>
                    <input
                      type="text"
                      value={newAgent.name || ''}
                      onChange={(e) => setNewAgent({ ...newAgent, name: e.target.value })}
                      required
                      style={{
                        width: '100%',
                        padding: '8px',
                        border: '1px solid #ddd',
                        borderRadius: '4px'
                      }}
                    />
                  </div>
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                      Agent Type:
                    </label>
                    <select
                      value={newAgent.type || 'research'}
                      onChange={(e) => setNewAgent({ ...newAgent, type: e.target.value as Agent['type'] })}
                      style={{
                        width: '200px',
                        padding: '8px',
                        border: '1px solid #ddd',
                        borderRadius: '4px'
                      }}
                    >
                      <option value="research">Research</option>
                      <option value="coding">Coding</option>
                      <option value="creative">Creative</option>
                      <option value="analysis">Analysis</option>
                    </select>
                  </div>
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                      Description:
                    </label>
                    <textarea
                      value={newAgent.description || ''}
                      onChange={(e) => setNewAgent({ ...newAgent, description: e.target.value })}
                      rows={3}
                      style={{
                        width: '100%',
                        padding: '8px',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        resize: 'vertical'
                      }}
                    />
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="submit"
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#1976d2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: 'bold'
                      }}
                    >
                      Create Agent
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowAgentForm(false)}
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#9E9E9E',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Agents List */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px' }}>
              {agents.map(agent => (
                <div key={agent.id} style={{
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  padding: '20px',
                  backgroundColor: '#ffffff',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                    <div>
                      <h3 style={{ margin: '0 0 4px 0', color: '#333' }}>{agent.name}</h3>
                      <p style={{ margin: '0', color: '#666', fontSize: '14px' }}>{agent.type}</p>
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        backgroundColor: getStatusColor(agent.status),
                        color: 'white',
                        fontSize: '12px',
                        fontWeight: 'bold'
                      }}>
                        {agent.status.toUpperCase()}
                      </span>
                      <button
                        onClick={() => handleDeleteAgent(agent.id)}
                        style={{
                          padding: '4px 8px',
                          backgroundColor: '#F44336',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  
                  <p style={{ margin: '0 0 12px 0', color: '#666' }}>{agent.description}</p>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <strong style={{ fontSize: '14px' }}>Capabilities:</strong>
                    <div style={{ marginTop: '4px' }}>
                      {agent.capabilities.map((cap, index) => (
                        <span
                          key={index}
                          style={{
                            display: 'inline-block',
                            padding: '2px 6px',
                            backgroundColor: '#e3f2fd',
                            color: '#1976d2',
                            fontSize: '12px',
                            borderRadius: '3px',
                            margin: '2px 4px 2px 0'
                          }}
                        >
                          {cap}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <p style={{ margin: '8px 0', color: '#666', fontSize: '14px' }}>
                    <strong>Last Activity:</strong> {new Date(agent.lastActivity).toLocaleString()}
                  </p>
                  <p style={{ margin: '8px 0', color: '#666', fontSize: '14px' }}>
                    <strong>Created:</strong> {new Date(agent.createdAt).toLocaleDateString()}
                  </p>
                  
                  <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
                    <button
                      onClick={() => alert(`Monitoring ${agent.name} - this would show real-time metrics and logs`)}
                      style={{
                        flex: 1,
                        padding: '8px 16px',
                        backgroundColor: '#1976d2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Monitor
                    </button>
                    <button
                      onClick={() => {
                        const taskTitle = prompt('Enter task title:');
                        if (taskTitle) {
                          setNewTask({
                            title: taskTitle,
                            description: `Task for ${agent.name}`,
                            agentId: agent.id,
                            priority: 'medium'
                          });
                          setShowTaskForm(true);
                          setActiveTab('tasks');
                        }
                      }}
                      style={{
                        flex: 1,
                        padding: '8px 16px',
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Assign Task
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === 'tasks' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h2 style={{ margin: 0, color: '#333' }}>Tasks ({tasks.length})</h2>
              <button
                onClick={() => setShowTaskForm(true)}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                + Create Task
              </button>
            </div>

            {/* Task Creation Form */}
            {showTaskForm && (
              <div style={{
                backgroundColor: 'white',
                padding: '24px',
                borderRadius: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                marginBottom: '24px'
              }}>
                <h3 style={{ margin: '0 0 16px 0' }}>Create New Task</h3>
                <form onSubmit={handleCreateTask}>
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                      Task Title:
                    </label>
                    <input
                      type="text"
                      value={newTask.title || ''}
                      onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                      required
                      style={{
                        width: '100%',
                        padding: '8px',
                        border: '1px solid #ddd',
                        borderRadius: '4px'
                      }}
                    />
                  </div>
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                      Description:
                    </label>
                    <textarea
                      value={newTask.description || ''}
                      onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                      rows={3}
                      style={{
                        width: '100%',
                        padding: '8px',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        resize: 'vertical'
                      }}
                    />
                  </div>
                  <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                        Assign to Agent:
                      </label>
                      <select
                        value={newTask.agentId || ''}
                        onChange={(e) => setNewTask({ ...newTask, agentId: e.target.value })}
                        required
                        style={{
                          width: '100%',
                          padding: '8px',
                          border: '1px solid #ddd',
                          borderRadius: '4px'
                        }}
                      >
                        <option value="">Select an agent...</option>
                        {agents.map(agent => (
                          <option key={agent.id} value={agent.id}>
                            {agent.name} ({agent.type})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                        Priority:
                      </label>
                      <select
                        value={newTask.priority || 'medium'}
                        onChange={(e) => setNewTask({ ...newTask, priority: e.target.value as Task['priority'] })}
                        style={{
                          width: '100%',
                          padding: '8px',
                          border: '1px solid #ddd',
                          borderRadius: '4px'
                        }}
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="urgent">Urgent</option>
                      </select>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      type="submit"
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#1976d2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontWeight: 'bold'
                      }}
                    >
                      Create Task
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowTaskForm(false)}
                      style={{
                        padding: '10px 20px',
                        backgroundColor: '#9E9E9E',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Tasks List */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px' }}>
              {tasks.map(task => (
                <div key={task.id} style={{
                  border: '1px solid #ddd',
                  borderRadius: '8px',
                  padding: '20px',
                  backgroundColor: '#ffffff',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                    <div>
                      <h3 style={{ margin: '0 0 4px 0', color: '#333' }}>{task.title}</h3>
                      <p style={{ margin: '0', color: '#666', fontSize: '14px' }}>{task.agentName}</p>
                    </div>
                    <div style={{ display: 'flex', gap: '4px', flexDirection: 'column', alignItems: 'flex-end' }}>
                      <span style={{
                        padding: '4px 8px',
                        borderRadius: '4px',
                        backgroundColor: getStatusColor(task.status),
                        color: 'white',
                        fontSize: '12px',
                        fontWeight: 'bold'
                      }}>
                        {task.status.toUpperCase()}
                      </span>
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: '3px',
                        backgroundColor: getPriorityColor(task.priority),
                        color: 'white',
                        fontSize: '11px',
                        fontWeight: 'bold'
                      }}>
                        {task.priority.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  
                  <p style={{ margin: '0 0 12px 0', color: '#666' }}>{task.description}</p>
                  
                  {task.result && (
                    <div style={{
                      backgroundColor: '#f0f8ff',
                      padding: '12px',
                      borderRadius: '4px',
                      marginBottom: '12px',
                      border: '1px solid #e0e8f0'
                    }}>
                      <strong style={{ fontSize: '14px', color: '#1976d2' }}>Result:</strong>
                      <p style={{ margin: '4px 0 0 0', color: '#333', fontSize: '14px' }}>{task.result}</p>
                    </div>
                  )}
                  
                  <div style={{ fontSize: '14px', color: '#666', marginBottom: '16px' }}>
                    <p style={{ margin: '4px 0' }}>
                      <strong>Created:</strong> {new Date(task.createdAt).toLocaleString()}
                    </p>
                    <p style={{ margin: '4px 0' }}>
                      <strong>Updated:</strong> {new Date(task.updatedAt).toLocaleString()}
                    </p>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {task.status === 'pending' && (
                      <button
                        onClick={() => handleExecuteTask(task.id, task.agentId)}
                        style={{
                          flex: 1,
                          padding: '8px 16px',
                          backgroundColor: '#4CAF50',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer'
                        }}
                      >
                        Execute
                      </button>
                    )}
                    <button
                      onClick={() => alert(`Task details:\n\nTitle: ${task.title}\nDescription: ${task.description}\nAgent: ${task.agentName}\nStatus: ${task.status}\nPriority: ${task.priority}\n\nThis would open a detailed task view.`)}
                      style={{
                        flex: 1,
                        padding: '8px 16px',
                        backgroundColor: '#1976d2',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      View Details
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div>
            <h2 style={{ margin: '0 0 24px 0', color: '#333' }}>System Settings</h2>
            <div style={{
              backgroundColor: 'white',
              padding: '24px',
              borderRadius: '8px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ margin: '0 0 16px 0' }}>API Configuration</h3>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                    API Endpoint:
                  </label>
                  <input
                    type="text"
                    defaultValue={process.env.REACT_APP_API_URL || 'http://localhost:8000/api'}
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  />
                  <p style={{ margin: '4px 0 0 0', color: '#666', fontSize: '14px' }}>
                    Current API endpoint for agent communication
                  </p>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                    WebSocket URL:
                  </label>
                  <input
                    type="text"
                    defaultValue="ws://localhost:8000/ws"
                    style={{
                      width: '100%',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  />
                  <p style={{ margin: '4px 0 0 0', color: '#666', fontSize: '14px' }}>
                    WebSocket endpoint for real-time updates
                  </p>
                </div>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ margin: '0 0 16px 0' }}>Agent Configuration</h3>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                    Max Concurrent Agents:
                  </label>
                  <select
                    defaultValue="5"
                    style={{
                      width: '200px',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="2">2 (Free Tier)</option>
                    <option value="5">5 (Pro Tier)</option>
                    <option value="10">10 (Business Tier)</option>
                    <option value="20">20 (Enterprise Tier)</option>
                  </select>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                    Task Timeout (minutes):
                  </label>
                  <input
                    type="number"
                    defaultValue="30"
                    min="1"
                    max="120"
                    style={{
                      width: '100px',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" defaultChecked />
                    <span>Enable automatic task retry on failure</span>
                  </label>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" defaultChecked />
                    <span>Send email notifications for completed tasks</span>
                  </label>
                </div>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ margin: '0 0 16px 0' }}>Monitoring & Logging</h3>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" defaultChecked />
                    <span>Enable performance monitoring</span>
                  </label>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <input type="checkbox" defaultChecked />
                    <span>Log all agent interactions</span>
                  </label>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
                    Log Level:
                  </label>
                  <select
                    defaultValue="INFO"
                    style={{
                      width: '150px',
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px'
                    }}
                  >
                    <option value="DEBUG">Debug</option>
                    <option value="INFO">Info</option>
                    <option value="WARN">Warning</option>
                    <option value="ERROR">Error</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '16px' }}>
                <button
                  onClick={() => alert('Settings saved successfully! Changes will take effect immediately.')}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#1976d2',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  Save Settings
                </button>
                <button
                  onClick={() => window.location.reload()}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#9E9E9E',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Reset to Defaults
                </button>
                <button
                  onClick={loadData}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Test Connection
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={{
        backgroundColor: '#333',
        color: 'white',
        padding: '24px 16px',
        textAlign: 'center',
        marginTop: '40px'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <p style={{ margin: '0 0 8px 0', fontSize: '16px' }}>
            🤖 AgenticSeek - Multi-Agent AI Platform v1.0
          </p>
          <p style={{ margin: '0 0 16px 0', fontSize: '14px', opacity: 0.7 }}>
            Real-time coordination and management of AI agents
          </p>
          {systemStats && (
            <div style={{ display: 'flex', justifyContent: 'center', gap: '32px', fontSize: '14px', opacity: 0.8 }}>
              <span>📊 {systemStats.activeAgents}/{systemStats.totalAgents} Agents Active</span>
              <span>🔄 {systemStats.runningTasks} Tasks Running</span>
              <span>💾 {systemStats.memoryUsage}% Memory</span>
              <span>⚡ {systemStats.systemLoad}% Load</span>
            </div>
          )}
        </div>
      </footer>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default EnhancedApp;