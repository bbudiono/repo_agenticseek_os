/**
 * Agent Communication Feed Component
 * 
 * * Purpose: Real-time feed of agent communications with filtering and interaction
 * * Issues & Complexity Summary: Complex real-time data management with WebSocket integration
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~350
 *   - Core Algorithm Complexity: High
 *   - Dependencies: 4 New, 3 Mod
 *   - State Management Complexity: High
 *   - Novelty/Uncertainty Factor: Medium
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 80%
 * * Problem Estimate (Inherent Problem Difficulty %): 85%
 * * Initial Code Complexity Estimate %: 80%
 * * Justification for Estimates: Real-time data streams with complex filtering and interaction
 * * Final Code Complexity (Actual %): 82%
 * * Overall Result Score (Success & Quality %): 90%
 * * Key Variances/Learnings: WebSocket message handling more complex than anticipated
 * * Last Updated: 2025-06-03
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  Typography,
  Box,
  Chip,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  TextField,
  MenuItem,
  IconButton,
  Badge,
  Tooltip,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Paper,
  Divider
} from '@mui/material';
import {
  SmartToy as AgentIcon,
  Send as SendIcon,
  FilterList as FilterIcon,
  Clear as ClearIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandIcon,
  KeyboardArrowUp as CollapseIcon
} from '@mui/icons-material';
import { useWebSocket } from '../hooks/useWebSocket';
import { UserTier } from '../config/copilotkit.config';

interface AgentMessage {
  id: string;
  agentId: string;
  agentName: string;
  message: string;
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'success' | 'coordination';
  targetAgent?: string;
  metadata?: {
    taskId?: string;
    workflowId?: string;
    performance?: number;
  };
}

interface AgentCommunicationFeedProps {
  userTier: UserTier;
  userId: string;
  isPreview?: boolean;
}

export const AgentCommunicationFeed: React.FC<AgentCommunicationFeedProps> = ({
  userTier,
  userId,
  isPreview = false
}) => {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [isExpanded, setIsExpanded] = useState(!isPreview);
  const [newMessageDialog, setNewMessageDialog] = useState(false);
  const [newMessage, setNewMessage] = useState('');
  const [selectedAgent, setSelectedAgent] = useState('');

  const { lastMessage, sendMessage, isConnected } = useWebSocket('/api/copilotkit/ws');

  // Get unique agents
  const agents = useMemo(() => {
    const agentSet = new Set(messages.map(m => m.agentId));
    return Array.from(agentSet).map(agentId => {
      const message = messages.find(m => m.agentId === agentId);
      return {
        id: agentId,
        name: message?.agentName || agentId,
        lastActive: message?.timestamp || ''
      };
    });
  }, [messages]);

  // Filtered messages
  const filteredMessages = useMemo(() => {
    let filtered = messages;

    if (filter !== 'all') {
      filtered = filtered.filter(m => m.type === filter);
    }

    if (searchTerm) {
      filtered = filtered.filter(m =>
        m.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        m.agentName.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    return filtered.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [messages, filter, searchTerm]);

  // Load initial messages
  useEffect(() => {
    const loadMessages = async () => {
      try {
        const response = await fetch('/api/copilotkit/agent-communications', {
          headers: {
            'User-ID': userId,
            'User-Tier': userTier,
            'Content-Type': 'application/json'
          }
        });

        if (response.ok) {
          const data = await response.json();
          setMessages(data.messages || []);
        }
      } catch (error) {
        console.error('Failed to load agent communications:', error);
      }
    };

    loadMessages();
  }, [userId, userTier]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'agent_communication') {
      const newMsg: AgentMessage = {
        id: lastMessage.id || `msg_${Date.now()}`,
        agentId: lastMessage.payload.agentId,
        agentName: lastMessage.payload.agentName,
        message: lastMessage.payload.message,
        timestamp: lastMessage.timestamp,
        type: lastMessage.payload.messageType || 'info',
        targetAgent: lastMessage.payload.targetAgent,
        metadata: lastMessage.payload.metadata
      };

      setMessages(prev => [newMsg, ...prev.slice(0, 99)]); // Keep last 100 messages
    }
  }, [lastMessage]);

  const handleSendMessage = async () => {
    if (!newMessage.trim() || !selectedAgent) return;

    const message: AgentMessage = {
      id: `user_msg_${Date.now()}`,
      agentId: 'user',
      agentName: 'User',
      message: newMessage,
      timestamp: new Date().toISOString(),
      type: 'info',
      targetAgent: selectedAgent
    };

    // Send via WebSocket
    sendMessage({
      type: 'user_message',
      payload: {
        targetAgent: selectedAgent,
        message: newMessage,
        userId
      },
      timestamp: new Date().toISOString()
    });

    // Add to local state
    setMessages(prev => [message, ...prev]);

    // Clear form
    setNewMessage('');
    setNewMessageDialog(false);
  };

  const getMessageTypeColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'success':
        return 'success';
      case 'coordination':
        return 'secondary';
      default:
        return 'primary';
    }
  };

  const getAgentAvatar = (agentId: string) => {
    const colors = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2', '#d32f2f'];
    const colorIndex = agentId.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
    
    return (
      <Avatar sx={{ bgcolor: colors[colorIndex], width: 32, height: 32 }}>
        <AgentIcon fontSize="small" />
      </Avatar>
    );
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  };

  if (isPreview && !isExpanded) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardHeader
          title="Agent Communications"
          subheader={`${messages.length} recent messages`}
          action={
            <Box display="flex" alignItems="center" gap={1}>
              <Badge color="success" variant="dot" invisible={!isConnected}>
                <AgentIcon />
              </Badge>
              <IconButton size="small" onClick={() => setIsExpanded(true)}>
                <ExpandIcon />
              </IconButton>
            </Box>
          }
        />
        <CardContent>
          <List dense>
            {filteredMessages.slice(0, 3).map((message) => (
              <ListItem key={message.id} sx={{ px: 0 }}>
                <ListItemAvatar>
                  {getAgentAvatar(message.agentId)}
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2" fontWeight="medium">
                        {message.agentName}
                      </Typography>
                      <Chip
                        label={message.type}
                        size="small"
                        color={getMessageTypeColor(message.type) as any}
                        sx={{ fontSize: '0.65rem', height: 18 }}
                      />
                    </Box>
                  }
                  secondary={
                    <Typography variant="caption" color="textSecondary">
                      {message.message.length > 60 ? `${message.message.substring(0, 60)}...` : message.message}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>
          {messages.length > 3 && (
            <Typography variant="caption" color="textSecondary" sx={{ textAlign: 'center', display: 'block', mt: 1 }}>
              +{messages.length - 3} more messages
            </Typography>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: isPreview ? '100%' : 'auto' }}>
      <CardHeader
        title="Agent Communication Feed"
        subheader={
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="body2" color="textSecondary">
              {filteredMessages.length} messages
            </Typography>
            <Badge color="success" variant="dot" invisible={!isConnected}>
              <Typography variant="caption">
                {isConnected ? 'Live' : 'Offline'}
              </Typography>
            </Badge>
          </Box>
        }
        action={
          <Box display="flex" alignItems="center" gap={1}>
            {isPreview && (
              <IconButton size="small" onClick={() => setIsExpanded(false)}>
                <CollapseIcon />
              </IconButton>
            )}
            <IconButton size="small" onClick={() => window.location.reload()}>
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />

      <CardContent>
        {/* Filters */}
        <Box display="flex" gap={2} mb={2} flexWrap="wrap">
          <TextField
            select
            label="Filter by Type"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            size="small"
            sx={{ minWidth: 150 }}
          >
            <MenuItem value="all">All Messages</MenuItem>
            <MenuItem value="info">Info</MenuItem>
            <MenuItem value="coordination">Coordination</MenuItem>
            <MenuItem value="warning">Warnings</MenuItem>
            <MenuItem value="error">Errors</MenuItem>
            <MenuItem value="success">Success</MenuItem>
          </TextField>

          <TextField
            label="Search messages"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            size="small"
            sx={{ minWidth: 200, flexGrow: 1 }}
            InputProps={{
              endAdornment: searchTerm && (
                <IconButton size="small" onClick={() => setSearchTerm('')}>
                  <ClearIcon fontSize="small" />
                </IconButton>
              )
            }}
          />
        </Box>

        {/* Message List */}
        <Paper variant="outlined" sx={{ maxHeight: 400, overflow: 'auto' }}>
          {filteredMessages.length === 0 ? (
            <Box p={3} textAlign="center">
              <AgentIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 2 }} />
              <Typography variant="body2" color="textSecondary">
                No agent communications yet
              </Typography>
            </Box>
          ) : (
            <List>
              {filteredMessages.map((message, index) => (
                <React.Fragment key={message.id}>
                  <ListItem alignItems="flex-start">
                    <ListItemAvatar>
                      {getAgentAvatar(message.agentId)}
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                          <Typography variant="subtitle2" fontWeight="medium">
                            {message.agentName}
                          </Typography>
                          <Chip
                            label={message.type}
                            size="small"
                            color={getMessageTypeColor(message.type) as any}
                            sx={{ fontSize: '0.65rem', height: 20 }}
                          />
                          {message.targetAgent && (
                            <Tooltip title={`Message to ${message.targetAgent}`}>
                              <Chip
                                label={`→ ${message.targetAgent}`}
                                size="small"
                                variant="outlined"
                                sx={{ fontSize: '0.65rem', height: 20 }}
                              />
                            </Tooltip>
                          )}
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="textPrimary" paragraph>
                            {message.message}
                          </Typography>
                          <Box display="flex" justifyContent="space-between" alignItems="center">
                            <Typography variant="caption" color="textSecondary">
                              {formatTimestamp(message.timestamp)}
                            </Typography>
                            {message.metadata?.performance && (
                              <Chip
                                label={`${message.metadata.performance}% efficient`}
                                size="small"
                                color="primary"
                                variant="outlined"
                                sx={{ fontSize: '0.6rem', height: 18 }}
                              />
                            )}
                          </Box>
                        </Box>
                      }
                    />
                  </ListItem>
                  {index < filteredMessages.length - 1 && <Divider variant="inset" component="li" />}
                </React.Fragment>
              ))}
            </List>
          )}
        </Paper>

        {/* Send Message FAB */}
        {userTier !== UserTier.FREE && (
          <Fab
            color="primary"
            size="small"
            sx={{ position: 'absolute', bottom: 16, right: 16 }}
            onClick={() => setNewMessageDialog(true)}
          >
            <SendIcon />
          </Fab>
        )}
      </CardContent>

      {/* New Message Dialog */}
      <Dialog open={newMessageDialog} onClose={() => setNewMessageDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Send Message to Agent</DialogTitle>
        <DialogContent>
          <TextField
            select
            label="Select Agent"
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            fullWidth
            margin="normal"
          >
            {agents.map((agent) => (
              <MenuItem key={agent.id} value={agent.id}>
                {agent.name} ({agent.id})
              </MenuItem>
            ))}
          </TextField>
          <TextField
            label="Message"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            fullWidth
            multiline
            rows={3}
            margin="normal"
            placeholder="Type your message to the agent..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewMessageDialog(false)}>Cancel</Button>
          <Button onClick={handleSendMessage} disabled={!newMessage.trim() || !selectedAgent} variant="contained">
            Send Message
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};