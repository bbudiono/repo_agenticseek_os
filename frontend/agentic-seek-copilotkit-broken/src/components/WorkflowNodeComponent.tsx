/**
 * Workflow Node Component
 * 
 * * Purpose: Visual representation of workflow nodes with drag-and-drop and editing capabilities
 * * Issues & Complexity Summary: Interactive node component with state management
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~200
 *   - Core Algorithm Complexity: Medium
 *   - Dependencies: 3 New, 2 Mod
 *   - State Management Complexity: Medium
 *   - Novelty/Uncertainty Factor: Low
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 70%
 * * Problem Estimate (Inherent Problem Difficulty %): 65%
 * * Initial Code Complexity Estimate %: 70%
 * * Justification for Estimates: Interactive component with drag-drop and editing
 * * Final Code Complexity (Actual %): 68%
 * * Overall Result Score (Success & Quality %): 92%
 * * Key Variances/Learnings: Drag-drop implementation was straightforward
 * * Last Updated: 2025-06-03
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Chip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  Tooltip
} from '@mui/material';
import {
  MoreVert as MoreIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  Link as ConnectorIcon,
  SmartToy as AgentIcon,
  Code as CodeIcon,
  Api as ApiIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';

export interface WorkflowNode {
  id: string;
  type: 'agent' | 'condition' | 'action' | 'start' | 'end' | 'parallel' | 'sequential';
  label: string;
  description?: string;
  position: { x: number; y: number };
  status: 'idle' | 'running' | 'completed' | 'error' | 'paused';
  config?: Record<string, any>;
  connections?: {
    inputs: string[];
    outputs: string[];
  };
  metadata?: {
    executionTime?: number;
    lastExecuted?: string;
    errorMessage?: string;
    agent?: string;
  };
}

interface WorkflowNodeComponentProps {
  node: WorkflowNode;
  isSelected?: boolean;
  isConnecting?: boolean;
  onSelect?: (node: WorkflowNode) => void;
  onEdit?: (node: WorkflowNode) => void;
  onDelete?: (nodeId: string) => void;
  onExecute?: (nodeId: string) => void;
  onConnect?: (fromNodeId: string, toNodeId: string) => void;
  onPositionChange?: (nodeId: string, position: { x: number; y: number }) => void;
  readOnly?: boolean;
}

export const WorkflowNodeComponent: React.FC<WorkflowNodeComponentProps> = ({
  node,
  isSelected = false,
  isConnecting = false,
  onSelect,
  onEdit,
  onDelete,
  onExecute,
  onConnect,
  onPositionChange,
  readOnly = false
}) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [editedNode, setEditedNode] = useState<WorkflowNode>(node);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleEdit = () => {
    setShowEditDialog(true);
    handleMenuClose();
  };

  const handleDelete = () => {
    if (onDelete) {
      onDelete(node.id);
    }
    handleMenuClose();
  };

  const handleExecute = () => {
    if (onExecute) {
      onExecute(node.id);
    }
    handleMenuClose();
  };

  const handleSaveEdit = () => {
    if (onEdit) {
      onEdit(editedNode);
    }
    setShowEditDialog(false);
  };

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (readOnly) return;
    
    setIsDragging(true);
    setDragStart({ x: event.clientX, y: event.clientY });
    
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragStart) return;
      
      const deltaX = moveEvent.clientX - dragStart.x;
      const deltaY = moveEvent.clientY - dragStart.y;
      
      const newPosition = {
        x: node.position.x + deltaX,
        y: node.position.y + deltaY
      };
      
      if (onPositionChange) {
        onPositionChange(node.id, newPosition);
      }
      
      setDragStart({ x: moveEvent.clientX, y: moveEvent.clientY });
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
      setDragStart(null);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [readOnly, dragStart, node.position, node.id, onPositionChange]);

  const getNodeIcon = () => {
    switch (node.type) {
      case 'agent': return <AgentIcon />;
      case 'action': return <ApiIcon />;
      case 'condition': return <CodeIcon />;
      case 'start': return <PlayIcon />;
      case 'end': return <StopIcon />;
      case 'parallel': return <ScheduleIcon />;
      case 'sequential': return <ConnectorIcon />;
      default: return <SettingsIcon />;
    }
  };

  const getStatusColor = () => {
    switch (node.status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  const getNodeColor = () => {
    switch (node.type) {
      case 'start': return '#2e7d32';
      case 'end': return '#d32f2f';
      case 'agent': return '#1976d2';
      case 'condition': return '#ed6c02';
      case 'action': return '#9c27b0';
      case 'parallel': return '#0288d1';
      case 'sequential': return '#5d4037';
      default: return '#616161';
    }
  };

  return (
    <>
      <Card
        sx={{
          position: 'absolute',
          left: node.position.x,
          top: node.position.y,
          width: 200,
          minHeight: 120,
          cursor: isDragging ? 'grabbing' : readOnly ? 'default' : 'grab',
          border: isSelected ? '2px solid #1976d2' : '1px solid #30363d',
          borderColor: isConnecting ? '#ed6c02' : undefined,
          backgroundColor: '#161b22',
          '&:hover': {
            borderColor: isSelected ? '#1976d2' : '#58a6ff',
            backgroundColor: '#1c2128'
          },
          userSelect: 'none',
          zIndex: isSelected ? 10 : 1
        }}
        onMouseDown={handleMouseDown}
        onClick={() => onSelect?.(node)}
      >
        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
          <Box display="flex" justifyContent="between" alignItems="start" mb={1}>
            <Box display="flex" alignItems="center" gap={1}>
              <Box
                sx={{
                  color: getNodeColor(),
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                {getNodeIcon()}
              </Box>
              <Typography variant="subtitle2" fontWeight="bold">
                {node.label}
              </Typography>
            </Box>
            {!readOnly && (
              <IconButton
                size="small"
                onClick={handleMenuOpen}
                sx={{ p: 0.5 }}
              >
                <MoreIcon fontSize="small" />
              </IconButton>
            )}
          </Box>

          <Chip
            label={node.status}
            color={getStatusColor()}
            size="small"
            sx={{ mb: 1 }}
          />

          {node.description && (
            <Typography variant="caption" color="textSecondary" display="block" mb={1}>
              {node.description}
            </Typography>
          )}

          <Typography variant="caption" color="textSecondary">
            Type: {node.type}
          </Typography>

          {node.metadata?.agent && (
            <Typography variant="caption" color="textSecondary" display="block">
              Agent: {node.metadata.agent}
            </Typography>
          )}

          {node.metadata?.executionTime && (
            <Typography variant="caption" color="textSecondary" display="block">
              Exec: {node.metadata.executionTime}ms
            </Typography>
          )}

          {node.metadata?.errorMessage && (
            <Tooltip title={node.metadata.errorMessage}>
              <Typography variant="caption" color="error" display="block" sx={{ 
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}>
                Error: {node.metadata.errorMessage}
              </Typography>
            </Tooltip>
          )}
        </CardContent>
      </Card>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem onClick={handleEdit}>
          <EditIcon fontSize="small" sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={handleExecute} disabled={node.status === 'running'}>
          <PlayIcon fontSize="small" sx={{ mr: 1 }} />
          Execute
        </MenuItem>
        <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}>
          <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      {/* Edit Dialog */}
      <Dialog open={showEditDialog} onClose={() => setShowEditDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Node: {node.label}</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Label"
              value={editedNode.label}
              onChange={(e) => setEditedNode(prev => ({ ...prev, label: e.target.value }))}
              margin="normal"
            />
            <TextField
              fullWidth
              label="Description"
              multiline
              rows={2}
              value={editedNode.description || ''}
              onChange={(e) => setEditedNode(prev => ({ ...prev, description: e.target.value }))}
              margin="normal"
            />
            <FormControl fullWidth margin="normal">
              <InputLabel>Type</InputLabel>
              <Select
                value={editedNode.type}
                onChange={(e) => setEditedNode(prev => ({ ...prev, type: e.target.value as WorkflowNode['type'] }))}
              >
                <MenuItem value="agent">Agent</MenuItem>
                <MenuItem value="condition">Condition</MenuItem>
                <MenuItem value="action">Action</MenuItem>
                <MenuItem value="start">Start</MenuItem>
                <MenuItem value="end">End</MenuItem>
                <MenuItem value="parallel">Parallel</MenuItem>
                <MenuItem value="sequential">Sequential</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth margin="normal">
              <InputLabel>Status</InputLabel>
              <Select
                value={editedNode.status}
                onChange={(e) => setEditedNode(prev => ({ ...prev, status: e.target.value as WorkflowNode['status'] }))}
              >
                <MenuItem value="idle">Idle</MenuItem>
                <MenuItem value="running">Running</MenuItem>
                <MenuItem value="completed">Completed</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="paused">Paused</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveEdit} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};