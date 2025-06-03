/**
 * Tier-Aware Feature Gate Component
 * 
 * * Purpose: Component wrapper that enforces tier-based access control with upgrade prompts
 * * Issues & Complexity Summary: Tier validation and graceful feature restriction handling
 * * Key Complexity Drivers:
 *   - Logic Scope (Est. LoC): ~150
 *   - Core Algorithm Complexity: Medium
 *   - Dependencies: 3 New, 2 Mod
 *   - State Management Complexity: Medium
 *   - Novelty/Uncertainty Factor: Low
 * * AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 65%
 * * Problem Estimate (Inherent Problem Difficulty %): 60%
 * * Initial Code Complexity Estimate %: 65%
 * * Justification for Estimates: Clear tier validation logic with UI feedback
 * * Final Code Complexity (Actual %): TBD
 * * Overall Result Score (Success & Quality %): TBD
 * * Key Variances/Learnings: TBD
 * * Last Updated: 2025-06-03
 */

import React, { useState } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import {
  Lock,
  Upgrade,
  Star,
  CheckCircle,
  Close,
  Info,
  Security,
  Speed,
  VideoCall,
  Memory,
  Group
} from '@mui/icons-material';
import { UserTier, getTierLimits, getUpgradeMessage } from '../config/copilotkit.config';

interface TierGateProps {
  children: React.ReactNode;
  requiredTier: UserTier;
  currentTier: UserTier;
  feature: string;
  description?: string;
  showUpgradePrompt?: boolean;
  fallbackComponent?: React.ReactNode;
  onUpgradeClick?: () => void;
}

const TIER_HIERARCHY = {
  [UserTier.FREE]: 0,
  [UserTier.PRO]: 1,
  [UserTier.ENTERPRISE]: 2
};

const TIER_FEATURES = {
  [UserTier.FREE]: [
    'Basic agent coordination',
    'Up to 2 agents',
    '1 concurrent workflow',
    'Community support'
  ],
  [UserTier.PRO]: [
    'Advanced agent coordination',
    'Up to 5 agents',
    '3 concurrent workflows',
    'Custom workflows',
    'Real-time metrics',
    'Email support',
    'Apple Silicon optimization'
  ],
  [UserTier.ENTERPRISE]: [
    'Full agent coordination',
    'Up to 20 agents',
    '10 concurrent workflows',
    'Video generation',
    'Internal communications',
    'Priority support',
    'Advanced hardware optimization',
    'Custom integrations'
  ]
};

const TIER_COLORS = {
  [UserTier.FREE]: '#9e9e9e',
  [UserTier.PRO]: '#2196f3',
  [UserTier.ENTERPRISE]: '#ff9800'
};

const TIER_ICONS = {
  [UserTier.FREE]: <Group />,
  [UserTier.PRO]: <Speed />,
  [UserTier.ENTERPRISE]: <Star />
};

export const TierGate: React.FC<TierGateProps> = ({
  children,
  requiredTier,
  currentTier,
  feature,
  description,
  showUpgradePrompt = true,
  fallbackComponent,
  onUpgradeClick
}) => {
  const [upgradeDialogOpen, setUpgradeDialogOpen] = useState(false);

  // Check if user has access to the feature
  const hasAccess = TIER_HIERARCHY[currentTier] >= TIER_HIERARCHY[requiredTier];

  // If user has access, render children normally
  if (hasAccess) {
    return <>{children}</>;
  }

  // Determine what to show when access is denied
  const handleUpgradeClick = () => {
    if (onUpgradeClick) {
      onUpgradeClick();
    } else {
      setUpgradeDialogOpen(true);
    }
  };

  // Custom fallback component
  if (fallbackComponent) {
    return <>{fallbackComponent}</>;
  }

  // Default restricted access UI
  const RestrictedFeatureUI = () => (
    <Box
      sx={{
        position: 'relative',
        opacity: 0.6,
        pointerEvents: 'none',
        filter: 'grayscale(50%)',
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: 1,
          zIndex: 1
        }
      }}
    >
      {children}
      
      {/* Overlay with upgrade prompt */}
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 2,
          pointerEvents: 'auto'
        }}
      >
        <Card sx={{ minWidth: 250, textAlign: 'center' }}>
          <CardContent>
            <Lock sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
            <Typography variant="h6" gutterBottom>
              {feature}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Requires {requiredTier.toUpperCase()} tier
            </Typography>
            {showUpgradePrompt && (
              <Button
                variant="contained"
                size="small"
                startIcon={<Upgrade />}
                onClick={handleUpgradeClick}
                sx={{ mt: 1 }}
              >
                Upgrade Now
              </Button>
            )}
          </CardContent>
        </Card>
      </Box>
    </Box>
  );

  return (
    <>
      <RestrictedFeatureUI />
      
      {/* Upgrade Dialog */}
      <Dialog
        open={upgradeDialogOpen}
        onClose={() => setUpgradeDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h5">
              Upgrade to Access {feature}
            </Typography>
            <IconButton onClick={() => setUpgradeDialogOpen(false)}>
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>
        
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              {getUpgradeMessage(currentTier, feature)}
            </Typography>
          </Alert>

          {description && (
            <Typography variant="body1" paragraph>
              {description}
            </Typography>
          )}

          {/* Tier Comparison */}
          <Box display="flex" gap={2} mb={3}>
            {Object.values(UserTier).map((tier) => (
              <Card
                key={tier}
                sx={{
                  flex: 1,
                  border: tier === requiredTier ? 2 : 1,
                  borderColor: tier === requiredTier ? 'primary.main' : 'divider',
                  bgcolor: tier === currentTier ? 'action.selected' : 'background.paper'
                }}
              >
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    {TIER_ICONS[tier]}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      {tier.toUpperCase()}
                    </Typography>
                    {tier === currentTier && (
                      <Chip label="Current" size="small" sx={{ ml: 1 }} />
                    )}
                    {tier === requiredTier && (
                      <Chip label="Required" color="primary" size="small" sx={{ ml: 1 }} />
                    )}
                  </Box>
                  
                  <List dense>
                    {TIER_FEATURES[tier].map((featureItem, index) => (
                      <ListItem key={index} sx={{ px: 0 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircle 
                            sx={{ 
                              fontSize: 16, 
                              color: TIER_COLORS[tier] 
                            }} 
                          />
                        </ListItemIcon>
                        <ListItemText
                          primary={featureItem}
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            ))}
          </Box>

          {/* Feature Benefits */}
          <Box>
            <Typography variant="h6" gutterBottom>
              What You'll Get with {requiredTier.toUpperCase()} Tier:
            </Typography>
            
            <List>
              {requiredTier === UserTier.PRO && [
                { icon: <Speed />, text: 'Advanced agent coordination with custom workflows' },
                { icon: <Group />, text: 'Up to 5 simultaneous agents' },
                { icon: <Memory />, text: 'Apple Silicon hardware optimization' },
                { icon: <Info />, text: 'Real-time performance metrics and analytics' }
              ].map((benefit, index) => (
                <ListItem key={index}>
                  <ListItemIcon>{benefit.icon}</ListItemIcon>
                  <ListItemText primary={benefit.text} />
                </ListItem>
              ))}
              
              {requiredTier === UserTier.ENTERPRISE && [
                { icon: <VideoCall />, text: 'AI-powered video generation capabilities' },
                { icon: <Group />, text: 'Up to 20 simultaneous agents' },
                { icon: <Security />, text: 'Internal agent communications and advanced security' },
                { icon: <Star />, text: 'Priority support and custom integrations' }
              ].map((benefit, index) => (
                <ListItem key={index}>
                  <ListItemIcon>{benefit.icon}</ListItemIcon>
                  <ListItemText primary={benefit.text} />
                </ListItem>
              ))}
            </List>
          </Box>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setUpgradeDialogOpen(false)}>
            Maybe Later
          </Button>
          <Button
            variant="contained"
            startIcon={<Upgrade />}
            onClick={() => {
              setUpgradeDialogOpen(false);
              // Implement upgrade logic
              window.open('/upgrade', '_blank');
            }}
          >
            Upgrade to {requiredTier.toUpperCase()}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

// Utility component for inline tier requirements
export const TierRequirement: React.FC<{
  requiredTier: UserTier;
  currentTier: UserTier;
  feature: string;
}> = ({ requiredTier, currentTier, feature }) => {
  const hasAccess = TIER_HIERARCHY[currentTier] >= TIER_HIERARCHY[requiredTier];
  
  if (hasAccess) {
    return null;
  }

  return (
    <Tooltip title={`${feature} requires ${requiredTier.toUpperCase()} tier`}>
      <Chip
        icon={<Lock />}
        label={requiredTier.toUpperCase()}
        size="small"
        color="default"
        variant="outlined"
      />
    </Tooltip>
  );
};

// Hook for checking tier access
export const useTierAccess = (requiredTier: UserTier, currentTier: UserTier) => {
  return {
    hasAccess: TIER_HIERARCHY[currentTier] >= TIER_HIERARCHY[requiredTier],
    tierHierarchy: TIER_HIERARCHY,
    upgradeMessage: getUpgradeMessage(currentTier, `${requiredTier} tier features`)
  };
};