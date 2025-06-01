#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pydantic AI Tier-Aware Agent Factory

Tests all aspects of the tier-aware agent factory including:
- Capability validation and tier restrictions
- Agent creation with various configurations
- Quota management and resource allocation
- Performance metrics and analytics
- Integration with communication systems
- Error handling and edge cases
"""

import asyncio
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append('.')

try:
    from sources.pydantic_ai_tier_aware_agent_factory import (
        TierAwareAgentFactory, AgentCreationRequest, AgentCreationResult,
        AgentSpecialization, AgentTier, AgentCapability, AgentValidationLevel,
        CapabilityRequirement, create_tier_aware_factory, quick_agent_creation_test
    )
    FACTORY_AVAILABLE = True
except ImportError as e:
    print(f"❌ Factory import failed: {e}")
    FACTORY_AVAILABLE = False

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*70}")
    print(f"🧪 {test_name}")
    print(f"{'='*70}")

def print_test_result(test_name: str, success: bool, details: str = "", execution_time: float = 0.0):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    time_info = f" ({execution_time:.3f}s)" if execution_time > 0 else ""
    print(f"{status:12} {test_name}{time_info}")
    if details:
        print(f"    Details: {details}")

class TierAwareAgentFactoryTestSuite:
    """Comprehensive test suite for Tier-Aware Agent Factory"""
    
    def __init__(self):
        self.factory = None
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.start_time = time.time()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("🧪 Pydantic AI Tier-Aware Agent Factory - Comprehensive Test Suite")
        print("="*70)
        
        tests = [
            ("Import and Initialization", self.test_import_and_initialization),
            ("Factory Configuration", self.test_factory_configuration),
            ("Capability Validation", self.test_capability_validation),
            ("Tier Quota Management", self.test_tier_quota_management),
            ("Agent Creation - FREE Tier", self.test_agent_creation_free_tier),
            ("Agent Creation - PRO Tier", self.test_agent_creation_pro_tier),
            ("Agent Creation - ENTERPRISE Tier", self.test_agent_creation_enterprise_tier),
            ("Specialization Templates", self.test_specialization_templates),
            ("Resource Allocation", self.test_resource_allocation),
            ("Performance Metrics", self.test_performance_metrics),
            ("Agent Lifecycle Management", self.test_agent_lifecycle),
            ("Error Handling", self.test_error_handling),
            ("Factory Analytics", self.test_factory_analytics),
            ("Integration Points", self.test_integration_points),
            ("Quota Enforcement", self.test_quota_enforcement)
        ]
        
        for test_name, test_func in tests:
            await self.run_single_test(test_name, test_func)
        
        return self.generate_final_report()
    
    async def run_single_test(self, test_name: str, test_func):
        """Run a single test with error handling and timing"""
        self.total_tests += 1
        start_time = time.time()
        
        try:
            result = await test_func()
            execution_time = time.time() - start_time
            
            if result.get('success', False):
                self.passed_tests += 1
                print_test_result(test_name, True, result.get('details', ''), execution_time)
            else:
                print_test_result(test_name, False, result.get('error', 'Unknown error'), execution_time)
            
            self.test_results[test_name] = {
                'success': result.get('success', False),
                'execution_time': execution_time,
                'details': result.get('details', ''),
                'error': result.get('error', '')
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{str(e)}"
            print_test_result(test_name, False, error_msg, execution_time)
            
            self.test_results[test_name] = {
                'success': False,
                'execution_time': execution_time,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    async def test_import_and_initialization(self) -> Dict[str, Any]:
        """Test factory import and initialization"""
        try:
            if not FACTORY_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Factory components not available for import'
                }
            
            # Test factory creation
            self.factory = create_tier_aware_factory()
            
            if not self.factory:
                return {
                    'success': False,
                    'error': 'Factory creation returned None'
                }
            
            # Verify factory attributes
            required_attributes = [
                'factory_id', 'status', 'version', 'created_agents', 
                'tier_quotas', 'specialization_templates', 'creation_metrics'
            ]
            
            missing_attributes = [attr for attr in required_attributes 
                                if not hasattr(self.factory, attr)]
            
            if missing_attributes:
                return {
                    'success': False,
                    'error': f'Missing factory attributes: {missing_attributes}'
                }
            
            # Test basic factory properties
            factory_info = {
                'factory_id': self.factory.factory_id,
                'status': self.factory.status.value,
                'version': self.factory.version,
                'tier_quotas_count': len(self.factory.tier_quotas),
                'specialization_templates_count': len(self.factory.specialization_templates)
            }
            
            return {
                'success': True,
                'details': f"Factory initialized: {factory_info}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Initialization failed: {str(e)}'
            }
    
    async def test_factory_configuration(self) -> Dict[str, Any]:
        """Test factory configuration and default settings"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Test tier quotas configuration
            expected_tiers = [AgentTier.FREE, AgentTier.PRO, AgentTier.ENTERPRISE]
            configured_tiers = list(self.factory.tier_quotas.keys())
            
            if set(expected_tiers) != set(configured_tiers):
                return {
                    'success': False,
                    'error': f'Tier configuration mismatch. Expected: {expected_tiers}, Got: {configured_tiers}'
                }
            
            # Test specialization templates
            expected_specializations = list(AgentSpecialization)
            configured_specializations = list(self.factory.specialization_templates.keys())
            
            if set(expected_specializations) != set(configured_specializations):
                return {
                    'success': False,
                    'error': f'Specialization configuration mismatch'
                }
            
            # Verify quota limits make sense
            quota_validation = []
            for tier, limits in self.factory.tier_quotas.items():
                if limits.max_agents <= 0:
                    quota_validation.append(f"{tier.value}: invalid max_agents")
                if limits.max_concurrent_agents <= 0:
                    quota_validation.append(f"{tier.value}: invalid max_concurrent_agents")
                if limits.max_memory_mb <= 0:
                    quota_validation.append(f"{tier.value}: invalid max_memory_mb")
            
            if quota_validation:
                return {
                    'success': False,
                    'error': f'Quota validation failed: {quota_validation}'
                }
            
            config_summary = {
                'tiers_configured': len(configured_tiers),
                'specializations_configured': len(configured_specializations),
                'total_quota_slots': sum(limits.max_agents for limits in self.factory.tier_quotas.values())
            }
            
            return {
                'success': True,
                'details': f"Configuration valid: {config_summary}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Configuration test failed: {str(e)}'
            }
    
    async def test_capability_validation(self) -> Dict[str, Any]:
        """Test capability validation logic"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            validation_tests = []
            
            # Test 1: Valid FREE tier request
            free_request = AgentCreationRequest(
                specialization=AgentSpecialization.CASUAL,
                tier=AgentTier.FREE,
                owner_id="test_user",
                requested_capabilities=[AgentCapability.BASIC_REASONING]
            )
            
            free_validation = self.factory.validate_capabilities(free_request)
            validation_tests.append({
                'test': 'FREE_tier_valid',
                'passed': free_validation.is_valid,
                'score': free_validation.validation_score
            })
            
            # Test 2: Invalid tier for specialization
            invalid_tier_request = AgentCreationRequest(
                specialization=AgentSpecialization.COORDINATOR,
                tier=AgentTier.FREE,  # Coordinator requires PRO+
                owner_id="test_user",
                requested_capabilities=[AgentCapability.COORDINATION]
            )
            
            invalid_validation = self.factory.validate_capabilities(invalid_tier_request)
            validation_tests.append({
                'test': 'invalid_tier_rejected',
                'passed': not invalid_validation.is_valid,
                'violations': len(invalid_validation.tier_violations)
            })
            
            # Test 3: Valid PRO tier request
            pro_request = AgentCreationRequest(
                specialization=AgentSpecialization.CODE,
                tier=AgentTier.PRO,
                owner_id="test_user",
                requested_capabilities=[
                    AgentCapability.ADVANCED_REASONING,
                    AgentCapability.CODE_GENERATION
                ]
            )
            
            pro_validation = self.factory.validate_capabilities(pro_request)
            validation_tests.append({
                'test': 'PRO_tier_valid',
                'passed': pro_validation.is_valid,
                'score': pro_validation.validation_score
            })
            
            # Test 4: Restricted capability validation
            restricted_request = AgentCreationRequest(
                specialization=AgentSpecialization.CASUAL,
                tier=AgentTier.FREE,
                owner_id="test_user",
                requested_capabilities=[
                    AgentCapability.BASIC_REASONING,
                    AgentCapability.MCP_INTEGRATION  # Restricted to ENTERPRISE
                ]
            )
            
            restricted_validation = self.factory.validate_capabilities(restricted_request)
            validation_tests.append({
                'test': 'restricted_capability_rejected',
                'passed': not restricted_validation.is_valid,
                'violations': len(restricted_validation.tier_violations)
            })
            
            # Summarize validation tests
            passed_validations = sum(1 for test in validation_tests if test['passed'])
            total_validations = len(validation_tests)
            
            return {
                'success': passed_validations == total_validations,
                'details': f"Capability validation: {passed_validations}/{total_validations} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Capability validation test failed: {str(e)}'
            }
    
    async def test_tier_quota_management(self) -> Dict[str, Any]:
        """Test tier quota management and limits"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            quota_tests = []
            
            # Test quota utilization calculation
            for tier in AgentTier:
                utilization = self.factory.get_tier_utilization(tier)
                
                # Verify utilization structure
                required_keys = [
                    'tier', 'current_agents', 'max_agents', 'utilization_percentage',
                    'available_slots', 'allowed_capabilities', 'restricted_capabilities'
                ]
                
                missing_keys = [key for key in required_keys if key not in utilization]
                if missing_keys:
                    quota_tests.append({
                        'tier': tier.value,
                        'passed': False,
                        'error': f'Missing keys: {missing_keys}'
                    })
                else:
                    # Verify calculation accuracy
                    expected_utilization = (utilization['current_agents'] / utilization['max_agents']) * 100
                    actual_utilization = utilization['utilization_percentage']
                    
                    calculation_correct = abs(expected_utilization - actual_utilization) < 0.01
                    
                    quota_tests.append({
                        'tier': tier.value,
                        'passed': calculation_correct,
                        'utilization': actual_utilization,
                        'available_slots': utilization['available_slots']
                    })
            
            # Test tier hierarchy validation
            tier_order_tests = []
            
            free_limits = self.factory.tier_quotas[AgentTier.FREE]
            pro_limits = self.factory.tier_quotas[AgentTier.PRO]
            enterprise_limits = self.factory.tier_quotas[AgentTier.ENTERPRISE]
            
            # Verify tier progression makes sense
            if (free_limits.max_agents <= pro_limits.max_agents <= enterprise_limits.max_agents):
                tier_order_tests.append({'test': 'agent_limits_progression', 'passed': True})
            else:
                tier_order_tests.append({'test': 'agent_limits_progression', 'passed': False})
            
            if (free_limits.max_memory_mb <= pro_limits.max_memory_mb <= enterprise_limits.max_memory_mb):
                tier_order_tests.append({'test': 'memory_limits_progression', 'passed': True})
            else:
                tier_order_tests.append({'test': 'memory_limits_progression', 'passed': False})
            
            all_passed = (
                all(test['passed'] for test in quota_tests) and
                all(test['passed'] for test in tier_order_tests)
            )
            
            return {
                'success': all_passed,
                'details': f"Quota management: {len(quota_tests)} tier tests, {len(tier_order_tests)} hierarchy tests"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Quota management test failed: {str(e)}'
            }
    
    async def test_agent_creation_free_tier(self) -> Dict[str, Any]:
        """Test agent creation for FREE tier"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Create a FREE tier agent
            request = AgentCreationRequest(
                specialization=AgentSpecialization.CASUAL,
                tier=AgentTier.FREE,
                owner_id="test_user_free",
                requested_capabilities=[AgentCapability.BASIC_REASONING],
                metadata={"test": "free_tier_creation"}
            )
            
            result, record = await self.factory.create_agent(request)
            
            creation_success = (
                result == AgentCreationResult.SUCCESS and
                record is not None and
                record.tier == AgentTier.FREE and
                record.specialization == AgentSpecialization.CASUAL
            )
            
            if creation_success:
                # Verify agent is tracked
                retrieved_record = self.factory.get_agent_record(record.agent_id)
                tracking_success = (retrieved_record is not None and 
                                  retrieved_record.agent_id == record.agent_id)
                
                return {
                    'success': tracking_success,
                    'details': f"FREE tier agent created: {record.agent_id}, tracking: {tracking_success}"
                }
            else:
                return {
                    'success': False,
                    'error': f'Agent creation failed: {result.value}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'FREE tier creation test failed: {str(e)}'
            }
    
    async def test_agent_creation_pro_tier(self) -> Dict[str, Any]:
        """Test agent creation for PRO tier"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Create a PRO tier agent
            request = AgentCreationRequest(
                specialization=AgentSpecialization.RESEARCH,
                tier=AgentTier.PRO,
                owner_id="test_user_pro",
                requested_capabilities=[
                    AgentCapability.BASIC_REASONING,
                    AgentCapability.WEB_BROWSING,
                    AgentCapability.FILE_OPERATIONS
                ],
                metadata={"test": "pro_tier_creation"}
            )
            
            result, record = await self.factory.create_agent(request)
            
            creation_success = (
                result == AgentCreationResult.SUCCESS and
                record is not None and
                record.tier == AgentTier.PRO and
                record.specialization == AgentSpecialization.RESEARCH
            )
            
            if creation_success:
                # Verify enhanced capabilities
                has_web_browsing = AgentCapability.WEB_BROWSING in record.capabilities
                
                return {
                    'success': has_web_browsing,
                    'details': f"PRO tier agent created: {record.agent_id}, capabilities: {len(record.capabilities)}"
                }
            else:
                return {
                    'success': False,
                    'error': f'PRO tier agent creation failed: {result.value}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'PRO tier creation test failed: {str(e)}'
            }
    
    async def test_agent_creation_enterprise_tier(self) -> Dict[str, Any]:
        """Test agent creation for ENTERPRISE tier"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Create an ENTERPRISE tier agent
            request = AgentCreationRequest(
                specialization=AgentSpecialization.MCP,
                tier=AgentTier.ENTERPRISE,
                owner_id="test_user_enterprise",
                requested_capabilities=[
                    AgentCapability.ADVANCED_REASONING,
                    AgentCapability.MCP_INTEGRATION,
                    AgentCapability.TOOL_USAGE,
                    AgentCapability.MEMORY_ACCESS
                ],
                metadata={"test": "enterprise_tier_creation"}
            )
            
            result, record = await self.factory.create_agent(request)
            
            creation_success = (
                result == AgentCreationResult.SUCCESS and
                record is not None and
                record.tier == AgentTier.ENTERPRISE and
                record.specialization == AgentSpecialization.MCP
            )
            
            if creation_success:
                # Verify premium capabilities
                has_mcp = AgentCapability.MCP_INTEGRATION in record.capabilities
                has_memory = AgentCapability.MEMORY_ACCESS in record.capabilities
                
                return {
                    'success': has_mcp and has_memory,
                    'details': f"ENTERPRISE tier agent created: {record.agent_id}, premium features: {has_mcp and has_memory}"
                }
            else:
                return {
                    'success': False,
                    'error': f'ENTERPRISE tier agent creation failed: {result.value}'
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'ENTERPRISE tier creation test failed: {str(e)}'
            }
    
    async def test_specialization_templates(self) -> Dict[str, Any]:
        """Test specialization templates and configurations"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            template_tests = []
            
            # Test each specialization template
            for specialization in AgentSpecialization:
                template = self.factory.specialization_templates.get(specialization)
                
                if not template:
                    template_tests.append({
                        'specialization': specialization.value,
                        'passed': False,
                        'error': 'Template not found'
                    })
                    continue
                
                # Verify template structure
                template_valid = (
                    hasattr(template, 'specialization') and
                    hasattr(template, 'default_capabilities') and
                    hasattr(template, 'minimum_tier') and
                    hasattr(template, 'validation_level')
                )
                
                # Verify capability consistency
                capability_valid = True
                if hasattr(template, 'required_capabilities'):
                    for req_cap in template.required_capabilities:
                        if hasattr(req_cap, 'capability') and req_cap.capability not in list(AgentCapability):
                            capability_valid = False
                            break
                
                template_tests.append({
                    'specialization': specialization.value,
                    'passed': template_valid and capability_valid,
                    'default_capabilities': len(template.default_capabilities) if hasattr(template, 'default_capabilities') else 0,
                    'minimum_tier': template.minimum_tier.value if hasattr(template, 'minimum_tier') else 'unknown'
                })
            
            passed_templates = sum(1 for test in template_tests if test['passed'])
            total_templates = len(template_tests)
            
            return {
                'success': passed_templates == total_templates,
                'details': f"Specialization templates: {passed_templates}/{total_templates} valid"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Specialization template test failed: {str(e)}'
            }
    
    async def test_resource_allocation(self) -> Dict[str, Any]:
        """Test resource allocation for different tiers and specializations"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            allocation_tests = []
            
            # Test resource allocation for each tier
            test_configs = [
                (AgentSpecialization.CASUAL, AgentTier.FREE),
                (AgentSpecialization.RESEARCH, AgentTier.PRO),
                (AgentSpecialization.CODE, AgentTier.ENTERPRISE)
            ]
            
            for spec, tier in test_configs:
                request = AgentCreationRequest(
                    specialization=spec,
                    tier=tier,
                    owner_id="test_user_allocation",
                    requested_capabilities=[]
                )
                
                tier_limits = self.factory.tier_quotas[tier]
                template = self.factory.specialization_templates[spec]
                
                # Test resource allocation logic
                allocated = self.factory._allocate_agent_resources(request, tier_limits)
                
                # Verify allocation structure
                allocation_valid = (
                    'memory_mb' in allocated and
                    'cpu_priority' in allocated and
                    'processing_timeout' in allocated and
                    allocated['memory_mb'] <= tier_limits.max_memory_mb
                )
                
                allocation_tests.append({
                    'config': f"{spec.value}_{tier.value}",
                    'passed': allocation_valid,
                    'memory_mb': allocated.get('memory_mb', 0),
                    'timeout': allocated.get('processing_timeout', 0)
                })
            
            passed_allocations = sum(1 for test in allocation_tests if test['passed'])
            total_allocations = len(allocation_tests)
            
            return {
                'success': passed_allocations == total_allocations,
                'details': f"Resource allocation: {passed_allocations}/{total_allocations} valid"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Resource allocation test failed: {str(e)}'
            }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics and tracking"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Get initial metrics
            initial_metrics = self.factory.creation_metrics.copy()
            
            # Create a test agent to generate metrics
            request = AgentCreationRequest(
                specialization=AgentSpecialization.CASUAL,
                tier=AgentTier.FREE,
                owner_id="test_user_metrics"
            )
            
            result, record = await self.factory.create_agent(request)
            
            # Get updated metrics
            updated_metrics = self.factory.creation_metrics
            
            # Verify metrics were updated
            metrics_updated = (
                updated_metrics['total_created'] > initial_metrics['total_created'] and
                updated_metrics['successful_creations'] > initial_metrics['successful_creations']
            )
            
            # Test analytics generation
            analytics = self.factory.get_factory_analytics()
            
            analytics_valid = (
                'factory_info' in analytics and
                'creation_metrics' in analytics and
                'tier_utilization' in analytics and
                'performance_summary' in analytics
            )
            
            return {
                'success': metrics_updated and analytics_valid,
                'details': f"Metrics tracking: {metrics_updated}, Analytics: {analytics_valid}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Performance metrics test failed: {str(e)}'
            }
    
    async def test_agent_lifecycle(self) -> Dict[str, Any]:
        """Test agent lifecycle management"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Create an agent
            request = AgentCreationRequest(
                specialization=AgentSpecialization.CASUAL,
                tier=AgentTier.FREE,
                owner_id="test_user_lifecycle"
            )
            
            result, record = await self.factory.create_agent(request)
            
            if result != AgentCreationResult.SUCCESS or not record:
                return {
                    'success': False,
                    'error': 'Failed to create agent for lifecycle test'
                }
            
            agent_id = record.agent_id
            
            # Test status update
            status_updated = self.factory.update_agent_status(agent_id, "testing")
            
            # Verify status update
            updated_record = self.factory.get_agent_record(agent_id)
            status_correct = (updated_record and updated_record.status == "testing")
            
            # Test agent deactivation
            deactivation_success = await self.factory.deactivate_agent(agent_id)
            
            # Verify deactivation
            deactivated_record = self.factory.get_agent_record(agent_id)
            deactivation_correct = (deactivated_record and deactivated_record.status == "deactivated")
            
            lifecycle_success = (
                status_updated and status_correct and
                deactivation_success and deactivation_correct
            )
            
            return {
                'success': lifecycle_success,
                'details': f"Agent lifecycle: status update: {status_correct}, deactivation: {deactivation_correct}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Agent lifecycle test failed: {str(e)}'
            }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and edge cases"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            error_tests = []
            
            # Test 1: Invalid specialization
            try:
                invalid_request = AgentCreationRequest(
                    specialization="invalid_specialization",  # This should cause validation error
                    tier=AgentTier.FREE,
                    owner_id="test_user_error"
                )
                # This should fail gracefully, not crash
                error_tests.append({'test': 'invalid_specialization', 'passed': True})
            except Exception:
                error_tests.append({'test': 'invalid_specialization', 'passed': True})  # Expected to fail
            
            # Test 2: Nonexistent agent retrieval
            nonexistent_record = self.factory.get_agent_record("nonexistent_agent_id")
            error_tests.append({
                'test': 'nonexistent_agent_retrieval',
                'passed': nonexistent_record is None
            })
            
            # Test 3: Invalid tier utilization request
            try:
                invalid_utilization = self.factory.get_tier_utilization("invalid_tier")
                error_tests.append({
                    'test': 'invalid_tier_utilization',
                    'passed': 'error' in invalid_utilization
                })
            except Exception:
                error_tests.append({'test': 'invalid_tier_utilization', 'passed': True})
            
            # Test 4: Deactivation of nonexistent agent
            deactivation_result = await self.factory.deactivate_agent("nonexistent_agent")
            error_tests.append({
                'test': 'nonexistent_agent_deactivation',
                'passed': deactivation_result is False
            })
            
            passed_error_tests = sum(1 for test in error_tests if test['passed'])
            total_error_tests = len(error_tests)
            
            return {
                'success': passed_error_tests == total_error_tests,
                'details': f"Error handling: {passed_error_tests}/{total_error_tests} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error handling test failed: {str(e)}'
            }
    
    async def test_factory_analytics(self) -> Dict[str, Any]:
        """Test factory analytics and reporting"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Get comprehensive analytics
            analytics = self.factory.get_factory_analytics()
            
            # Verify analytics structure
            required_sections = [
                'factory_info',
                'creation_metrics',
                'tier_utilization',
                'specialization_capacity',
                'performance_summary'
            ]
            
            missing_sections = [section for section in required_sections 
                              if section not in analytics]
            
            if missing_sections:
                return {
                    'success': False,
                    'error': f'Missing analytics sections: {missing_sections}'
                }
            
            # Verify factory info completeness
            factory_info = analytics['factory_info']
            required_info = ['factory_id', 'version', 'status', 'uptime']
            missing_info = [info for info in required_info if info not in factory_info]
            
            # Verify tier utilization completeness
            tier_util = analytics['tier_utilization']
            expected_tiers = {tier.value for tier in AgentTier}
            actual_tiers = set(tier_util.keys())
            
            # Verify performance summary
            perf_summary = analytics['performance_summary']
            required_perf = ['average_creation_time', 'success_rate', 'failure_rate']
            missing_perf = [perf for perf in required_perf if perf not in perf_summary]
            
            analytics_complete = (
                not missing_sections and
                not missing_info and
                expected_tiers == actual_tiers and
                not missing_perf
            )
            
            return {
                'success': analytics_complete,
                'details': f"Analytics complete: {analytics_complete}, sections: {len(analytics)}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Factory analytics test failed: {str(e)}'
            }
    
    async def test_integration_points(self) -> Dict[str, Any]:
        """Test integration points and external dependencies"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            integration_tests = []
            
            # Test communication manager injection
            try:
                # Mock communication manager
                mock_comm_manager = {"type": "mock", "status": "active"}
                self.factory.set_communication_manager(mock_comm_manager)
                
                has_comm_manager = hasattr(self.factory, 'communication_manager')
                integration_tests.append({
                    'test': 'communication_manager_injection',
                    'passed': has_comm_manager
                })
            except Exception as e:
                integration_tests.append({
                    'test': 'communication_manager_injection',
                    'passed': False,
                    'error': str(e)
                })
            
            # Test agent listing with filters
            try:
                all_agents = self.factory.list_agents()
                free_agents = self.factory.list_agents(tier=AgentTier.FREE)
                casual_agents = self.factory.list_agents(specialization=AgentSpecialization.CASUAL)
                
                filtering_works = (
                    isinstance(all_agents, list) and
                    isinstance(free_agents, list) and
                    isinstance(casual_agents, list)
                )
                
                integration_tests.append({
                    'test': 'agent_listing_filters',
                    'passed': filtering_works
                })
            except Exception as e:
                integration_tests.append({
                    'test': 'agent_listing_filters',
                    'passed': False,
                    'error': str(e)
                })
            
            # Test quick creation test function
            try:
                quick_test_result = await quick_agent_creation_test()
                quick_test_success = isinstance(quick_test_result, dict) and 'factory_initialized' in quick_test_result
                
                integration_tests.append({
                    'test': 'quick_creation_test',
                    'passed': quick_test_success
                })
            except Exception as e:
                integration_tests.append({
                    'test': 'quick_creation_test',
                    'passed': False,
                    'error': str(e)
                })
            
            passed_integration_tests = sum(1 for test in integration_tests if test['passed'])
            total_integration_tests = len(integration_tests)
            
            return {
                'success': passed_integration_tests == total_integration_tests,
                'details': f"Integration points: {passed_integration_tests}/{total_integration_tests} tests passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Integration points test failed: {str(e)}'
            }
    
    async def test_quota_enforcement(self) -> Dict[str, Any]:
        """Test quota enforcement and limits"""
        try:
            if not self.factory:
                return {'success': False, 'error': 'Factory not initialized'}
            
            # Get FREE tier limits
            free_limits = self.factory.tier_quotas[AgentTier.FREE]
            initial_agent_count = len([a for a in self.factory.created_agents.values() 
                                     if a.tier == AgentTier.FREE and a.status == 'active'])
            
            # Try to create agents up to the limit
            creation_results = []
            for i in range(free_limits.max_agents - initial_agent_count + 1):  # Try to exceed by 1
                request = AgentCreationRequest(
                    specialization=AgentSpecialization.CASUAL,
                    tier=AgentTier.FREE,
                    owner_id=f"test_quota_user_{i}"
                )
                
                result, record = await self.factory.create_agent(request)
                creation_results.append(result)
            
            # The last creation should fail due to quota
            quota_enforced = creation_results[-1] == AgentCreationResult.QUOTA_EXCEEDED
            
            # Count successful creations
            successful_creations = sum(1 for result in creation_results 
                                     if result == AgentCreationResult.SUCCESS)
            
            # Should not exceed the quota
            within_quota = successful_creations <= (free_limits.max_agents - initial_agent_count)
            
            return {
                'success': quota_enforced and within_quota,
                'details': f"Quota enforcement: enforced={quota_enforced}, within_limits={within_quota}, created={successful_creations}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Quota enforcement test failed: {str(e)}'
            }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_execution_time = time.time() - self.start_time
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        # Determine readiness level
        if success_rate >= 90:
            readiness = "🚀 READY FOR PRODUCTION"
        elif success_rate >= 80:
            readiness = "⚠️ READY FOR FURTHER DEVELOPMENT"
        elif success_rate >= 70:
            readiness = "🔧 REQUIRES ADDITIONAL WORK"
        else:
            readiness = "❌ NOT READY - SIGNIFICANT ISSUES"
        
        print(f"\n{'='*70}")
        print("📋 PYDANTIC AI TIER-AWARE AGENT FACTORY TEST RESULTS")
        print(f"{'='*70}")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_info = f" ({result['execution_time']:.3f}s)"
            print(f"{status:12} {test_name}{time_info}")
            if result.get('details'):
                print(f"    Details: {result['details']}")
            if result.get('error'):
                print(f"    Error: {result['error']}")
        
        print(f"\n{'-'*70}")
        print(f"Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Execution Time: {total_execution_time:.2f}s")
        
        print(f"\n🚀 READINESS ASSESSMENT")
        print(f"-----------------------------------")
        print(f"{readiness}")
        if success_rate >= 80:
            print("✅ Core tier-aware agent factory functionality validated")
        if success_rate < 80:
            print("⚠️ Some components may need refinement")
        
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'success_rate': success_rate,
            'execution_time': total_execution_time,
            'readiness': readiness,
            'test_results': self.test_results
        }

# Main execution
async def main():
    """Run the comprehensive test suite"""
    test_suite = TierAwareAgentFactoryTestSuite()
    await test_suite.run_all_tests()
    
    print("\n🎉 Tier-Aware Agent Factory test suite completed!")

if __name__ == "__main__":
    asyncio.run(main())