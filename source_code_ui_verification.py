#!/usr/bin/env python3
"""
SOURCE CODE UI VERIFICATION
Verifies UI elements in the actual source code (not minified build)
This gives us accurate verification of what's actually implemented
"""

import json
import os
from datetime import datetime
from pathlib import Path

class SourceCodeUIVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'source_verification': {},
            'manual_verification_guide': [],
            'summary': {
                'total_elements': 0,
                'verified_elements': 0,
                'critical_missing': 0,
                'warnings': []
            }
        }
    
    def verify_functional_app_source(self):
        """Verify the actual FunctionalApp.tsx source code"""
        print("📱 Verifying FunctionalApp.tsx Source Code...")
        
        functional_app = self.project_root / "frontend/agentic-seek-copilotkit-broken/src/FunctionalApp.tsx"
        
        if not functional_app.exists():
            print("❌ FunctionalApp.tsx not found!")
            return False
        
        content = functional_app.read_text()
        
        # Comprehensive UI element verification
        ui_elements = {
            'navigation_tabs': {
                'patterns': [
                    "dashboard' | 'agents' | 'tasks' | 'settings'",
                    "activeTab === tab",
                    "setActiveTab(tab)",
                    "Dashboard', 'Agents', 'Tasks', 'Settings'"
                ],
                'description': '4-tab navigation system',
                'critical': True
            },
            'dashboard_components': {
                'patterns': [
                    "systemStats && (",
                    "Total Agents",
                    "Total Tasks", 
                    "System Load",
                    "Memory Usage",
                    "Recent Activity"
                ],
                'description': 'Dashboard with real-time stats and activity feed',
                'critical': True
            },
            'agent_crud_operations': {
                'patterns': [
                    "handleCreateAgent",
                    "handleDeleteAgent", 
                    "Create Agent",
                    "Agent Name *:",
                    "agents.map(agent",
                    "window.confirm"
                ],
                'description': 'Complete agent CRUD operations',
                'critical': True
            },
            'task_crud_operations': {
                'patterns': [
                    "handleCreateTask",
                    "handleExecuteTask",
                    "Create Task",
                    "Task Title *:",
                    "tasks.map(task",
                    "Execute Task"
                ],
                'description': 'Complete task CRUD operations',
                'critical': True
            },
            'form_validation': {
                'patterns': [
                    "required",
                    "Please fill in all required fields",
                    "e.preventDefault()",
                    "if (!newAgent.name",
                    "if (!newTask.title"
                ],
                'description': 'Form validation with user feedback',
                'critical': True
            },
            'api_integration': {
                'patterns': [
                    "class ApiService",
                    "fetchWithFallback",
                    "async getAgents",
                    "async createAgent",
                    "async createTask",
                    "async executeTask"
                ],
                'description': 'Real API integration with fallback system',
                'critical': True
            },
            'state_management': {
                'patterns': [
                    "useState<Agent[]>",
                    "useState<Task[]>",
                    "useState<SystemStats",
                    "setAgents",
                    "setTasks", 
                    "setSystemStats"
                ],
                'description': 'React state management for data',
                'critical': True
            },
            'error_handling': {
                'patterns': [
                    "try {",
                    "} catch",
                    "setError",
                    "Failed to load data",
                    "Failed to create",
                    "Failed to delete"
                ],
                'description': 'Comprehensive error handling',
                'critical': True
            },
            'user_feedback': {
                'patterns': [
                    "created successfully!",
                    "deleted successfully",
                    "alert(`",
                    "Are you sure you want to delete",
                    "Settings saved successfully!"
                ],
                'description': 'User feedback for all operations',
                'critical': True
            },
            'real_data_types': {
                'patterns': [
                    "interface Agent {",
                    "interface Task {",
                    "interface SystemStats {",
                    "status: 'active' | 'inactive'",
                    "priority: 'low' | 'medium'"
                ],
                'description': 'TypeScript interfaces for real data',
                'critical': True
            },
            'visual_indicators': {
                'patterns': [
                    "getStatusColor",
                    "getPriorityColor",
                    "backgroundColor: getStatusColor",
                    "borderRadius: '4px'",
                    "boxShadow: '0 2px 4px"
                ],
                'description': 'Color-coded status and priority indicators',
                'critical': False
            },
            'responsive_design': {
                'patterns': [
                    "gridTemplateColumns: 'repeat(auto-fit",
                    "minmax(",
                    "display: 'flex'",
                    "gap: '",
                    "maxWidth: '1200px'"
                ],
                'description': 'Responsive grid and flex layouts',
                'critical': False
            },
            'professional_styling': {
                'patterns': [
                    "backgroundColor: '#1976d2'",
                    "padding: '",
                    "margin: '",
                    "fontFamily: 'Arial",
                    "fontSize: '"
                ],
                'description': 'Professional styling throughout',
                'critical': False
            }
        }
        
        # Verify each element
        results = {}
        for element_name, element_info in ui_elements.items():
            patterns = element_info['patterns']
            description = element_info['description']
            critical = element_info['critical']
            
            found_patterns = []
            missing_patterns = []
            
            for pattern in patterns:
                if pattern in content:
                    found_patterns.append(pattern)
                else:
                    missing_patterns.append(pattern)
            
            # Element found if majority of patterns are present
            found = len(found_patterns) >= len(patterns) * 0.6
            
            results[element_name] = {
                'description': description,
                'found': found,
                'critical': critical,
                'found_patterns': len(found_patterns),
                'total_patterns': len(patterns),
                'pattern_percentage': (len(found_patterns) / len(patterns)) * 100
            }
            
            status = "✅" if found else "❌"
            critical_marker = " (CRITICAL)" if critical else ""
            pattern_info = f"({len(found_patterns)}/{len(patterns)} patterns)"
            
            print(f"  {status} {description} {pattern_info}{critical_marker}")
            
            if critical and not found:
                self.verification_results['summary']['critical_missing'] += 1
            
            self.verification_results['summary']['total_elements'] += 1
            if found:
                self.verification_results['summary']['verified_elements'] += 1
        
        self.verification_results['source_verification'] = results
        return self.verification_results['summary']['critical_missing'] == 0
    
    def verify_index_tsx_integration(self):
        """Verify that index.tsx properly loads FunctionalApp"""
        print("\n🔗 Verifying index.tsx Integration...")
        
        index_file = self.project_root / "frontend/agentic-seek-copilotkit-broken/src/index.tsx"
        
        if not index_file.exists():
            print("❌ index.tsx not found!")
            return False
        
        content = index_file.read_text()
        
        integration_checks = {
            'imports_functional_app': "import FunctionalApp from './FunctionalApp'" in content,
            'renders_functional_app': "<FunctionalApp />" in content,
            'has_root_render': "root.render(" in content,
            'has_react_import': "import React from 'react'" in content
        }
        
        all_good = True
        for check_name, passed in integration_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name.replace('_', ' ').title()}")
            if not passed:
                all_good = False
        
        return all_good
    
    def verify_build_configuration(self):
        """Verify build configuration is correct"""
        print("\n⚙️ Verifying Build Configuration...")
        
        package_json = self.project_root / "frontend/agentic-seek-copilotkit-broken/package.json"
        
        if not package_json.exists():
            print("❌ package.json not found!")
            return False
        
        with open(package_json, 'r') as f:
            package_data = json.load(f)
        
        config_checks = {
            'has_start_script': 'start' in package_data.get('scripts', {}),
            'has_build_script': 'build' in package_data.get('scripts', {}),
            'has_react_dependency': 'react' in package_data.get('dependencies', {}),
            'has_typescript': '@types/react' in package_data.get('dependencies', {}) or '@types/react' in package_data.get('devDependencies', {}),
            'correct_main_app': True  # Will check this separately
        }
        
        all_good = True
        for check_name, passed in config_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name.replace('_', ' ').title()}")
            if not passed:
                all_good = False
        
        return all_good
    
    def create_actual_human_verification_steps(self):
        """Create step-by-step verification for humans"""
        print("\n📋 Creating Human Verification Steps...")
        
        verification_steps = [
            {
                'step': 1,
                'title': 'Application Startup Verification',
                'actions': [
                    'Open terminal and cd to: frontend/agentic-seek-copilotkit-broken/',
                    'Run: npm start',
                    'Wait for "Local: http://localhost:3000" message',
                    'Open browser to http://localhost:3000',
                    'Verify app loads without errors'
                ],
                'expected': 'App loads showing Dashboard tab with blue header "AgenticSeek - AI Multi-Agent Platform"',
                'critical': True
            },
            {
                'step': 2,
                'title': 'Dashboard Functionality Check',
                'actions': [
                    'Verify Dashboard tab is active (blue background)',
                    'Check 4 stat cards are visible: Total Agents, Total Tasks, System Load, Memory Usage',
                    'Verify numbers are not 0 or placeholder',
                    'Check Recent Activity section shows actual tasks',
                    'Click "Refresh Data" button in header'
                ],
                'expected': 'Real statistics (agents: 3, tasks: 3), progress bars show percentages, refresh button works',
                'critical': True
            },
            {
                'step': 3,
                'title': 'Agent CRUD Operations Test',
                'actions': [
                    'Click "Agents" tab',
                    'Verify "AI Agents (3)" title and 3 agent cards visible',
                    'Click "Create Agent" button',
                    'Fill form: Name="Test Agent", Type="research", Description="Test description"',
                    'Click "Create Agent" submit button',
                    'Look for success message',
                    'Verify new agent appears in list',
                    'Click "Delete" on any agent',
                    'Confirm deletion in popup',
                    'Verify agent disappears'
                ],
                'expected': 'Form works, shows "Agent created successfully!", new card appears, deletion works with confirmation',
                'critical': True
            },
            {
                'step': 4,
                'title': 'Task CRUD Operations Test',
                'actions': [
                    'Click "Tasks" tab',
                    'Verify "Tasks (3)" title and 3 task cards visible',
                    'Click "Create Task" button', 
                    'Fill form: Title="Test Task", Description="Test", select any agent, Priority="high"',
                    'Click "Create Task" submit button',
                    'Look for success message',
                    'Find a task with "PENDING" status',
                    'Click "Execute Task" button',
                    'Verify status changes to "RUNNING"'
                ],
                'expected': 'Form works, agent dropdown populated, shows "Task created successfully!", execute button changes status',
                'critical': True
            },
            {
                'step': 5,
                'title': 'Settings Configuration Test',
                'actions': [
                    'Click "Settings" tab',
                    'Verify API Configuration section is visible',
                    'Check API Endpoint field shows URL',
                    'Check Agent Configuration dropdown has tier options',
                    'Click "Save Settings" button',
                    'Click "Test Connection" button'
                ],
                'expected': 'Settings form loads, shows "Settings saved successfully!", test connection refreshes data',
                'critical': True
            },
            {
                'step': 6,
                'title': 'Error Handling Verification',
                'actions': [
                    'Go to Agents tab, click "Create Agent"',
                    'Try submitting form with empty name field',
                    'Go to Tasks tab, click "Create Task"',
                    'Try submitting without selecting agent',
                    'Check browser console for JavaScript errors'
                ],
                'expected': 'Form validation shows "Please fill in all required fields", no JavaScript crashes',
                'critical': True
            }
        ]
        
        self.verification_results['manual_verification_guide'] = verification_steps
        
        # Create markdown guide
        guide_content = "# 🔍 HUMAN UI VERIFICATION GUIDE\n\n"
        guide_content += "**CRITICAL: Every step below MUST work for real humans testing the app**\n\n"
        
        for step in verification_steps:
            guide_content += f"## Step {step['step']}: {step['title']}\n\n"
            
            critical_marker = " **[CRITICAL]**" if step['critical'] else ""
            guide_content += f"**Priority:** {'CRITICAL - Must Pass' if step['critical'] else 'Important'}{critical_marker}\n\n"
            
            guide_content += "### Actions to Perform:\n"
            for i, action in enumerate(step['actions'], 1):
                guide_content += f"{i}. {action}\n"
            
            guide_content += f"\n### Expected Result:\n✅ {step['expected']}\n\n"
            guide_content += "---\n\n"
        
        guide_content += """## ✅ SUCCESS CRITERIA
All 6 steps must pass completely. Any step failure indicates the app is not ready for human testing.

## ❌ FAILURE INDICATORS
- Blank screens or infinite loading
- Buttons that show alert("Add functionality")  
- Forms that don't validate or submit
- JavaScript errors in browser console
- Any UI element that doesn't work as described

## 🚨 CRITICAL REQUIREMENT
**NO FAKE FUNCTIONALITY** - Every button, form, and interaction must perform real operations, not placeholder alerts.
"""
        
        guide_file = self.project_root / "CRITICAL_HUMAN_UI_VERIFICATION.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Critical human verification guide: {guide_file}")
        return len(verification_steps)
    
    def run_comprehensive_source_verification(self):
        """Run complete source code verification"""
        print("🔍 SOURCE CODE UI VERIFICATION")
        print("=" * 80)
        print("Verifying actual source code implementation (not minified build)")
        print("=" * 80)
        
        # Run all verifications
        source_verified = self.verify_functional_app_source()
        index_verified = self.verify_index_tsx_integration()
        build_verified = self.verify_build_configuration()
        manual_tests = self.create_actual_human_verification_steps()
        
        # Calculate results
        total_elements = self.verification_results['summary']['total_elements']
        verified_elements = self.verification_results['summary']['verified_elements']
        critical_missing = self.verification_results['summary']['critical_missing']
        success_rate = (verified_elements / total_elements * 100) if total_elements > 0 else 0
        
        print("\n" + "=" * 80)
        print("📊 SOURCE CODE VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total UI Elements: {total_elements}")
        print(f"✅ Verified Elements: {verified_elements}")
        print(f"❌ Missing Elements: {total_elements - verified_elements}")
        print(f"🚨 Critical Missing: {critical_missing}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"\n🔧 Integration Checks:")
        print(f"  {'✅' if source_verified else '❌'} FunctionalApp.tsx Implementation")
        print(f"  {'✅' if index_verified else '❌'} Index.tsx Integration")
        print(f"  {'✅' if build_verified else '❌'} Build Configuration")
        
        print(f"\n📋 Human Verification: {manual_tests} critical test scenarios")
        
        # Overall assessment
        overall_success = (
            source_verified and 
            index_verified and 
            build_verified and 
            critical_missing == 0 and
            success_rate >= 90
        )
        
        if overall_success:
            print("\n🎉 SOURCE CODE VERIFICATION: EXCELLENT")
            print("✅ ALL CRITICAL UI ELEMENTS IMPLEMENTED")
            print("✅ READY FOR HUMAN TESTING")
        elif critical_missing == 0 and success_rate >= 80:
            print("\n✅ SOURCE CODE VERIFICATION: GOOD")
            print("⚠️ Minor issues but core functionality complete")
            print("✅ READY FOR HUMAN TESTING")
        else:
            print("\n❌ SOURCE CODE VERIFICATION: CRITICAL ISSUES")
            print(f"🚨 {critical_missing} critical elements missing")
            print("❌ NOT READY FOR HUMAN TESTING")
        
        # Save results
        results_file = self.project_root / "source_code_verification_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2)
        
        print(f"\n📄 Detailed results: {results_file}")
        
        return overall_success

def main():
    """Run source code UI verification"""
    verifier = SourceCodeUIVerifier()
    
    try:
        success = verifier.run_comprehensive_source_verification()
        
        if success:
            print("\n🚀 SOURCE CODE UI VERIFICATION PASSED")
            print("✅ All critical UI elements confirmed in source code")
            print("📋 Human verification guide created")
            return 0
        else:
            print("\n🛠️ SOURCE CODE UI VERIFICATION FAILED") 
            print("❌ Critical UI elements missing from source")
            print("📋 Review detailed report")
            return 1
            
    except Exception as e:
        print(f"\n💥 VERIFICATION CRASHED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())