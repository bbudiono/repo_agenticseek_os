# CLAUDE.md - AgenticSeek Development Guide

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the AgenticSeek codebase.

## 1. Project Overview & Quick Start

AgenticSeek is a comprehensive **100% local alternative to Manus AI** - a privacy-focused, voice-enabled AI assistant system that autonomously browses the web, writes code, and plans tasks while keeping all data on the user's device. The project features a sophisticated multi-agent architecture with specialized agents for different tasks, designed specifically for local reasoning models with zero cloud dependency.

### Critical Development Rules

1. **Privacy-First Mandate**: ALL processing must operate locally by default. Cloud integrations are optional and must be explicitly configured by users with clear warnings.

2. **Multi-Agent Architecture Integrity**: Maintain the 6-agent system (Casual, Coder, File, Browser, Planner, MCP). Changes to one agent must consider impacts on routing system and other agents.

3. **Container-First Development**: Use Docker Compose for consistent development environments. Test all changes in containerized setup before deployment.

4. **Safety & Security**: Code execution must be properly sandboxed. Web automation must use stealth capabilities to avoid detection.

5. **Local LLM Priority**: Always prefer local LLM providers (Ollama, LM-Studio) over cloud providers. Cloud usage requires explicit user consent.

6. **Testing Strategy**: Validate agent routing accuracy, LLM integrations, tool execution safety, and complete multi-agent workflows.

## 2. Development Environment & Build System

### Core Services Architecture
- **Backend API**: FastAPI server on port 8001 (host machine)
- **Frontend**: React application on port 3000 (Docker container)
- **Redis**: Task queue and caching on port 6379 (Docker container)
- **SearXNG**: Local search engine on port 8080 (Docker container)
- **LLM Server**: Optional distributed LLM on port 11434 (Ollama)
- **macOS Native App**: SwiftUI wrapper for enhanced user experience

### Essential Build Commands

#### Full System Startup
```bash
# Start all services with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs for debugging
docker-compose logs -f

# Start backend (must run on host due to chromedriver)
python api.py
```

#### Development Setup
```bash
# Python environment setup
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Download ML routing models
cd llm_router && ./dl_safetensors.sh

# Frontend development
cd frontend/agentic-seek-front
npm install
npm start  # Development server
npm run build  # Production build
```

#### macOS Native App Development
```bash
# Build macOS app
cd _macOS
xcodebuild -project "AgenticSeek.xcodeproj" -scheme "AgenticSeek" build

# Run macOS app (connects to localhost backend)
open _macOS/AgenticSeek.app
```

#### Testing & Validation
```bash
# Comprehensive backend testing
python test_suite.py
python comprehensive_test_suite.py
python headless_test_suite.py

# API endpoint testing
python _macOS/tests/test_endpoints.py

# Agent routing accuracy testing
python -c "from sources.router import test_routing_accuracy; test_routing_accuracy()"

# Browser automation testing
python fully_headless_e2e_test.py
```

#### Configuration Management
```bash
# Validate configuration
python -c "import configparser; c = configparser.ConfigParser(); c.read('config.ini'); print('Config valid')"

# Test LLM provider connectivity
python -c "from sources.llm_provider import test_providers; test_providers()"

# Agent routing diagnostics
python -c "from sources.router import diagnose_routing; diagnose_routing()"
```

### Build Failure Recovery Protocol

1. **Immediate Actions**:
   - Stop all running services: `docker-compose down`
   - Check Docker container status: `docker-compose ps`
   - View service logs: `docker-compose logs --tail=50`
   - Document failure context and error messages

2. **Diagnostic Commands** (Run in Parallel):
   ```bash
   # Check Docker services
   docker-compose ps &
   docker-compose logs --tail=50 &
   
   # Validate Python environment
   python -c "import sources.agents; print('Agents module OK')" &
   python -c "import sources.tools; print('Tools module OK')" &
   
   # Test LLM connectivity
   curl -X POST http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"test"}' &
   
   # Check Redis connectivity
   redis-cli ping &
   
   # Test SearXNG
   curl http://localhost:8080/search?q=test&format=json &
   wait
   ```

3. **Recovery Steps**:
   - Restart failed services: `docker-compose restart [service-name]`
   - Rebuild containers: `docker-compose build --no-cache`
   - Check port conflicts and resource constraints
   - Verify LLM model availability
   - Test agent routing functionality
   - Validate browser automation setup

## 3. Architecture & Project Structure

### Directory Structure
```
/{root}/
├── _localhost/                             # Core platform (suggested reorganization)
│   ├── sources/                           # Core Python backend
│   │   ├── agents/                        # Multi-agent system (6 specialized agents)
│   │   ├── tools/                         # Execution tools (Python, Bash, web search)
│   │   ├── browser.py                     # Web automation capabilities
│   │   ├── llm_provider.py               # LLM abstraction layer
│   │   ├── router.py                      # Intelligent agent routing
│   │   ├── cascading_provider.py         # Provider fallback system
│   │   └── memory.py                      # Conversation memory management
│   ├── frontend/                          # React web interface
│   │   └── agentic-seek-front/           # React application
│   ├── prompts/                           # Agent personality systems
│   │   ├── base/                         # Default agent prompts
│   │   └── jarvis/                       # Iron Man inspired personality
│   ├── llm_server/                        # Custom LLM server for distributed setup
│   ├── llm_router/                        # Agent routing model and configuration
│   ├── searxng/                           # Local search engine configuration
│   ├── docs/                              # Technical documentation
│   ├── tests/                             # Comprehensive test suites
│   ├── api.py                             # FastAPI backend server
│   ├── cli.py                             # Command-line interface
│   ├── config.ini                         # Central configuration
│   └── docker-compose.yml                 # Service orchestration
├── _macOS/                                # macOS native wrapper
│   ├── AgenticSeek/                       # SwiftUI native app
│   │   ├── AgenticSeekApp.swift          # Main app entry point
│   │   ├── ContentView.swift             # Main UI (needs refactoring)
│   │   ├── WebViewManager.swift          # React frontend integration
│   │   ├── ServiceManager.swift          # Docker service management
│   │   ├── ConfigurationView.swift       # Model provider configuration
│   │   └── ModelManagementView.swift     # Local model management
│   ├── tests/                             # Native app testing
│   │   ├── test_endpoints.py             # API testing framework
│   │   └── test_model_api.py             # Model API testing
│   └── AgenticSeek.xcodeproj             # Xcode project
└── claude/                                # Claude Code configuration
    └── CLAUDE.md                          # This file
```

### Core Components

1. **Multi-Agent System** (`sources/agents/`)
   - **CasualAgent.py**: General conversation and queries
   - **CoderAgent.py**: Programming tasks and code execution in multiple languages
   - **FileAgent.py**: File system operations and document management
   - **BrowserAgent.py**: Autonomous web browsing and form automation
   - **PlannerAgent.py**: Complex multi-step task planning and coordination
   - **McpAgent.py**: Model Context Protocol integration (under development)

2. **Tool Execution System** (`sources/tools/`)
   - **Code Execution**:
     - `PyInterpreter.py` - Python code execution with sandboxing
     - `BashInterpreter.py` - Shell command execution
     - `C_Interpreter.py`, `GoInterpreter.py`, `JavaInterpreter.py` - Multi-language support
   - **Web Automation**:
     - `webSearch.py` - SearXNG integration for web search
     - `tools.py` - Browser automation using Selenium
   - **File Management**:
     - `fileFinder.py` - File system operations
   - **Safety & Security**:
     - `safety.py` - Execution safety checks and validation

3. **LLM Provider Integration** (`sources/llm_provider.py`)
   - **Local Providers**:
     - Ollama integration for local models
     - LM-Studio compatibility
     - Custom LLM server support
   - **Cloud Providers** (Optional):
     - OpenAI API integration
     - Google Gemini support
     - Anthropic Claude integration
     - DeepSeek API integration
   - **Routing Intelligence**:
     - Adaptive classifier for agent selection
     - Performance monitoring and optimization
     - Cost-aware provider switching

4. **Frontend Interface** (`frontend/agentic-seek-front/`)
   - **Core Components**:
     - `App.js` - Main React application
     - Chat interface for AI conversations
     - Agent selector and routing controls
   - **Configuration**:
     - LLM provider configuration interface
     - Local/cloud model selection
     - Voice controls for speech interface
   - **Task Management**:
     - Complex task breakdown interface
     - Real-time execution tracking
     - Output and result visualization

5. **macOS Native App** (`_macOS/AgenticSeek/`)
   - **Native Integration**:
     - SwiftUI interface wrapping React frontend
     - Docker service management
     - Model download and management
   - **System Integration**:
     - Menu bar support
     - macOS-specific features
     - Native file access

## 4. Development Workflow & Task Management

### Standard Development Process

1. **Environment Setup**: Start all services with `docker-compose up -d`
2. **Agent Development**: Implement changes in the appropriate agent module
3. **Tool Integration**: Add or modify tools in the `sources/tools/` directory
4. **Router Training**: Update routing classifier if needed (see `llm_router/`)
5. **Testing**: Run agent tests, tool tests, and integration tests
6. **Configuration Update**: Modify `config.ini` for new features or providers
7. **Frontend Integration**: Update React components if UI changes are needed
8. **macOS App Integration**: Update native app if system integration needed
9. **Service Validation**: Ensure all Docker services remain functional

### Multi-Agent Development Workflow

1. **Agent Selection**: Choose the appropriate agent (Casual, Coder, File, Browser, Planner, MCP)
2. **Prompt Engineering**: Update agent prompts in `prompts/base/` or `prompts/jarvis/`
3. **Tool Integration**: Ensure agent can access required tools
4. **Router Training**: Update routing classifier if needed
5. **Cross-Agent Testing**: Validate agent coordination and handoffs
6. **Performance Validation**: Test agent routing accuracy (>90% target)

### Multi-Agent Task Processing Workflow

1. **Task Reception**:
   - User input via CLI, API, React frontend, or macOS app
   - Input validation and sanitization
   - Language detection and translation if needed
   - Initial task classification and complexity estimation

2. **Agent Routing**:
   - Complex router uses ML models (BART + LLM) for sophisticated agent selection
   - Simple router provides fallback mechanism for reliability
   - Context and conversation history considered
   - Planner agent invoked for multi-step tasks

3. **Task Execution**:
   - Selected agent processes the request using available tools
   - Real-time progress tracking and status updates
   - Tool execution with proper sandboxing and safety checks
   - Cross-agent communication and coordination

4. **Result Processing**:
   - Output validation and formatting
   - Memory storage for conversation context
   - User feedback integration
   - Performance metrics collection

### Task Prioritization System

- **P0 (Critical)**: Service failures, security vulnerabilities, agent routing failures, privacy breaches
- **P1 (High)**: Agent accuracy, tool execution reliability, LLM integration stability, user experience
- **P2 (Medium)**: Performance optimizations, new tools, UI/UX enhancements, additional LLM providers
- **P3 (Low)**: Documentation, developer experience, code quality improvements

### Code Quality Requirements

- **Agent Routing Accuracy**: Maintain >90% correct agent selection
- **Tool Execution Safety**: 100% success rate for sandboxed execution
- **Test Coverage**: Minimum 80% code coverage for agent routing and tool execution
- **Documentation**: All agents and tools must have comprehensive Python docstrings
- **Performance**: Agent routing < 500ms, tool execution < 30s, API responses < 2s
- **Privacy**: All processing must work offline, cloud integrations optional with explicit consent
- **Safety**: Code execution must be properly sandboxed and validated

## 5. UI/UX Design Standards & Component Library

### Design System Compliance (.cursorrules)

All UI development must strictly adhere to the AgenticSeek .cursorrules file located at the project root. Key requirements include:

#### Colors (AI Assistant Theme) - MANDATORY
- **Primary**: `DesignSystem.Colors.primary` (#2563EB - AI Technology Blue) - AI assistant and technology
- **Secondary**: `DesignSystem.Colors.secondary` (#059669 - Success Green) - Success and completion
- **Agent**: `DesignSystem.Colors.agent` (#7C3AED - Violet) - Agent identification and routing
- **Code**: `DesignSystem.Colors.code` (#1F2937 - Gray 800) - Code blocks and technical content
- **Error**: `DesignSystem.Colors.error` (#DC2626 - Red 600) - Errors and critical actions
- **Warning**: `DesignSystem.Colors.warning` (#F59E0B - Amber 500) - Warnings and cautions
- **Background**: `#F9FAFB` (Gray 50) - Clean workspace
- **Surface**: `#FFFFFF` (White) - Chat bubbles and cards
- **NEVER** hardcode color values - always reference DesignSystem.Colors

#### Typography (AI Interface) - MANDATORY
- **ALWAYS** use DesignSystem.Typography font definitions
- **Primary Fonts**: Inter for UI text, Fira Code for code blocks
- **Code Text**: `DesignSystem.Typography.code` (Fira Code) for all code execution and technical content
- **Chat Text**: `DesignSystem.Typography.body` (Inter) for conversation interface
- **Agent Labels**: `DesignSystem.Typography.title2` for agent identification
- **Hierarchy**: Use proper heading hierarchy (headline → title1 → title2 → body)

#### Spacing System (4pt Grid) - MANDATORY
- **ALWAYS** use DesignSystem.Spacing values
- **Base Unit**: 4pt grid system (xxxs=2pt, xxs=4pt, xs=8pt, sm=12pt, md=16pt, lg=20pt, xl=24pt, xxl=32pt, xxxl=40pt)
- **Component Padding**: Use semantic spacing (chatPadding, cardPadding, buttonPadding, agentPadding)
- **NEVER** use arbitrary spacing values

#### Corner Radius System - MANDATORY
- **Chat Bubbles**: `DesignSystem.CornerRadius.message` (16pt)
- **Cards**: `DesignSystem.CornerRadius.card` (12pt)
- **Buttons**: `DesignSystem.CornerRadius.button` (8pt)
- **Input Fields**: `DesignSystem.CornerRadius.textField` (6pt)
- **Agent Avatars**: `DesignSystem.CornerRadius.avatar` (1000pt - fully rounded)

### Component Standards (.cursorrules MANDATORY)

#### Agent Interface Components
- **Agent Avatars**: Use `.agentAvatarStyle(agent:)` modifier with color-coded identification
- **Message Bubbles**: Use `.messageBubbleStyle(isUser:)` for chat interface
- **Agent Selector**: Use `.agentSelectorStyle()` for routing controls
- **Status Indicators**: Use `.statusIndicatorStyle(status:)` for service health
- **ALWAYS** include agent type identification in UI elements

#### Chat Interface Components
- **Chat Input**: Use `.chatInputStyle()` modifier for message input
- **Message List**: Use `.messageListStyle()` for conversation display
- **Typing Indicators**: Use `.typingIndicatorStyle()` for agent processing feedback
- **Code Blocks**: Use `.codeBlockStyle()` with syntax highlighting
- **Tool Output**: Use `.toolOutputStyle()` for execution results

#### Configuration Components
- **Provider Selector**: Use `.providerSelectorStyle()` for LLM provider selection
- **Model Selector**: Use `.modelSelectorStyle()` for model configuration
- **Privacy Toggle**: Use `.privacyToggleStyle()` for local/cloud warnings
- **API Key Input**: Use `.secureInputStyle()` for sensitive configuration
- **Service Status**: Use `.serviceStatusStyle()` for Docker service monitoring

#### AI Assistant Specific Requirements
- **Agent Identification**: Each agent type must have distinct color identification
- **Privacy Indicators**: Clear visual indication when processing locally
- **Cloud Warnings**: Prominent warnings for cloud LLM usage
- **Multi-Agent Coordination**: Visual representation of agent handoffs and task routing

### Accessibility Requirements (CRITICAL)

#### AI Interface Accessibility
- **Screen Reader**: All agent communications properly announced with context
- **Contrast**: WCAG AAA compliance for all text and UI elements
- **Keyboard Navigation**: Full keyboard support for chat interface and controls
- **Voice Integration**: Built-in speech-to-text and text-to-speech capabilities
- **Agent Identification**: Clear audio cues for different agents

#### Code Execution Accessibility
- **Code Announcements**: Screen reader support for code blocks and execution results
- **Execution Status**: Audio feedback for long-running operations
- **Error Navigation**: Keyboard shortcuts for reviewing errors and outputs
- **Tool Feedback**: Clear indication of tool usage and results

### Animation and Motion

- **Message Delivery**: Smooth chat bubble animations with agent transitions
- **Agent Switching**: Subtle transitions between agents with visual continuity
- **Code Execution**: Progress indicators for long operations with real-time updates
- **Voice Recognition**: Visual feedback for speech input and processing
- **Task Completion**: Satisfying completion animations for successful operations
- **Service Status**: Animated indicators for service health and connectivity

## 6. Testing Strategy & User-Centric Quality Assurance

### Testing Hierarchy (User-Centric Focus)

1. **User Experience Tests**: Primary focus on real user scenarios and task completion
2. **Accessibility Deep Tests**: WCAG AAA compliance and universal design validation
3. **Agent Interaction Tests**: User-centric agent routing and coordination testing
4. **Layout & Responsive Tests**: Dynamic Type, window sizing, and visual hierarchy
5. **Content Quality Tests**: Information architecture and content usefulness validation
6. **Performance-UX Tests**: Performance impact on user experience and satisfaction
7. **Edge Case User Impact**: How boundary conditions affect real user workflows
8. **State Management UX**: User impact of state preservation and restoration
9. **Navigation Flow Tests**: Complete user journey validation and optimization

### User-Centric Quality Assurance Framework

#### Comprehensive Testing Structure
```
Tests/
├── SwiftUI-Compliance/          # SwiftUI technical implementation testing
├── User-Experience/             # User-centric design and usability testing  
├── Layout-Validation/           # Responsive design and spacing validation
├── Content-Auditing/            # Content quality and information architecture
├── Accessibility-Deep/          # Comprehensive accessibility compliance
├── Performance-UX/              # Performance impact on user experience
├── Edge-Cases/                  # Boundary conditions and error scenarios
├── State-Management/            # SwiftUI state and data flow testing
└── Navigation-Flow/             # User journey and navigation testing
```

#### Critical User-Centric Success Criteria
- **Zero placeholder content** in production builds
- **100% VoiceOver compatibility** for all interactive elements
- **WCAG AAA compliance** for color contrast and accessibility
- **<100ms UI response time** for 95% of user interactions
- **>95% task completion rate** for primary user workflows
- **User satisfaction scores >4.5/5.0** across all user personas
- **Accessibility task completion rate >90%** for users with disabilities

### User-Centric Agent System Testing Requirements

- **User-Perceived Routing Accuracy**: >95% user satisfaction with agent selection
- **Task Completion Success**: >95% successful completion of user-intended tasks
- **Agent Response Comprehension**: >90% user understanding of agent responses
- **Tool Execution Transparency**: Users understand what tools are being used and why
- **Error Recovery User Experience**: >90% user satisfaction with error handling and recovery
- **Multi-Agent Coordination Clarity**: Users understand task handoffs and agent collaboration

### Feedback-Driven Continuous Improvement Protocol

#### Error and Feedback Response Framework
When any user feedback indicates issues or errors are discovered, the following action plan must be executed:

##### Immediate Response (Within 24 Hours)
1. **Issue Classification**: Categorize by user impact severity
   - P0: Prevents primary task completion or violates accessibility
   - P1: Significantly degrades user experience
   - P2: Minor user experience impact
   - P3: Enhancement opportunities

2. **User Impact Assessment**: Determine affected user populations
   - Primary user personas affected
   - Accessibility impact assessment
   - Task completion impact analysis
   - User satisfaction degradation measurement

3. **Root Cause Analysis**: Identify underlying system issues
   - SwiftUI implementation problems
   - Design system compliance failures
   - User experience design flaws
   - Content clarity and usefulness issues

##### Testing Response Protocol (Within 1 Week)
1. **Enhanced Test Coverage Creation**:
   - Create specific test cases for the identified issue
   - Develop regression tests to prevent recurrence
   - Add user scenario tests covering the failure case
   - Implement automated checks where possible

2. **Expanded Testing Scope**:
   - Test similar patterns throughout the application
   - Validate related user workflows and scenarios
   - Check for systemic issues in design or implementation
   - Verify accessibility impact across all user types

3. **User Validation Testing**:
   - Conduct targeted user testing with affected personas
   - Validate proposed solutions with real users
   - Measure improvement in user satisfaction and task completion
   - Verify accessibility improvements with assistive technology users

##### Long-term Improvement Integration (Within 1 Month)
1. **Process Enhancement**:
   - Update testing frameworks to catch similar issues
   - Enhance design system guidelines to prevent recurrence
   - Improve development process documentation
   - Add preventive measures to coding standards

2. **Continuous Monitoring Implementation**:
   - Set up metrics to track related user experience indicators
   - Implement automated alerts for similar issues
   - Establish regular user feedback collection and analysis
   - Create dashboard for tracking user-centric quality metrics

### User-Centric Test Coverage Requirements

- **User Task Completion**: 100% coverage for all primary user workflows
- **Accessibility Compliance**: 100% WCAG AAA compliance across all interfaces
- **Content Quality**: 100% validation of user-facing content for clarity and usefulness
- **Error Scenario Recovery**: 100% coverage of error states with user-friendly resolution paths
- **Agent Interaction UX**: 100% testing of user understanding and satisfaction with agent behavior
- **Layout Responsiveness**: 100% testing across all Dynamic Type sizes and window configurations
- **Performance Impact on UX**: Continuous monitoring of performance degradation effects on user experience

### User-Centric Testing Commands and Validation

#### Daily User Experience Validation
```bash
# User-centric testing suite execution
python _macOS/Tests/User-Experience/run_user_scenario_tests.py --comprehensive

# Accessibility compliance validation
python _macOS/Tests/Accessibility-Deep/run_accessibility_audit.py --wcag-aaa

# Layout and responsive design testing
python _macOS/Tests/Layout-Validation/run_dynamic_layout_tests.py --all-sizes

# Content quality and usefulness validation
python _macOS/Tests/Content-Auditing/run_content_quality_tests.py --real-scenarios
```

#### Weekly User Experience Audits
```bash
# Comprehensive user workflow testing
python _macOS/Tests/Navigation-Flow/run_complete_user_journeys.py --all-personas

# SwiftUI user experience compliance
python _macOS/Tests/SwiftUI-Compliance/run_ux_compliance_tests.py --design-system

# Performance impact on user experience
python _macOS/Tests/Performance-UX/run_ux_performance_tests.py --user-perception

# Edge case user impact assessment
python _macOS/Tests/Edge-Cases/run_user_impact_edge_cases.py --real-world-scenarios
```

#### Monthly User Satisfaction Monitoring
```bash
# User satisfaction metrics collection
python _macOS/Tests/collect_user_satisfaction_metrics.py --all-personas

# Accessibility user testing coordination
python _macOS/Tests/coordinate_assistive_technology_testing.py --comprehensive

# Long-term user experience trend analysis
python _macOS/Tests/analyze_ux_trends.py --longitudinal-data

# Continuous improvement recommendation generation
python _macOS/Tests/generate_ux_improvement_recommendations.py --data-driven
```

### Testing Commands (Parallel Execution)

```bash
# Run comprehensive test suite
python test_suite.py --parallel --coverage

# Run specific test categories in parallel
python comprehensive_test_suite.py --agents &
python headless_test_suite.py --browser &
python _macOS/tests/test_endpoints.py --api &
wait

# Agent routing accuracy tests
python -c "
from sources.router import test_routing_accuracy, validate_agent_selection
test_routing_accuracy(min_accuracy=0.9)
validate_agent_selection(test_cases=1000)
"

# Tool execution safety tests
python -c "
from sources.tools.safety import run_safety_tests
from sources.tools.PyInterpreter import test_sandboxing
run_safety_tests()
test_sandboxing()
"

# LLM provider integration tests
python -c "
from sources.llm_provider import test_all_providers
from sources.cascading_provider import test_failover
test_all_providers()
test_failover()
"

# Browser automation tests
python fully_headless_e2e_test.py --stealth --comprehensive

# macOS app integration tests
cd _macOS && xcodebuild test -scheme "AgenticSeek"

# Performance and load testing
python _macOS/tests/test_endpoints.py --load-testing --stress-testing
```

### Test Organization

```
tests/
├── unit/                          # Isolated component tests
│   ├── agents/                    # Individual agent tests
│   │   ├── test_casual_agent.py
│   │   ├── test_coder_agent.py
│   │   ├── test_browser_agent.py
│   │   └── test_planner_agent.py
│   ├── tools/                     # Tool execution tests
│   │   ├── test_python_execution.py
│   │   ├── test_browser_automation.py
│   │   └── test_safety_validation.py
│   └── routing/                   # Routing system tests
│       ├── test_agent_selection.py
│       └── test_routing_accuracy.py
├── integration/                   # Component interaction tests
│   ├── agent_coordination/        # Multi-agent collaboration tests
│   ├── llm_integration/          # LLM provider integration tests
│   └── tool_chaining/            # Multi-tool execution tests
├── e2e/                          # End-to-end workflow tests
│   ├── conversation_flows/       # Complete conversation tests
│   ├── task_execution/           # Complex task completion tests
│   └── multi_interface/          # Cross-interface testing (CLI, API, macOS)
├── performance/                   # Performance and load tests
│   ├── agent_routing/            # Routing performance benchmarks
│   ├── tool_execution/           # Tool execution performance
│   └── concurrent_sessions/      # Multi-user load tests
├── security/                     # Security and privacy tests
│   ├── sandboxing/              # Code execution safety tests
│   ├── data_privacy/            # Local processing validation
│   └── llm_security/            # LLM provider security tests
└── macOS/                        # Native app tests
    ├── service_integration/      # Docker service management tests
    ├── ui_integration/          # SwiftUI interface tests
    └── performance/             # Native app performance tests
```

## 7. Documentation Standards & Knowledge Management

### Key Documentation Files

- **README.md**: Project overview and quick start guide
- **ARCHITECTURE.md**: System design and component relationships (if exists)
- **BUILD_FIXES_SUMMARY.md**: Build issues and solutions tracking
- **TESTING_SUMMARY.md**: Testing approach and results summary
- **Agent Architecture Documentation**: Multi-agent system design and coordination
- **Tool Development Guide**: Adding new tools and capabilities
- **LLM Integration Guide**: Provider setup and configuration
- **Privacy and Security Guidelines**: Local processing and data protection

### Agent System Documentation

- **Agent Routing Documentation**: Classifier training and accuracy optimization
- **Agent Prompt Engineering**: Best practices for prompt design and testing
- **Multi-Agent Coordination**: Task handoff protocols and communication patterns
- **Tool Integration Patterns**: How agents interact with available tools
- **Performance Optimization**: Agent response time and resource usage optimization

### Documentation Requirements

1. **Agent Documentation**: All agents must have comprehensive capability descriptions
2. **Tool Documentation**: All tools must document inputs, outputs, and safety measures
3. **Architecture Decisions**: Document significant design choices in dedicated files
4. **API Documentation**: Maintain comprehensive API documentation for all endpoints
5. **Configuration Documentation**: Document all configuration options and their impacts
6. **Privacy Documentation**: Document data handling and local processing guarantees

### Documentation Update Protocol

1. **Before Implementation**: Update design documents and agent specifications
2. **During Development**: Maintain inline code documentation and comments
3. **After Implementation**: Update user guides, API docs, and architecture documents
4. **Regular Reviews**: Monthly documentation audits and accuracy verification
5. **Agent Updates**: Document changes to agent capabilities and routing logic

## 8. Security & Performance Guidelines

### Security Requirements (P0 CRITICAL)

1. **Privacy-First Architecture**: All processing operates locally by default with no data transmission
2. **Code Execution Safety**: Proper sandboxing for all code execution tools with validation
3. **LLM Provider Security**: Secure API key management with local storage and validation
4. **Browser Automation Security**: Stealth mode and anti-detection capabilities
5. **Input Validation**: Comprehensive sanitization of all user inputs and tool parameters
6. **Audit Logging**: Comprehensive logging without exposing sensitive data
7. **Agent Communication Security**: Secure inter-agent communication and data sharing

### Privacy Protection Standards

- **Local Processing Guarantee**: Core functionality works without internet connectivity
- **Cloud Provider Warnings**: Explicit user consent required for cloud LLM usage
- **Data Retention**: Conversation data stored locally with user control
- **Secure Storage**: API keys and sensitive configuration encrypted at rest
- **Network Isolation**: Optional network isolation mode for maximum privacy

### Performance Standards

- **Agent Routing**: < 500ms for agent selection and task routing
- **Tool Execution**: < 30s for standard operations, < 5min for complex tasks
- **API Response**: < 2s for typical API responses
- **Memory Usage**: < 1GB under normal operation with multiple agents
- **LLM Response**: < 10s for local models, < 30s for cloud models
- **Browser Automation**: Efficient page interaction with stealth capabilities
- **Multi-Agent Coordination**: Minimal latency for agent handoffs

### Performance Monitoring

```bash
# Monitor system resource usage
htop -p $(pgrep -f "python.*api.py")

# Monitor Docker service performance
docker stats

# Profile agent routing performance
python -c "
from sources.router import profile_routing_performance
profile_routing_performance(iterations=1000)
"

# Monitor tool execution performance
python -c "
from sources.tools.tools import profile_tool_execution
profile_tool_execution()
"

# Browser automation performance
python -c "
from sources.browser import profile_browser_performance
profile_browser_performance()
"

# LLM provider response times
python -c "
from sources.llm_provider import benchmark_providers
benchmark_providers()
"
```

## 9. LLM Provider Integration & AI Services

### LLM Provider Architecture

1. **Local Providers (Primary)**:
   - **Ollama**: Local model hosting with GPU acceleration
   - **LM-Studio**: User-friendly local model management
   - **Custom LLM Server**: Distributed local processing
   - **Benefits**: Complete privacy, no API costs, offline operation

2. **Cloud Providers (Optional)**:
   - **OpenAI**: GPT-4, GPT-3.5-turbo for high-quality responses
   - **Anthropic**: Claude models for reasoning and analysis
   - **Google**: Gemini models for fast, cost-effective processing
   - **DeepSeek**: Specialized coding and reasoning models
   - **Configuration**: API keys stored securely, explicit user consent required

3. **Cascading Provider System**:
   - Intelligent fallback between providers
   - Performance monitoring and optimization
   - Cost-aware provider switching
   - Failure recovery and retry logic

### Agent-Specific LLM Integration

- **CasualAgent**: General conversation models (local preferred)
- **CoderAgent**: Code-specialized models (DeepSeek, Code Llama)
- **BrowserAgent**: Instruction-following models with web context
- **PlannerAgent**: Reasoning-capable models for complex task breakdown
- **FileAgent**: Document processing and analysis models
- **McpAgent**: Protocol-specific models for MCP integration

### LLM Integration Security

1. **API Key Management**: Secure storage in configuration with encryption
2. **Local-First Priority**: Always prefer local models when available
3. **Privacy Controls**: User consent required for cloud provider usage
4. **Cost Monitoring**: Track and limit API usage with warnings
5. **Response Validation**: Validate and sanitize all LLM responses
6. **Audit Logging**: Log usage patterns without exposing content

### LLM Provider Configuration

```python
# Example LLM provider configuration in config.ini
[MAIN]
is_local = True
provider_name = ollama
provider_model = deepseek-r1:14b
fallback_provider = openai
fallback_model = gpt-4

[OLLAMA]
base_url = http://localhost:11434
models_path = ~/.ollama/models
gpu_acceleration = True

[OPENAI]
api_key_encrypted = true
model_preference = gpt-4
max_tokens = 4000
temperature = 0.7

[ANTHROPIC]
api_key_encrypted = true
model_preference = claude-3-sonnet
safety_filtering = true
```

## 10. Troubleshooting & Build Recovery

### Common Issues and Solutions

#### Docker Service Issues
1. **Service startup failures**: Check `docker-compose logs` for specific errors
2. **Port conflicts**: Verify ports 3000, 6379, 8080, 8001 are available
3. **Resource constraints**: Monitor Docker resource usage and limits
4. **Network connectivity**: Test inter-service communication
5. **Volume mounting**: Verify directory permissions and paths

#### Backend Application Issues
1. **Python dependencies**: Verify virtual environment and requirements
2. **Agent routing failures**: Check ML model availability in `llm_router/`
3. **Tool execution errors**: Validate sandboxing and safety configurations
4. **LLM provider connectivity**: Test API keys and network access
5. **Configuration errors**: Validate `config.ini` syntax and values

#### Browser Automation Issues
1. **Chromedriver compatibility**: Ensure Chrome and chromedriver versions match
2. **Stealth mode failures**: Verify undetected-chromedriver installation
3. **Network access**: Check firewall and proxy configurations
4. **Headless mode**: Validate display configuration for headless operation
5. **Session management**: Monitor browser session lifecycle

#### LLM Integration Issues
1. **Local model availability**: Verify Ollama models are downloaded
2. **API key validation**: Test cloud provider API keys and quotas
3. **Response parsing**: Validate LLM response format compatibility
4. **Provider failover**: Test cascading provider functionality
5. **Performance issues**: Monitor response times and resource usage

#### macOS App Issues
1. **Service discovery**: Fix hardcoded path assumptions
2. **Docker integration**: Verify Docker Desktop connectivity
3. **SwiftUI rendering**: Address layout and state management issues
4. **WebView integration**: Test React frontend loading
5. **Configuration persistence**: Implement settings storage

### Emergency Protocols

#### P0 Critical Violations (Privacy Breach, Security Failure)
1. **Immediate Stop**: Halt all development activities and running services
2. **Isolation**: Disconnect affected systems from network if necessary
3. **Assessment**: Document the violation context and potential impact
4. **Escalation**: Follow incident reporting procedures
5. **Recovery**: Implement fix with comprehensive validation
6. **Post-mortem**: Document lessons learned and prevention measures

#### Service Failure Recovery
1. **Service Restart**: `docker-compose restart [service-name]`
2. **Container Rebuild**: `docker-compose build --no-cache`
3. **Log Analysis**: Review service logs for error patterns
4. **Resource Check**: Monitor system resources and constraints
5. **Configuration Validation**: Verify service configurations
6. **Network Diagnostics**: Test inter-service connectivity

### Diagnostic Tools and Commands

```bash
# Comprehensive system diagnostics
./scripts/diagnose_system.sh --verbose

# Service health checks
docker-compose ps && docker-compose logs --tail=20

# Agent routing diagnostics
python -c "
from sources.router import diagnose_routing_system
diagnose_routing_system(verbose=True)
"

# Tool execution diagnostics
python -c "
from sources.tools.safety import run_diagnostic_tests
run_diagnostic_tests()
"

# LLM provider connectivity tests
python -c "
from sources.llm_provider import test_provider_connectivity
test_provider_connectivity(all_providers=True)
"

# Browser automation diagnostics
python -c "
from sources.browser import test_browser_setup
test_browser_setup()
"

# macOS app service integration test
cd _macOS && python tests/test_endpoints.py --health-check
```

## 11. Compliance Checklist & Verification

### Pre-Commit Checklist

- [ ] All tests passing (Unit, Integration, E2E, Agent routing, Tool execution)
- [ ] Docker services build and start successfully
- [ ] Agent routing accuracy maintained (>90% correct classification)
- [ ] Tool execution safety verified (sandboxing, validation)
- [ ] LLM integration stability tested (local and cloud providers)
- [ ] Code coverage maintained or improved (>80% for critical paths)
- [ ] Documentation updated (agent docs, tool docs, API docs)
- [ ] Security review completed (privacy, sandboxing, API keys)
- [ ] Performance impact assessed (routing times, tool execution, memory usage)
- [ ] Privacy compliance verified (local processing, optional cloud usage)
- [ ] Browser automation stealth capabilities tested
- [ ] macOS app integration validated (if applicable)

### Release Checklist

- [ ] Full test suite passing (100% for critical agent and tool paths)
- [ ] Performance benchmarks met (routing < 500ms, tools < 30s, API < 2s)
- [ ] Security audit completed (sandboxing validation, privacy review)
- [ ] Agent routing accuracy validation (>90% on comprehensive test dataset)
- [ ] LLM integration testing (local models, cloud providers, failover)
- [ ] Multi-agent workflow testing (task coordination, handoffs)
- [ ] Documentation up to date (user guides, agent docs, setup instructions)
- [ ] Privacy compliance verified (local processing, optional cloud usage)
- [ ] Docker deployment validation (all services, networking, persistence)
- [ ] Voice interface testing (speech-to-text, text-to-speech accuracy)
- [ ] Browser automation comprehensive testing (stealth mode, anti-detection)
- [ ] macOS app deployment testing (service integration, user experience)

### Quality Gates

1. **Code Review**: Minimum 1 approver for agent and tool changes
2. **Automated Testing**: All test suites must pass with adequate coverage
3. **Performance Testing**: No regression in agent routing or tool execution times
4. **Security Scanning**: Clean security scan results for privacy and sandboxing
5. **Agent Accuracy**: Maintain >90% agent routing accuracy benchmark
6. **Privacy Validation**: Verify local-first operation and optional cloud usage
7. **LLM Integration**: Stable connection and response handling for all providers
8. **Multi-Agent Coordination**: Successful task handoffs and collaboration

## 12. Sandbox Test-Driven Development (TDD) Framework

### Sandbox Environment Architecture (.cursorrules Compliance)

The Sandbox TDD approach provides an isolated development environment that strictly adheres to the `.cursorrules` design system requirements while enabling rapid prototyping, feature development, and comprehensive testing without affecting production code.

#### Core Sandbox Principles

1. **Complete Isolation**: Sandbox operates independently from main codebase
2. **Design System Enforcement**: 100% `.cursorrules` compliance validation
3. **TDD-First Development**: All features developed using test-driven methodology
4. **Rapid Iteration**: Fast feedback loops for development and testing
5. **Safe Experimentation**: Risk-free environment for trying new approaches
6. **Production Preparation**: Seamless transition from sandbox to production

#### Sandbox Directory Structure
```
_Sandbox/
├── Environment/                           # Isolated development environment
│   ├── TestDrivenFeatures/               # TDD feature development
│   │   ├── NewFeature_TDD/               # Individual feature TDD workspace
│   │   │   ├── 01_WriteTests/            # Write tests first (Red)
│   │   │   ├── 02_ImplementCode/         # Implement to pass tests (Green)
│   │   │   ├── 03_RefactorImprove/       # Refactor and improve (Refactor)
│   │   │   └── 04_ProductionReady/       # Production-ready implementation
│   │   └── FeaturePrototypes/            # Experimental feature prototypes
│   ├── DesignSystemValidation/           # .cursorrules compliance testing
│   │   ├── ColorSystemTesting/           # Color palette and usage validation
│   │   ├── TypographyTesting/            # Font and text hierarchy testing
│   │   ├── SpacingSystemTesting/         # 4pt grid system validation
│   │   └── ComponentLibraryTesting/      # Component design system testing
│   ├── UserExperienceLab/                # UX experimentation and testing
│   │   ├── PersonaBasedTesting/          # User persona simulation
│   │   ├── AccessibilityLab/             # Accessibility feature testing
│   │   ├── PerformanceTestbench/         # Performance optimization testing
│   │   └── InteractionDesignLab/         # Interaction pattern development
│   └── IntegrationStaging/               # Pre-production integration testing
└── Tools/                                # Sandbox-specific development tools
    ├── sandbox_tdd_runner.py             # Automated TDD workflow runner
    ├── design_system_validator.py        # .cursorrules compliance checker
    ├── feature_migration_tool.py         # Sandbox to production migration
    └── sandbox_cleanup_automation.py     # Environment cleanup and reset
```

### Sandbox TDD Methodology (Red-Green-Refactor-Deploy)

#### Phase 1: RED - Write Failing Tests First
```bash
# Sandbox TDD workflow initiation
cd _Sandbox/Environment/TestDrivenFeatures/NewFeature_TDD/01_WriteTests/

# Create comprehensive test suite before any implementation
python sandbox_tdd_runner.py --phase=red --feature="new_feature_name"
```

**Test Categories for Sandbox TDD**:
1. **Functional Tests**: Core feature functionality validation
2. **Design System Tests**: .cursorrules compliance verification
3. **User Experience Tests**: User-centric scenario validation
4. **Accessibility Tests**: WCAG AAA compliance testing
5. **Performance Tests**: Performance benchmark validation
6. **Integration Tests**: Component interaction validation

#### Phase 2: GREEN - Implement Minimal Code to Pass Tests
```bash
# Move to implementation phase
cd _Sandbox/Environment/TestDrivenFeatures/NewFeature_TDD/02_ImplementCode/

# Implement minimal code to pass all tests
python sandbox_tdd_runner.py --phase=green --feature="new_feature_name"
```

**Implementation Priorities**:
1. **Functionality First**: Make tests pass with minimal code
2. **Design System Compliance**: Ensure .cursorrules adherence
3. **Accessibility Integration**: Include accessibility from start
4. **Performance Awareness**: Consider performance implications early

#### Phase 3: REFACTOR - Improve Code Quality and Design
```bash
# Move to refactoring phase
cd _Sandbox/Environment/TestDrivenFeatures/NewFeature_TDD/03_RefactorImprove/

# Refactor while maintaining test passage
python sandbox_tdd_runner.py --phase=refactor --feature="new_feature_name"
```

**Refactoring Focus Areas**:
1. **Code Quality**: Clean code principles and maintainability
2. **Design System Optimization**: Enhanced .cursorrules compliance
3. **Performance Optimization**: Improved efficiency and responsiveness
4. **Accessibility Enhancement**: Advanced accessibility features
5. **User Experience Polish**: Refined user interactions and feedback

#### Phase 4: DEPLOY - Production Preparation and Migration
```bash
# Move to production preparation
cd _Sandbox/Environment/TestDrivenFeatures/NewFeature_TDD/04_ProductionReady/

# Prepare for production migration
python feature_migration_tool.py --validate --feature="new_feature_name"
python feature_migration_tool.py --migrate --feature="new_feature_name"
```

### Design System Enforcement in Sandbox

#### Automated .cursorrules Compliance Validation
```bash
# Continuous design system validation
python _Sandbox/Tools/design_system_validator.py --comprehensive

# Specific design system component testing
python _Sandbox/Tools/design_system_validator.py --colors --typography --spacing
```

**Design System Validation Categories**:
1. **Color System Compliance**: 100% DesignSystem.Colors usage
2. **Typography Hierarchy**: Proper font usage and text scaling
3. **Spacing System**: 4pt grid system adherence
4. **Component Standards**: Consistent component implementation
5. **Accessibility Integration**: Built-in accessibility compliance

#### Sandbox Design System Testing Framework
```swift
// Sandbox-specific design system testing
class SandboxDesignSystemTests: XCTestCase {
    func testColorSystemCompliance() {
        // Validate all colors use DesignSystem.Colors
        let sandboxColorUsage = scanSandboxForHardcodedColors()
        XCTAssertTrue(sandboxColorUsage.isEmpty, 
                     "Sandbox contains hardcoded colors: \(sandboxColorUsage)")
    }
    
    func testSpacingSystemCompliance() {
        // Validate 4pt grid system usage
        let spacingViolations = scanSandboxForArbitrarySpacing()
        XCTAssertTrue(spacingViolations.isEmpty,
                     "Sandbox spacing violations: \(spacingViolations)")
    }
}
```

### Sandbox User Experience Testing

#### Real User Persona Testing in Sandbox
```python
class SandboxUserPersonaTesting:
    def test_tech_novice_sandbox_experience(self):
        """Test new features with tech novice persona in sandbox"""
        # Isolated testing of sandbox features with specific personas
        
    def test_accessibility_user_sandbox_experience(self):
        """Test accessibility compliance in sandbox environment"""
        # Validate accessibility features before production deployment
        
    def test_power_user_sandbox_efficiency(self):
        """Test advanced user workflows in sandbox"""
        # Validate efficiency and advanced features
```

#### Sandbox Performance Benchmarking
```bash
# Sandbox-specific performance testing
cd _Sandbox/Environment/UserExperienceLab/PerformanceTestbench/

# Run performance benchmarks in isolation
python sandbox_performance_tests.py --feature="new_feature" --benchmarks=all
```

### Sandbox Integration with Main Development Workflow

#### Daily Sandbox Development Cycle
```bash
# Morning: Initialize sandbox TDD session
python _Sandbox/Tools/sandbox_tdd_runner.py --init --daily-session

# Development: TDD cycle execution
python _Sandbox/Tools/sandbox_tdd_runner.py --tdd-cycle --feature="current_feature"

# Evening: Validate and prepare for production consideration
python _Sandbox/Tools/sandbox_tdd_runner.py --validate --production-readiness
```

#### Weekly Sandbox Integration
```bash
# Weekly sandbox review and integration
python _Sandbox/Tools/feature_migration_tool.py --weekly-review

# Sandbox cleanup and optimization
python _Sandbox/Tools/sandbox_cleanup_automation.py --weekly-maintenance
```

#### Sandbox to Production Migration Protocol
```bash
# Pre-migration validation
python _Sandbox/Tools/feature_migration_tool.py --pre-migration-check --feature="feature_name"

# Design system compliance verification
python _Sandbox/Tools/design_system_validator.py --production-readiness --feature="feature_name"

# User experience validation
python _Sandbox/Tools/sandbox_tdd_runner.py --ux-validation --feature="feature_name"

# Performance impact assessment
python _Sandbox/Tools/sandbox_performance_tests.py --production-impact --feature="feature_name"

# Execute migration
python _Sandbox/Tools/feature_migration_tool.py --migrate --feature="feature_name" --confirmed
```

### Sandbox Testing Automation

#### Automated Sandbox TDD Runner
```python
#!/usr/bin/env python3
"""
Sandbox TDD Automation Framework
Manages the complete Red-Green-Refactor-Deploy cycle in isolated environment
"""

import os
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional

class SandboxTDDRunner:
    def __init__(self, sandbox_root: str = "_Sandbox"):
        self.sandbox_root = sandbox_root
        self.current_feature = None
        self.current_phase = None
    
    def init_tdd_cycle(self, feature_name: str) -> bool:
        """Initialize new TDD cycle for feature in sandbox"""
        # Create isolated feature development environment
        # Set up test framework
        # Initialize design system validation
        # Create user persona testing setup
        
    def run_red_phase(self, feature_name: str) -> bool:
        """Execute RED phase - write failing tests first"""
        # Create comprehensive test suite
        # Validate tests fail appropriately
        # Ensure design system test coverage
        # Include accessibility test requirements
        
    def run_green_phase(self, feature_name: str) -> bool:
        """Execute GREEN phase - implement to pass tests"""
        # Implement minimal functionality
        # Validate test passage
        # Ensure design system compliance
        # Maintain accessibility requirements
        
    def run_refactor_phase(self, feature_name: str) -> bool:
        """Execute REFACTOR phase - improve while maintaining tests"""
        # Optimize code quality
        # Enhance design system compliance
        # Improve performance
        # Polish user experience
        
    def validate_production_readiness(self, feature_name: str) -> Dict:
        """Validate feature readiness for production migration"""
        # Comprehensive testing validation
        # Design system compliance verification
        # Performance impact assessment
        # User experience validation
```

### Sandbox Quality Assurance Integration

#### Sandbox-Specific Quality Gates
1. **TDD Completion**: All TDD phases successfully completed
2. **Design System Compliance**: 100% .cursorrules adherence
3. **User Experience Validation**: All personas successfully tested
4. **Accessibility Compliance**: WCAG AAA standards met
5. **Performance Benchmarks**: No regression in performance metrics
6. **Integration Testing**: Seamless integration with existing codebase

#### Sandbox Success Metrics
- **TDD Cycle Time**: Target <4 hours for Red-Green-Refactor cycle
- **Design System Compliance**: 100% automated validation success
- **Test Coverage**: >95% code coverage in sandbox features
- **User Experience Score**: >4.8/5.0 across all personas
- **Performance Impact**: <5% degradation in critical metrics
- **Migration Success**: >98% successful sandbox to production migrations

### Sandbox Development Commands

#### Essential Sandbox Workflow
```bash
# Initialize new sandbox feature development
python _Sandbox/Tools/sandbox_tdd_runner.py --init-feature="feature_name"

# Run complete TDD cycle
python _Sandbox/Tools/sandbox_tdd_runner.py --full-tdd-cycle="feature_name"

# Validate design system compliance
python _Sandbox/Tools/design_system_validator.py --validate="feature_name"

# Test user experience in sandbox
python _Sandbox/Tools/sandbox_ux_testing.py --all-personas="feature_name"

# Prepare production migration
python _Sandbox/Tools/feature_migration_tool.py --prepare-migration="feature_name"
```

#### Daily Sandbox Operations
```bash
# Daily sandbox initialization
python _Sandbox/Tools/sandbox_tdd_runner.py --daily-init

# Continuous TDD development
python _Sandbox/Tools/sandbox_tdd_runner.py --continuous-tdd

# Daily design system validation
python _Sandbox/Tools/design_system_validator.py --daily-check

# Daily cleanup and optimization
python _Sandbox/Tools/sandbox_cleanup_automation.py --daily-maintenance
```

## 13. Advanced Protocols & Automation

### Multi-Agent Coordination Protocols

1. **Task Decomposition**: Planner agent breaks complex tasks into agent-specific subtasks
2. **Agent Handoffs**: Standardized protocols for passing context between agents
3. **Resource Sharing**: Shared memory and state management across agents
4. **Conflict Resolution**: Handling competing agent requests and priorities
5. **Performance Optimization**: Dynamic load balancing across agents

### MCP Server Integration

The project integrates with Model Context Protocol (MCP) servers for enhanced AI capabilities:

#### Core MCP Integrations
- **Enhanced Reasoning**: Monte Carlo Tree Search and Beam Search algorithms
- **Structured Thinking**: Anthropic's structured reasoning tools
- **Context Management**: Memory optimization using Redis and vector storage
- **Code Intelligence**: Enhanced programming problem-solving capabilities

#### Configuration and Usage
```bash
# Configure MCP environment
source ~/.config/mcp/.env

# Test MCP server availability
curl -X POST http://localhost:3001/mcp/status

# Integrate with agent system
python -c "
from sources.agents.mcp_agent import test_mcp_integration
test_mcp_integration()
"
```

### Automation Tools

1. **Agent Training**: Automated routing accuracy improvement
2. **Performance Monitoring**: Continuous monitoring of agent and tool performance
3. **Security Scanning**: Automated privacy and security validation
4. **Documentation Generation**: Auto-generated API and agent documentation
5. **Deployment Pipeline**: Automated testing and deployment workflows

### Agent Operational Protocols

1. **Self-Improvement**: Agents learn from successful task completions
2. **Performance Monitoring**: Continuous tracking of agent effectiveness
3. **Error Recovery**: Automated recovery from common agent failures
4. **Knowledge Management**: Shared knowledge base across all agents
5. **Capability Discovery**: Dynamic discovery of new tools and capabilities

## 13. Reference Tables & Glossary

### File Extensions and Types

| Extension | Type | Purpose |
|-----------|------|---------|
| .py | Python Source | Core application logic and agents |
| .js/.jsx | React Source | Frontend React components |
| .swift | Swift Source | macOS native app code |
| .yml/.yaml | Configuration | Docker Compose and service configs |
| .ini | Configuration | Main application configuration |
| .md | Documentation | Markdown documentation files |
| .json | Data/Config | JSON configuration and data files |
| .sh | Shell Script | Build and automation scripts |
| .txt | Prompts | Agent prompt templates |
| .crx | Browser Extension | NoCaptcha browser extension |

### Core Directories

| Directory | Purpose |
|-----------|---------|
| sources/ | Core Python application code |
| sources/agents/ | Multi-agent system implementation |
| sources/tools/ | Tool execution and automation |
| frontend/ | React web interface |
| _macOS/ | macOS native app wrapper |
| llm_router/ | Agent routing ML models |
| prompts/ | Agent personality and behavior |
| searxng/ | Local search engine configuration |
| tests/ | Comprehensive test suites |
| docs/ | Project documentation |
| claude/ | Claude Code configuration |

### Key Commands Reference

| Command | Purpose |
|---------|---------|
| `docker-compose up -d` | Start all services |
| `python api.py` | Start backend API server |
| `python cli.py` | Command-line interface |
| `npm start` | React development server |
| `python test_suite.py` | Run comprehensive tests |
| `xcodebuild build` | Build macOS app |

### Agent Types and Capabilities

| Agent | Primary Capabilities | Tools Used |
|-------|---------------------|------------|
| CasualAgent | General conversation, Q&A | Web search, basic reasoning |
| CoderAgent | Programming, code execution | PyInterpreter, BashInterpreter, file tools |
| FileAgent | File operations, document processing | File finder, document tools |
| BrowserAgent | Web automation, form filling | Selenium, web tools |
| PlannerAgent | Task planning, coordination | All tools, multi-agent coordination |
| McpAgent | Protocol integration | MCP tools, external services |

### LLM Provider Types

| Provider Type | Examples | Use Cases |
|---------------|----------|-----------|
| Local | Ollama, LM-Studio | Privacy, offline operation, no costs |
| Cloud | OpenAI, Anthropic, Google | High quality, latest models |
| Specialized | DeepSeek | Code generation, technical tasks |
| Custom | Local LLM Server | Distributed processing |

### Glossary

- **Agent Routing**: ML-based system for selecting the most appropriate agent for tasks
- **Cascading Provider**: Fallback system between local and cloud LLM providers
- **Multi-Agent Coordination**: System for agents to collaborate on complex tasks
- **Tool Sandboxing**: Security isolation for code execution and system operations
- **Stealth Mode**: Anti-detection capabilities for web automation
- **Local-First**: Architecture prioritizing local processing over cloud services
- **Privacy-First**: Design principle ensuring user data remains on device
- **MCP**: Model Context Protocol for AI service integration
- **Agent Handoff**: Process of transferring tasks between different agents
- **Tool Execution**: Running specific capabilities through the tool system

---

**Version**: 1.0  
**Last Updated**: 2025-05-31  
**Maintainer**: AgenticSeek Development Team

For additional help or clarification on any of these protocols, refer to the specific documentation files in the project, the comprehensive `.cursorrules` file for UI/UX standards, the comprehensive test suites, or the agent-specific documentation.

### Quick Reference Commands

```bash
# Essential development workflow
docker-compose up -d                      # Start all services
python api.py                            # Start backend (host machine)
cd frontend/agentic-seek-front && npm start  # Start frontend (port 3000)

# Testing and validation
python test_suite.py --comprehensive      # Full test suite
python _macOS/tests/test_endpoints.py     # API testing
python -c "from sources.router import test_routing_accuracy; test_routing_accuracy()"

# Troubleshooting and diagnostics
docker-compose logs -f                    # View service logs
docker-compose ps                         # Check service status
python -c "from sources.llm_provider import test_providers; test_providers()"

# macOS app development
cd _macOS && xcodebuild build             # Build native app
open _macOS/AgenticSeek.app              # Run native app
```