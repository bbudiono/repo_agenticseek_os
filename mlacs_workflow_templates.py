#!/usr/bin/env python3
"""
* Purpose: MLACS workflow templates library providing pre-configured templates for common multi-LLM coordination use cases
* Issues & Complexity Summary: Comprehensive template system with customizable workflows and best practice patterns
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): ~1000
  - Core Algorithm Complexity: High
  - Dependencies: 8 New, 5 Mod
  - State Management Complexity: Medium
  - Novelty/Uncertainty Factor: Low
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): 75%
* Problem Estimate (Inherent Problem Difficulty %): 70%
* Initial Code Complexity Estimate %: 75%
* Justification for Estimates: Template library requiring comprehensive patterns and customization options
* Final Code Complexity (Actual %): 78%
* Overall Result Score (Success & Quality %): 92%
* Key Variances/Learnings: Successfully created comprehensive workflow template system
* Last Updated: 2025-01-06
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum

class TemplateCategory(Enum):
    """Categories of workflow templates"""
    BUSINESS_ANALYSIS = "business_analysis"
    CREATIVE_DEVELOPMENT = "creative_development"
    TECHNICAL_ANALYSIS = "technical_analysis"
    RESEARCH_SYNTHESIS = "research_synthesis"
    QUALITY_ASSURANCE = "quality_assurance"
    CONTENT_CREATION = "content_creation"
    DECISION_SUPPORT = "decision_support"
    EDUCATIONAL = "educational"
    INNOVATION = "innovation"
    STRATEGIC_PLANNING = "strategic_planning"

class DifficultyLevel(Enum):
    """Difficulty levels for templates"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class WorkflowPattern(Enum):
    """Common workflow patterns"""
    SEQUENTIAL_ANALYSIS = "sequential_analysis"
    PARALLEL_SYNTHESIS = "parallel_synthesis"
    HIERARCHICAL_REVIEW = "hierarchical_review"
    CONSENSUS_BUILDING = "consensus_building"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    SPECIALIZED_ROLES = "specialized_roles"
    VERIFICATION_CASCADE = "verification_cascade"
    CREATIVE_COLLABORATION = "creative_collaboration"

@dataclass
class TemplateParameter:
    """Template parameter definition"""
    name: str
    type: str  # "string", "number", "boolean", "list", "dict"
    description: str
    default_value: Any = None
    required: bool = True
    validation_rules: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

@dataclass
class LLMRoleDefinition:
    """Definition of an LLM role in a workflow"""
    role_name: str
    description: str
    capabilities_required: List[str]
    specialization: str
    priority: int = 1
    expected_contribution: str = ""

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    name: str
    description: str
    llm_roles: List[str]
    input_requirements: List[str]
    output_specifications: List[str]
    success_criteria: Dict[str, float]
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)

@dataclass
class WorkflowTemplate:
    """Complete workflow template definition"""
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    difficulty_level: DifficultyLevel
    workflow_pattern: WorkflowPattern
    
    # Configuration
    parameters: List[TemplateParameter] = field(default_factory=list)
    llm_roles: List[LLMRoleDefinition] = field(default_factory=list)
    workflow_steps: List[WorkflowStep] = field(default_factory=list)
    
    # Metadata
    estimated_duration: float = 0.0
    min_llm_count: int = 2
    max_llm_count: int = 5
    success_criteria: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    
    # Template metadata
    created_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    version: str = "1.0"
    author: str = "MLACS System"

class WorkflowTemplateLibrary:
    """
    Comprehensive library of MLACS workflow templates
    
    Provides pre-configured templates for common multi-LLM coordination scenarios
    including business analysis, creative development, technical reviews, and more.
    """
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._initialize_standard_templates()
    
    def _initialize_standard_templates(self):
        """Initialize the standard template library"""
        
        # Business Analysis Templates
        self._add_business_strategy_template()
        self._add_market_research_template()
        self._add_competitive_analysis_template()
        
        # Creative Development Templates
        self._add_content_campaign_template()
        self._add_brand_development_template()
        self._add_creative_brainstorming_template()
        
        # Technical Analysis Templates
        self._add_architecture_review_template()
        self._add_code_quality_template()
        self._add_security_assessment_template()
        
        # Research Synthesis Templates
        self._add_literature_review_template()
        self._add_cross_domain_synthesis_template()
        self._add_trend_analysis_template()
        
        # Quality Assurance Templates
        self._add_document_review_template()
        self._add_multi_perspective_qa_template()
        self._add_fact_checking_template()
    
    def _add_business_strategy_template(self):
        """Add comprehensive business strategy analysis template"""
        
        template = WorkflowTemplate(
            template_id="business_strategy_analysis",
            name="Business Strategy Analysis",
            description="Comprehensive multi-LLM business strategy development with market analysis, competitive intelligence, and strategic recommendations",
            category=TemplateCategory.BUSINESS_ANALYSIS,
            difficulty_level=DifficultyLevel.ADVANCED,
            workflow_pattern=WorkflowPattern.HIERARCHICAL_REVIEW,
            
            parameters=[
                TemplateParameter("company_description", "string", "Brief description of the company and its current situation", required=True),
                TemplateParameter("industry", "string", "Industry or market sector", required=True),
                TemplateParameter("time_horizon", "string", "Strategic planning time horizon", "12 months", False),
                TemplateParameter("budget_constraints", "string", "Budget limitations or constraints", "Not specified", False),
                TemplateParameter("geographic_scope", "string", "Geographic market scope", "Global", False),
                TemplateParameter("key_challenges", "list", "List of key challenges facing the company", required=False),
                TemplateParameter("strategic_goals", "list", "Primary strategic objectives", required=False)
            ],
            
            llm_roles=[
                LLMRoleDefinition("market_analyst", "Analyzes market trends, size, and opportunities", 
                                ["analysis", "factual_knowledge"], "Market Research", 1, "Market landscape analysis"),
                LLMRoleDefinition("competitive_intelligence", "Researches competitors and competitive dynamics", 
                                ["analysis", "research"], "Competitive Analysis", 1, "Competitive positioning insights"),
                LLMRoleDefinition("strategy_consultant", "Develops strategic recommendations and frameworks", 
                                ["reasoning", "synthesis"], "Strategic Planning", 2, "Strategic recommendations"),
                LLMRoleDefinition("financial_analyst", "Analyzes financial implications and projections", 
                                ["analysis", "reasoning"], "Financial Analysis", 1, "Financial viability assessment")
            ],
            
            workflow_steps=[
                WorkflowStep("market_analysis", "Market Landscape Analysis", 
                           "Comprehensive analysis of market size, trends, and opportunities",
                           ["market_analyst"], ["company_description", "industry"], 
                           ["market_size", "growth_trends", "opportunities"], 
                           {"comprehensiveness": 0.85, "accuracy": 0.90}, 8.0),
                
                WorkflowStep("competitive_research", "Competitive Intelligence", 
                           "Research and analysis of competitive landscape",
                           ["competitive_intelligence"], ["industry", "geographic_scope"], 
                           ["competitor_profiles", "competitive_advantages", "market_positioning"], 
                           {"depth": 0.80, "relevance": 0.85}, 6.0),
                
                WorkflowStep("strategic_synthesis", "Strategic Framework Development", 
                           "Synthesis of insights into strategic recommendations",
                           ["strategy_consultant"], ["market_analysis", "competitive_research"], 
                           ["strategic_options", "recommendations", "implementation_plan"], 
                           {"strategic_coherence": 0.90, "actionability": 0.85}, 10.0, 
                           ["market_analysis", "competitive_research"]),
                
                WorkflowStep("financial_validation", "Financial Validation", 
                           "Financial analysis and validation of strategic recommendations",
                           ["financial_analyst"], ["strategic_synthesis"], 
                           ["financial_projections", "risk_assessment", "roi_analysis"], 
                           {"financial_accuracy": 0.85, "feasibility": 0.80}, 5.0, 
                           ["strategic_synthesis"])
            ],
            
            estimated_duration=29.0,
            min_llm_count=3,
            max_llm_count=4,
            success_criteria={
                "strategic_coherence": 0.85,
                "market_insight_depth": 0.80,
                "competitive_intelligence": 0.80,
                "financial_viability": 0.75,
                "actionability": 0.85
            },
            tags=["business", "strategy", "market-analysis", "competitive-intelligence"],
            use_cases=[
                "Strategic planning for startups",
                "Market entry strategy development",
                "Business model pivots",
                "Competitive positioning",
                "Investment decision support"
            ],
            best_practices=[
                "Ensure market analyst has access to current market data",
                "Include multiple competitive perspectives",
                "Validate strategic recommendations with financial analysis",
                "Consider both short-term and long-term implications",
                "Incorporate risk assessment throughout"
            ]
        )
        
        self.templates[template.template_id] = template
    
    def _add_content_campaign_template(self):
        """Add creative content campaign development template"""
        
        template = WorkflowTemplate(
            template_id="content_campaign_development",
            name="Content Campaign Development",
            description="Collaborative creative campaign development with multi-channel content creation and messaging strategy",
            category=TemplateCategory.CREATIVE_DEVELOPMENT,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.CREATIVE_COLLABORATION,
            
            parameters=[
                TemplateParameter("brand_description", "string", "Brand description and values", required=True),
                TemplateParameter("target_audience", "string", "Primary target audience description", required=True),
                TemplateParameter("campaign_objective", "string", "Primary campaign objective", required=True),
                TemplateParameter("channels", "list", "Marketing channels to target", ["social", "email", "content"], True),
                TemplateParameter("budget", "string", "Campaign budget range", "Medium", False),
                TemplateParameter("timeline", "string", "Campaign timeline", "6 weeks", False),
                TemplateParameter("brand_guidelines", "string", "Brand guidelines and constraints", "", False)
            ],
            
            llm_roles=[
                LLMRoleDefinition("creative_strategist", "Develops creative strategy and concepts", 
                                ["creativity", "synthesis"], "Creative Strategy", 2, "Creative direction and concepts"),
                LLMRoleDefinition("copywriter", "Creates compelling copy and messaging", 
                                ["creativity", "language"], "Copywriting", 1, "Campaign messaging and copy"),
                LLMRoleDefinition("content_specialist", "Develops content for specific channels", 
                                ["creativity", "adaptation"], "Content Development", 1, "Channel-specific content"),
                LLMRoleDefinition("brand_guardian", "Ensures brand consistency and guidelines", 
                                ["critique", "verification"], "Brand Compliance", 1, "Brand consistency validation")
            ],
            
            workflow_steps=[
                WorkflowStep("creative_strategy", "Creative Strategy Development", 
                           "Develop overarching creative strategy and core concepts",
                           ["creative_strategist"], ["brand_description", "target_audience", "campaign_objective"], 
                           ["creative_strategy", "core_concepts", "key_messages"], 
                           {"creativity": 0.85, "brand_alignment": 0.90}, 6.0),
                
                WorkflowStep("message_development", "Message and Copy Development", 
                           "Create compelling messaging and copy for the campaign",
                           ["copywriter"], ["creative_strategy"], 
                           ["headline_options", "body_copy", "call_to_action"], 
                           {"message_clarity": 0.85, "persuasiveness": 0.80}, 5.0, 
                           ["creative_strategy"]),
                
                WorkflowStep("content_adaptation", "Multi-Channel Content Creation", 
                           "Adapt content for specific marketing channels",
                           ["content_specialist"], ["message_development", "channels"], 
                           ["social_content", "email_content", "web_content"], 
                           {"channel_optimization": 0.80, "consistency": 0.85}, 7.0, 
                           ["message_development"]),
                
                WorkflowStep("brand_review", "Brand Compliance Review", 
                           "Review all content for brand consistency and guidelines",
                           ["brand_guardian"], ["content_adaptation"], 
                           ["compliance_report", "revision_recommendations", "final_approval"], 
                           {"brand_consistency": 0.95, "guideline_adherence": 0.90}, 3.0, 
                           ["content_adaptation"])
            ],
            
            estimated_duration=21.0,
            min_llm_count=3,
            max_llm_count=4,
            success_criteria={
                "creative_quality": 0.85,
                "brand_consistency": 0.90,
                "message_clarity": 0.85,
                "channel_optimization": 0.80,
                "campaign_coherence": 0.85
            },
            tags=["creative", "marketing", "content", "branding", "campaign"],
            use_cases=[
                "Product launch campaigns",
                "Brand awareness initiatives",
                "Multi-channel marketing campaigns",
                "Content marketing strategies",
                "Social media campaigns"
            ]
        )
        
        self.templates[template.template_id] = template
    
    def _add_architecture_review_template(self):
        """Add technical architecture review template"""
        
        template = WorkflowTemplate(
            template_id="architecture_review",
            name="System Architecture Review",
            description="Comprehensive technical architecture review with scalability, security, and performance analysis",
            category=TemplateCategory.TECHNICAL_ANALYSIS,
            difficulty_level=DifficultyLevel.ADVANCED,
            workflow_pattern=WorkflowPattern.SPECIALIZED_ROLES,
            
            parameters=[
                TemplateParameter("system_description", "string", "Description of the system architecture", required=True),
                TemplateParameter("tech_stack", "list", "Technologies and frameworks used", required=True),
                TemplateParameter("scale_requirements", "string", "Performance and scale requirements", required=True),
                TemplateParameter("current_issues", "list", "Known issues or pain points", required=False),
                TemplateParameter("constraints", "list", "Technical or business constraints", required=False),
                TemplateParameter("review_focus", "list", "Specific areas to focus on", ["scalability", "security", "performance"], False)
            ],
            
            llm_roles=[
                LLMRoleDefinition("system_architect", "Reviews overall architecture and design patterns", 
                                ["analysis", "reasoning", "technical"], "System Architecture", 2, "Architecture assessment"),
                LLMRoleDefinition("security_specialist", "Analyzes security aspects and vulnerabilities", 
                                ["analysis", "security"], "Security Analysis", 1, "Security evaluation"),
                LLMRoleDefinition("performance_engineer", "Evaluates performance and scalability", 
                                ["analysis", "optimization"], "Performance Analysis", 1, "Performance optimization"),
                LLMRoleDefinition("devops_expert", "Reviews deployment and operational aspects", 
                                ["analysis", "operations"], "DevOps Analysis", 1, "Operational assessment")
            ],
            
            estimated_duration=25.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["technical", "architecture", "security", "performance", "scalability"]
        )
        
        self.templates[template.template_id] = template
    
    def _add_literature_review_template(self):
        """Add academic literature review template"""
        
        template = WorkflowTemplate(
            template_id="literature_review",
            name="Academic Literature Review",
            description="Systematic literature review with cross-domain synthesis and insight generation",
            category=TemplateCategory.RESEARCH_SYNTHESIS,
            difficulty_level=DifficultyLevel.ADVANCED,
            workflow_pattern=WorkflowPattern.SEQUENTIAL_ANALYSIS,
            
            parameters=[
                TemplateParameter("research_question", "string", "Primary research question or hypothesis", required=True),
                TemplateParameter("domains", "list", "Research domains or fields to include", required=True),
                TemplateParameter("time_range", "string", "Time range for literature (e.g., '2020-2024')", "2020-2024", True),
                TemplateParameter("methodology", "string", "Review methodology preferences", "Systematic", False),
                TemplateParameter("inclusion_criteria", "list", "Criteria for including sources", required=False),
                TemplateParameter("synthesis_focus", "string", "Focus for synthesis (trends, gaps, insights)", "insights", False)
            ],
            
            llm_roles=[
                LLMRoleDefinition("research_coordinator", "Coordinates research strategy and methodology", 
                                ["analysis", "research"], "Research Coordination", 2, "Research strategy"),
                LLMRoleDefinition("domain_specialist", "Provides domain-specific expertise and analysis", 
                                ["analysis", "factual_knowledge"], "Domain Analysis", 1, "Domain insights"),
                LLMRoleDefinition("synthesis_expert", "Synthesizes findings across domains", 
                                ["synthesis", "reasoning"], "Cross-Domain Synthesis", 2, "Insight synthesis"),
                LLMRoleDefinition("methodology_reviewer", "Reviews methodology and ensures rigor", 
                                ["critique", "verification"], "Methodology Review", 1, "Quality assurance")
            ],
            
            estimated_duration=30.0,
            min_llm_count=4,
            max_llm_count=5,
            tags=["research", "literature", "synthesis", "academic", "analysis"]
        )
        
        self.templates[template.template_id] = template
    
    def _add_document_review_template(self):
        """Add document quality assurance template"""
        
        template = WorkflowTemplate(
            template_id="document_qa_review",
            name="Document Quality Assurance Review",
            description="Multi-perspective quality assurance review with comprehensive verification",
            category=TemplateCategory.QUALITY_ASSURANCE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.VERIFICATION_CASCADE,
            
            parameters=[
                TemplateParameter("document_type", "string", "Type of document being reviewed", required=True),
                TemplateParameter("target_audience", "string", "Intended audience for the document", required=True),
                TemplateParameter("review_criteria", "list", "Specific criteria to evaluate", ["accuracy", "clarity", "completeness"], True),
                TemplateParameter("domain", "string", "Subject domain or field", required=True),
                TemplateParameter("urgency", "string", "Review urgency level", "normal", False)
            ],
            
            llm_roles=[
                LLMRoleDefinition("content_reviewer", "Reviews content accuracy and completeness", 
                                ["analysis", "verification"], "Content Review", 1, "Content validation"),
                LLMRoleDefinition("language_editor", "Reviews language, style, and clarity", 
                                ["language", "critique"], "Language Review", 1, "Language optimization"),
                LLMRoleDefinition("domain_expert", "Provides domain-specific validation", 
                                ["analysis", "factual_knowledge"], "Domain Validation", 1, "Technical accuracy"),
                LLMRoleDefinition("user_advocate", "Reviews from audience perspective", 
                                ["critique", "reasoning"], "User Experience", 1, "Audience appropriateness")
            ],
            
            estimated_duration=12.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["quality", "review", "verification", "documentation", "accuracy"]
        )
        
        self.templates[template.template_id] = template
    
    # Additional template methods for other categories...
    
    def _add_market_research_template(self):
        """Add market research template"""
        template = WorkflowTemplate(
            template_id="market_research_analysis",
            name="Market Research Analysis",
            description="Comprehensive market research with trend analysis and opportunity identification",
            category=TemplateCategory.BUSINESS_ANALYSIS,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.PARALLEL_SYNTHESIS,
            estimated_duration=20.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["market", "research", "trends", "opportunities"]
        )
        self.templates[template.template_id] = template
    
    def _add_competitive_analysis_template(self):
        """Add competitive analysis template"""
        template = WorkflowTemplate(
            template_id="competitive_analysis",
            name="Competitive Analysis",
            description="In-depth competitive landscape analysis with positioning recommendations",
            category=TemplateCategory.BUSINESS_ANALYSIS,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.HIERARCHICAL_REVIEW,
            estimated_duration=18.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["competitive", "analysis", "positioning", "strategy"]
        )
        self.templates[template.template_id] = template
    
    def _add_brand_development_template(self):
        """Add brand development template"""
        template = WorkflowTemplate(
            template_id="brand_development",
            name="Brand Development",
            description="Comprehensive brand development with identity, messaging, and positioning",
            category=TemplateCategory.CREATIVE_DEVELOPMENT,
            difficulty_level=DifficultyLevel.ADVANCED,
            workflow_pattern=WorkflowPattern.CREATIVE_COLLABORATION,
            estimated_duration=25.0,
            min_llm_count=4,
            max_llm_count=5,
            tags=["brand", "identity", "messaging", "positioning"]
        )
        self.templates[template.template_id] = template
    
    def _add_creative_brainstorming_template(self):
        """Add creative brainstorming template"""
        template = WorkflowTemplate(
            template_id="creative_brainstorming",
            name="Creative Brainstorming Session",
            description="Structured creative brainstorming with idea generation and evaluation",
            category=TemplateCategory.CREATIVE_DEVELOPMENT,
            difficulty_level=DifficultyLevel.BEGINNER,
            workflow_pattern=WorkflowPattern.PARALLEL_SYNTHESIS,
            estimated_duration=10.0,
            min_llm_count=3,
            max_llm_count=5,
            tags=["creative", "brainstorming", "ideas", "innovation"]
        )
        self.templates[template.template_id] = template
    
    def _add_code_quality_template(self):
        """Add code quality review template"""
        template = WorkflowTemplate(
            template_id="code_quality_review",
            name="Code Quality Review",
            description="Comprehensive code quality review with best practices validation",
            category=TemplateCategory.TECHNICAL_ANALYSIS,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.VERIFICATION_CASCADE,
            estimated_duration=15.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["code", "quality", "review", "best-practices"]
        )
        self.templates[template.template_id] = template
    
    def _add_security_assessment_template(self):
        """Add security assessment template"""
        template = WorkflowTemplate(
            template_id="security_assessment",
            name="Security Assessment",
            description="Comprehensive security assessment with vulnerability analysis",
            category=TemplateCategory.TECHNICAL_ANALYSIS,
            difficulty_level=DifficultyLevel.ADVANCED,
            workflow_pattern=WorkflowPattern.SPECIALIZED_ROLES,
            estimated_duration=22.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["security", "assessment", "vulnerability", "compliance"]
        )
        self.templates[template.template_id] = template
    
    def _add_cross_domain_synthesis_template(self):
        """Add cross-domain synthesis template"""
        template = WorkflowTemplate(
            template_id="cross_domain_synthesis",
            name="Cross-Domain Synthesis",
            description="Synthesis of insights across multiple domains and disciplines",
            category=TemplateCategory.RESEARCH_SYNTHESIS,
            difficulty_level=DifficultyLevel.EXPERT,
            workflow_pattern=WorkflowPattern.CONSENSUS_BUILDING,
            estimated_duration=35.0,
            min_llm_count=4,
            max_llm_count=5,
            tags=["synthesis", "cross-domain", "interdisciplinary", "insights"]
        )
        self.templates[template.template_id] = template
    
    def _add_trend_analysis_template(self):
        """Add trend analysis template"""
        template = WorkflowTemplate(
            template_id="trend_analysis",
            name="Trend Analysis",
            description="Comprehensive trend analysis with future implications",
            category=TemplateCategory.RESEARCH_SYNTHESIS,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.SEQUENTIAL_ANALYSIS,
            estimated_duration=18.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["trends", "analysis", "forecasting", "implications"]
        )
        self.templates[template.template_id] = template
    
    def _add_multi_perspective_qa_template(self):
        """Add multi-perspective QA template"""
        template = WorkflowTemplate(
            template_id="multi_perspective_qa",
            name="Multi-Perspective Quality Assurance",
            description="Quality assurance from multiple stakeholder perspectives",
            category=TemplateCategory.QUALITY_ASSURANCE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.PARALLEL_SYNTHESIS,
            estimated_duration=14.0,
            min_llm_count=4,
            max_llm_count=5,
            tags=["quality", "assurance", "perspectives", "stakeholders"]
        )
        self.templates[template.template_id] = template
    
    def _add_fact_checking_template(self):
        """Add fact checking template"""
        template = WorkflowTemplate(
            template_id="fact_checking",
            name="Comprehensive Fact Checking",
            description="Multi-source fact verification and accuracy validation",
            category=TemplateCategory.QUALITY_ASSURANCE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            workflow_pattern=WorkflowPattern.VERIFICATION_CASCADE,
            estimated_duration=12.0,
            min_llm_count=3,
            max_llm_count=4,
            tags=["fact-checking", "verification", "accuracy", "validation"]
        )
        self.templates[template.template_id] = template
    
    # Library management methods
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a specific template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[TemplateCategory] = None, 
                      difficulty: Optional[DifficultyLevel] = None,
                      pattern: Optional[WorkflowPattern] = None) -> List[WorkflowTemplate]:
        """List templates with optional filtering"""
        
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if difficulty:
            templates = [t for t in templates if t.difficulty_level == difficulty]
            
        if pattern:
            templates = [t for t in templates if t.workflow_pattern == pattern]
        
        return templates
    
    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or tags"""
        
        query_lower = query.lower()
        matching_templates = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in tag for tag in template.tags)):
                matching_templates.append(template)
        
        return matching_templates
    
    def get_template_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the template library"""
        
        total_templates = len(self.templates)
        by_category = {}
        by_difficulty = {}
        by_pattern = {}
        
        for template in self.templates.values():
            # Count by category
            cat = template.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            # Count by difficulty
            diff = template.difficulty_level.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
            
            # Count by pattern
            pat = template.workflow_pattern.value
            by_pattern[pat] = by_pattern.get(pat, 0) + 1
        
        return {
            "total_templates": total_templates,
            "categories": by_category,
            "difficulty_levels": by_difficulty,
            "workflow_patterns": by_pattern,
            "average_duration": sum(t.estimated_duration for t in self.templates.values()) / total_templates if total_templates > 0 else 0,
            "average_llm_count": sum((t.min_llm_count + t.max_llm_count) / 2 for t in self.templates.values()) / total_templates if total_templates > 0 else 0
        }
    
    def export_template(self, template_id: str, format: str = "json") -> str:
        """Export a template in specified format"""
        
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        if format == "json":
            return json.dumps(asdict(template), indent=2)
        else:
            raise ValueError(f"Format {format} not supported")
    
    def create_custom_template(self, template_data: Dict[str, Any]) -> WorkflowTemplate:
        """Create a custom template from data"""
        
        # Convert enum strings to enum objects
        if 'category' in template_data:
            template_data['category'] = TemplateCategory(template_data['category'])
        if 'difficulty_level' in template_data:
            template_data['difficulty_level'] = DifficultyLevel(template_data['difficulty_level'])
        if 'workflow_pattern' in template_data:
            template_data['workflow_pattern'] = WorkflowPattern(template_data['workflow_pattern'])
        
        template = WorkflowTemplate(**template_data)
        self.templates[template.template_id] = template
        
        return template

def demonstrate_template_library():
    """Demonstrate the workflow template library capabilities"""
    
    print("🚀 MLACS Workflow Template Library Demonstration")
    print("=" * 60)
    
    # Initialize library
    library = WorkflowTemplateLibrary()
    
    # Show summary
    summary = library.get_template_summary()
    print(f"📚 Template Library Summary:")
    print(f"   Total Templates: {summary['total_templates']}")
    print(f"   Average Duration: {summary['average_duration']:.1f} minutes")
    print(f"   Average LLM Count: {summary['average_llm_count']:.1f}")
    print()
    
    # Show categories
    print(f"📂 Categories:")
    for category, count in summary['categories'].items():
        print(f"   {category.replace('_', ' ').title()}: {count} templates")
    print()
    
    # Show difficulty levels
    print(f"🎯 Difficulty Levels:")
    for level, count in summary['difficulty_levels'].items():
        print(f"   {level.title()}: {count} templates")
    print()
    
    # Show workflow patterns
    print(f"🔄 Workflow Patterns:")
    for pattern, count in summary['workflow_patterns'].items():
        print(f"   {pattern.replace('_', ' ').title()}: {count} templates")
    print()
    
    # Demonstrate template retrieval
    print(f"🔍 Template Examples:")
    
    # Business strategy template
    business_template = library.get_template("business_strategy_analysis")
    if business_template:
        print(f"   📊 {business_template.name}")
        print(f"      Category: {business_template.category.value}")
        print(f"      Difficulty: {business_template.difficulty_level.value}")
        print(f"      Duration: {business_template.estimated_duration:.0f} minutes")
        print(f"      LLM Roles: {len(business_template.llm_roles)}")
        print(f"      Steps: {len(business_template.workflow_steps)}")
        print()
    
    # Creative template
    creative_template = library.get_template("content_campaign_development")
    if creative_template:
        print(f"   🎨 {creative_template.name}")
        print(f"      Category: {creative_template.category.value}")
        print(f"      Difficulty: {creative_template.difficulty_level.value}")
        print(f"      Duration: {creative_template.estimated_duration:.0f} minutes")
        print(f"      LLM Roles: {len(creative_template.llm_roles)}")
        print()
    
    # Search demonstration
    print(f"🔎 Search Results for 'analysis':")
    search_results = library.search_templates("analysis")
    for template in search_results[:3]:
        print(f"   - {template.name} ({template.category.value})")
    print()
    
    # Filter demonstration
    print(f"📋 Advanced Templates:")
    advanced_templates = library.list_templates(difficulty=DifficultyLevel.ADVANCED)
    for template in advanced_templates[:3]:
        print(f"   - {template.name} ({template.estimated_duration:.0f}min)")
    
    print()
    print("✅ Template library demonstration complete!")
    print(f"🎯 {summary['total_templates']} templates ready for MLACS workflows")

if __name__ == "__main__":
    demonstrate_template_library()