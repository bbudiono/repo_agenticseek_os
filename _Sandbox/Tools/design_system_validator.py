#!/usr/bin/env python3
"""
Design System Compliance Validator
Enforces .cursorrules compliance across all sandbox development
Validates AgenticSeek design system standards
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DesignSystemValidator:
    """
    Validates .cursorrules compliance for AgenticSeek design system
    """
    
    def __init__(self, sandbox_root: str = "_Sandbox"):
        self.sandbox_root = Path(sandbox_root)
        self.design_system_rules = self._load_design_system_rules()
        
    def _load_design_system_rules(self) -> Dict:
        """Load design system rules from .cursorrules"""
        rules = {
            "colors": {
                "required_prefix": "DesignSystem.Colors.",
                "allowed_colors": [
                    "primary", "secondary", "agent", "success", "warning", 
                    "error", "code", "background", "surface", "onPrimary", "onSecondary"
                ],
                "forbidden_patterns": [
                    r"Color\(.*\)",
                    r"\.blue", r"\.red", r"\.green", r"\.yellow", r"\.purple",
                    r"#[0-9A-Fa-f]{6}", r"UIColor\(",
                    r"NSColor\(", r"SwiftUI\.Color\."
                ]
            },
            "typography": {
                "required_prefix": "DesignSystem.Typography.",
                "allowed_fonts": [
                    "headline", "title1", "title2", "title3", "body", "callout",
                    "caption", "code", "button", "agentLabel", "chatText"
                ],
                "forbidden_patterns": [
                    r"\.font\(\.system",
                    r"Font\.system",
                    r"Font\.custom",
                    r"UIFont\(",
                    r"NSFont\("
                ]
            },
            "spacing": {
                "required_prefix": "DesignSystem.Spacing.",
                "allowed_spacing": [
                    "xxxs", "xxs", "xs", "sm", "md", "lg", "xl", "xxl", "xxxl",
                    "chatPadding", "cardPadding", "buttonPadding", "agentPadding"
                ],
                "forbidden_patterns": [
                    r"\.padding\([0-9]+",
                    r"\.frame\(.*width:\s*[0-9]+",
                    r"\.frame\(.*height:\s*[0-9]+",
                    r"CGFloat\([0-9]+\)"
                ]
            },
            "components": {
                "required_modifiers": [
                    ".agentAvatarStyle",
                    ".messageBubbleStyle", 
                    ".agentSelectorStyle",
                    ".statusIndicatorStyle",
                    ".chatInputStyle",
                    ".primaryButtonStyle",
                    ".secondaryButtonStyle"
                ],
                "accessibility_requirements": [
                    ".accessibilityLabel",
                    ".accessibilityHint",
                    ".accessibilityValue"
                ]
            }
        }
        return rules
    
    def validate_feature(self, feature_name: str) -> Dict:
        """Validate design system compliance for a specific feature"""
        logger.info(f"Validating design system compliance for feature: {feature_name}")
        
        feature_path = self.sandbox_root / "Environment" / "TestDrivenFeatures" / f"{feature_name}_TDD"
        
        if not feature_path.exists():
            logger.error(f"Feature {feature_name} not found")
            return {"error": f"Feature {feature_name} not found"}
        
        validation_results = {
            "feature_name": feature_name,
            "overall_compliant": False,
            "color_compliance": self._validate_colors(feature_path),
            "typography_compliance": self._validate_typography(feature_path),
            "spacing_compliance": self._validate_spacing(feature_path),
            "component_compliance": self._validate_components(feature_path),
            "accessibility_compliance": self._validate_accessibility(feature_path)
        }
        
        # Determine overall compliance
        validation_results["overall_compliant"] = all([
            validation_results["color_compliance"]["compliant"],
            validation_results["typography_compliance"]["compliant"],
            validation_results["spacing_compliance"]["compliant"],
            validation_results["component_compliance"]["compliant"],
            validation_results["accessibility_compliance"]["compliant"]
        ])
        
        # Save validation report
        report_path = feature_path / "design_system_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results
    
    def _validate_colors(self, feature_path: Path) -> Dict:
        """Validate color system compliance"""
        logger.info("Validating color system compliance")
        
        violations = []
        compliant_usage = []
        
        # Scan all Swift files in feature
        for swift_file in feature_path.rglob("*.swift"):
            content = swift_file.read_text()
            
            # Check for forbidden color patterns
            for pattern in self.design_system_rules["colors"]["forbidden_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "file": str(swift_file.relative_to(feature_path)),
                        "line": line_num,
                        "violation": match.group(),
                        "pattern": pattern,
                        "message": "Use DesignSystem.Colors instead of hardcoded colors"
                    })
            
            # Check for compliant usage
            compliant_pattern = r"DesignSystem\.Colors\.\w+"
            matches = re.finditer(compliant_pattern, content)
            for match in matches:
                compliant_usage.append(match.group())
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "compliant_usage": list(set(compliant_usage)),
            "violation_count": len(violations)
        }
    
    def _validate_typography(self, feature_path: Path) -> Dict:
        """Validate typography system compliance"""
        logger.info("Validating typography system compliance")
        
        violations = []
        compliant_usage = []
        
        for swift_file in feature_path.rglob("*.swift"):
            content = swift_file.read_text()
            
            # Check for forbidden typography patterns
            for pattern in self.design_system_rules["typography"]["forbidden_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "file": str(swift_file.relative_to(feature_path)),
                        "line": line_num,
                        "violation": match.group(),
                        "pattern": pattern,
                        "message": "Use DesignSystem.Typography instead of system fonts"
                    })
            
            # Check for compliant usage
            compliant_pattern = r"DesignSystem\.Typography\.\w+"
            matches = re.finditer(compliant_pattern, content)
            for match in matches:
                compliant_usage.append(match.group())
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "compliant_usage": list(set(compliant_usage)),
            "violation_count": len(violations)
        }
    
    def _validate_spacing(self, feature_path: Path) -> Dict:
        """Validate spacing system compliance"""
        logger.info("Validating spacing system compliance")
        
        violations = []
        compliant_usage = []
        
        for swift_file in feature_path.rglob("*.swift"):
            content = swift_file.read_text()
            
            # Check for forbidden spacing patterns
            for pattern in self.design_system_rules["spacing"]["forbidden_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "file": str(swift_file.relative_to(feature_path)),
                        "line": line_num,
                        "violation": match.group(),
                        "pattern": pattern,
                        "message": "Use DesignSystem.Spacing instead of hardcoded values"
                    })
            
            # Check for compliant usage
            compliant_pattern = r"DesignSystem\.Spacing\.\w+"
            matches = re.finditer(compliant_pattern, content)
            for match in matches:
                compliant_usage.append(match.group())
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "compliant_usage": list(set(compliant_usage)),
            "violation_count": len(violations)
        }
    
    def _validate_components(self, feature_path: Path) -> Dict:
        """Validate component system compliance"""
        logger.info("Validating component system compliance")
        
        missing_modifiers = []
        found_modifiers = []
        
        for swift_file in feature_path.rglob("*.swift"):
            content = swift_file.read_text()
            
            # Check for required modifiers usage
            for modifier in self.design_system_rules["components"]["required_modifiers"]:
                if modifier in content:
                    found_modifiers.append(modifier)
        
        # Determine missing modifiers (context-dependent)
        expected_modifiers = self._determine_expected_modifiers(feature_path)
        missing_modifiers = [mod for mod in expected_modifiers if mod not in found_modifiers]
        
        return {
            "compliant": len(missing_modifiers) == 0,
            "found_modifiers": found_modifiers,
            "missing_modifiers": missing_modifiers,
            "modifier_coverage": len(found_modifiers) / max(len(expected_modifiers), 1)
        }
    
    def _validate_accessibility(self, feature_path: Path) -> Dict:
        """Validate accessibility compliance"""
        logger.info("Validating accessibility compliance")
        
        accessibility_usage = []
        missing_accessibility = []
        
        for swift_file in feature_path.rglob("*.swift"):
            content = swift_file.read_text()
            
            # Check for accessibility requirements
            for requirement in self.design_system_rules["components"]["accessibility_requirements"]:
                if requirement in content:
                    accessibility_usage.append(requirement)
        
        # Basic compliance check
        has_labels = ".accessibilityLabel" in str(accessibility_usage)
        has_hints = ".accessibilityHint" in str(accessibility_usage)
        
        compliance_score = len(accessibility_usage) / len(self.design_system_rules["components"]["accessibility_requirements"])
        
        return {
            "compliant": compliance_score >= 0.8,  # 80% compliance threshold
            "accessibility_usage": accessibility_usage,
            "compliance_score": compliance_score,
            "has_labels": has_labels,
            "has_hints": has_hints
        }
    
    def _determine_expected_modifiers(self, feature_path: Path) -> List[str]:
        """Determine which modifiers should be expected based on feature content"""
        expected = []
        
        # Scan feature files to determine context
        for swift_file in feature_path.rglob("*.swift"):
            content = swift_file.read_text().lower()
            
            if "button" in content:
                expected.extend([".primaryButtonStyle", ".secondaryButtonStyle"])
            if "message" in content or "chat" in content:
                expected.extend([".messageBubbleStyle", ".chatInputStyle"])
            if "agent" in content:
                expected.extend([".agentAvatarStyle", ".agentSelectorStyle"])
            if "status" in content:
                expected.append(".statusIndicatorStyle")
        
        return list(set(expected))
    
    def validate_comprehensive(self) -> Dict:
        """Run comprehensive validation across entire sandbox"""
        logger.info("Running comprehensive design system validation")
        
        features_path = self.sandbox_root / "Environment" / "TestDrivenFeatures"
        
        if not features_path.exists():
            return {"error": "No features found in sandbox"}
        
        results = {
            "timestamp": "2025-05-31",
            "overall_compliance": False,
            "features": {},
            "summary": {
                "total_features": 0,
                "compliant_features": 0,
                "total_violations": 0
            }
        }
        
        # Validate each feature
        for feature_dir in features_path.iterdir():
            if feature_dir.is_dir() and feature_dir.name.endswith("_TDD"):
                feature_name = feature_dir.name.replace("_TDD", "")
                feature_results = self.validate_feature(feature_name)
                results["features"][feature_name] = feature_results
                
                results["summary"]["total_features"] += 1
                if feature_results.get("overall_compliant", False):
                    results["summary"]["compliant_features"] += 1
        
        # Calculate overall compliance
        if results["summary"]["total_features"] > 0:
            compliance_rate = results["summary"]["compliant_features"] / results["summary"]["total_features"]
            results["overall_compliance"] = compliance_rate >= 1.0  # 100% compliance required
        
        # Save comprehensive report
        report_path = self.sandbox_root / "design_system_comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_compliance_report(self, feature_name: str) -> str:
        """Generate human-readable compliance report"""
        results = self.validate_feature(feature_name)
        
        report = f"""
# Design System Compliance Report
## Feature: {feature_name}

### Overall Compliance: {'✅ PASS' if results['overall_compliant'] else '❌ FAIL'}

### Color System Compliance
- **Status**: {'✅ COMPLIANT' if results['color_compliance']['compliant'] else '❌ VIOLATIONS FOUND'}
- **Violations**: {results['color_compliance']['violation_count']}
- **Compliant Usage**: {len(results['color_compliance']['compliant_usage'])} instances

### Typography System Compliance  
- **Status**: {'✅ COMPLIANT' if results['typography_compliance']['compliant'] else '❌ VIOLATIONS FOUND'}
- **Violations**: {results['typography_compliance']['violation_count']}
- **Compliant Usage**: {len(results['typography_compliance']['compliant_usage'])} instances

### Spacing System Compliance
- **Status**: {'✅ COMPLIANT' if results['spacing_compliance']['compliant'] else '❌ VIOLATIONS FOUND'}  
- **Violations**: {results['spacing_compliance']['violation_count']}
- **Compliant Usage**: {len(results['spacing_compliance']['compliant_usage'])} instances

### Component System Compliance
- **Status**: {'✅ COMPLIANT' if results['component_compliance']['compliant'] else '❌ IMPROVEMENTS NEEDED'}
- **Modifier Coverage**: {results['component_compliance']['modifier_coverage']:.1%}
- **Found Modifiers**: {len(results['component_compliance']['found_modifiers'])}

### Accessibility Compliance
- **Status**: {'✅ COMPLIANT' if results['accessibility_compliance']['compliant'] else '❌ IMPROVEMENTS NEEDED'}
- **Compliance Score**: {results['accessibility_compliance']['compliance_score']:.1%}
- **Has Labels**: {'✅' if results['accessibility_compliance']['has_labels'] else '❌'}
- **Has Hints**: {'✅' if results['accessibility_compliance']['has_hints'] else '❌'}

### Detailed Violations
"""
        
        # Add detailed violations
        for category in ['color_compliance', 'typography_compliance', 'spacing_compliance']:
            violations = results[category]['violations']
            if violations:
                report += f"\n#### {category.replace('_', ' ').title()} Violations:\n"
                for violation in violations:
                    report += f"- **{violation['file']}:{violation['line']}** - {violation['violation']} ({violation['message']})\n"
        
        return report

def main():
    """Main CLI interface for Design System Validator"""
    parser = argparse.ArgumentParser(description="Design System Compliance Validator")
    parser.add_argument("--validate", help="Validate specific feature")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive validation")
    parser.add_argument("--report", help="Generate compliance report for feature")
    parser.add_argument("--colors", action="store_true", help="Validate colors only")
    parser.add_argument("--typography", action="store_true", help="Validate typography only")
    parser.add_argument("--spacing", action="store_true", help="Validate spacing only")
    
    args = parser.parse_args()
    
    validator = DesignSystemValidator()
    
    if args.validate:
        result = validator.validate_feature(args.validate)
        print(json.dumps(result, indent=2))
    elif args.comprehensive:
        result = validator.validate_comprehensive()
        print(json.dumps(result, indent=2))
    elif args.report:
        report = validator.generate_compliance_report(args.report)
        print(report)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()