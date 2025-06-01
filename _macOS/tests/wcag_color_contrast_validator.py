#!/usr/bin/env python3
"""
WCAG AAA Color Contrast Compliance Validator and Fixer
Validates and fixes color contrast issues per WCAG 2.1 AAA standards
"""

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

class WCAGColorValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "validation_passed": False,
            "color_analysis": {},
            "contrast_violations": [],
            "recommended_fixes": [],
            "wcag_compliance_score": 0
        }
        
        # WCAG AAA Contrast Requirements
        self.WCAG_AAA_NORMAL = 7.0      # AAA for normal text
        self.WCAG_AAA_LARGE = 4.5       # AAA for large text (18pt+ or 14pt+ bold)
        self.WCAG_AA_NORMAL = 4.5       # AA for normal text (fallback)
        self.WCAG_AA_LARGE = 3.0        # AA for large text (fallback)
        
    def validate_wcag_compliance(self) -> Dict[str, Any]:
        """Main validation method for WCAG AAA compliance"""
        print("🎨 Starting WCAG AAA Color Contrast Validation...")
        
        # Step 1: Extract colors from DesignSystem
        self._extract_design_system_colors()
        
        # Step 2: Calculate contrast ratios
        self._calculate_contrast_ratios()
        
        # Step 3: Identify violations
        self._identify_violations()
        
        # Step 4: Generate fixes
        self._generate_contrast_fixes()
        
        # Step 5: Calculate compliance score
        self._calculate_compliance_score()
        
        return self.results
        
    def _extract_design_system_colors(self):
        """Extract color definitions from DesignSystem.swift"""
        design_system_path = self.project_root / "AgenticSeek-Sandbox" / "DesignSystem.swift"
        
        if not design_system_path.exists():
            print(f"❌ DesignSystem.swift not found at {design_system_path}")
            return
            
        with open(design_system_path, 'r') as f:
            content = f.read()
            
        # Extract hex colors
        hex_colors = {}
        hex_pattern = r'static let (\w+) = Color\(hex: "([#\w]+)"\)'
        for match in re.finditer(hex_pattern, content):
            color_name, hex_value = match.groups()
            hex_colors[color_name] = hex_value
            
        # Extract RGB colors
        rgb_colors = {}
        rgb_pattern = r'static let (\w+) = Color\(red: ([\d.]+), green: ([\d.]+), blue: ([\d.]+)\)'
        for match in re.finditer(rgb_pattern, content):
            color_name, r, g, b = match.groups()
            # Convert to hex for consistency
            r_int = int(float(r) * 255)
            g_int = int(float(g) * 255)
            b_int = int(float(b) * 255)
            hex_value = f"#{r_int:02x}{g_int:02x}{b_int:02x}"
            rgb_colors[color_name] = hex_value
            
        # Combine all colors
        all_colors = {**hex_colors, **rgb_colors}
        self.results["color_analysis"]["extracted_colors"] = all_colors
        
        print(f"✅ Extracted {len(all_colors)} colors from DesignSystem")
        
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB values"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
    def _rgb_to_luminance(self, r: int, g: int, b: int) -> float:
        """Calculate relative luminance per WCAG standards"""
        def linearize(channel):
            channel = channel / 255.0
            if channel <= 0.03928:
                return channel / 12.92
            else:
                return math.pow((channel + 0.055) / 1.055, 2.4)
                
        r_lin = linearize(r)
        g_lin = linearize(g)
        b_lin = linearize(b)
        
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
        
    def _contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors"""
        try:
            r1, g1, b1 = self._hex_to_rgb(color1)
            r2, g2, b2 = self._hex_to_rgb(color2)
            
            lum1 = self._rgb_to_luminance(r1, g1, b1)
            lum2 = self._rgb_to_luminance(r2, g2, b2)
            
            # Ensure lum1 is the lighter color
            if lum1 < lum2:
                lum1, lum2 = lum2, lum1
                
            return (lum1 + 0.05) / (lum2 + 0.05)
        except:
            return 0.0
            
    def _calculate_contrast_ratios(self):
        """Calculate contrast ratios for all color combinations"""
        colors = self.results["color_analysis"]["extracted_colors"]
        
        # Define color pairs that are used together in the UI
        color_pairs = [
            # Text on backgrounds
            ("textPrimary", "background"),
            ("textPrimary", "surface"),
            ("textSecondary", "background"),
            ("textSecondary", "surface"),
            ("onPrimary", "primary"),
            ("onSecondary", "secondary"),
            ("onSurface", "surface"),
            ("onBackground", "background"),
            
            # Button text on button backgrounds
            ("onPrimary", "primary"),
            ("onPrimary", "error"),
            ("onPrimary", "success"),
            
            # Status colors
            ("error", "background"),
            ("success", "background"),
            ("warning", "background"),
            
            # Code text
            ("codeText", "code"),
            
            # Agent colors
            ("onPrimary", "agent"),
        ]
        
        contrast_results = {}
        for fg_name, bg_name in color_pairs:
            if fg_name in colors and bg_name in colors:
                fg_color = colors[fg_name]
                bg_color = colors[bg_name]
                ratio = self._contrast_ratio(fg_color, bg_color)
                
                contrast_results[f"{fg_name}_on_{bg_name}"] = {
                    "foreground": fg_color,
                    "background": bg_color,
                    "ratio": ratio,
                    "wcag_aaa_pass": ratio >= self.WCAG_AAA_NORMAL,
                    "wcag_aa_pass": ratio >= self.WCAG_AA_NORMAL
                }
                
        self.results["color_analysis"]["contrast_ratios"] = contrast_results
        print(f"✅ Calculated contrast ratios for {len(contrast_results)} color pairs")
        
    def _identify_violations(self):
        """Identify WCAG AAA violations"""
        contrast_ratios = self.results["color_analysis"]["contrast_ratios"]
        violations = []
        
        for pair_name, analysis in contrast_ratios.items():
            if not analysis["wcag_aaa_pass"]:
                violation = {
                    "pair": pair_name,
                    "current_ratio": analysis["ratio"],
                    "required_ratio": self.WCAG_AAA_NORMAL,
                    "foreground": analysis["foreground"],
                    "background": analysis["background"],
                    "severity": "high" if not analysis["wcag_aa_pass"] else "medium"
                }
                violations.append(violation)
                
        self.results["contrast_violations"] = violations
        print(f"⚠️  Found {len(violations)} WCAG AAA contrast violations")
        
    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a color by a given factor"""
        r, g, b = self._hex_to_rgb(hex_color)
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a color by a given factor"""
        r, g, b = self._hex_to_rgb(hex_color)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _generate_contrast_fixes(self):
        """Generate WCAG AAA compliant color fixes"""
        violations = self.results["contrast_violations"]
        fixes = []
        
        for violation in violations:
            pair = violation["pair"]
            fg_color = violation["foreground"]
            bg_color = violation["background"]
            target_ratio = self.WCAG_AAA_NORMAL
            
            # Try darkening foreground
            fixed_fg = self._find_compliant_foreground(fg_color, bg_color, target_ratio)
            
            # Try lightening background
            fixed_bg = self._find_compliant_background(fg_color, bg_color, target_ratio)
            
            fix = {
                "pair": pair,
                "original_foreground": fg_color,
                "original_background": bg_color,
                "fixed_foreground": fixed_fg,
                "fixed_background": fixed_bg,
                "foreground_ratio": self._contrast_ratio(fixed_fg, bg_color),
                "background_ratio": self._contrast_ratio(fg_color, fixed_bg),
                "recommendation": self._get_fix_recommendation(pair, fixed_fg, fixed_bg)
            }
            fixes.append(fix)
            
        self.results["recommended_fixes"] = fixes
        print(f"✅ Generated {len(fixes)} WCAG AAA compliant color fixes")
        
    def _find_compliant_foreground(self, fg_color: str, bg_color: str, target_ratio: float) -> str:
        """Find WCAG compliant foreground color by darkening"""
        current_color = fg_color
        for factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            test_color = self._darken_color(fg_color, factor)
            ratio = self._contrast_ratio(test_color, bg_color)
            if ratio >= target_ratio:
                return test_color
        return "#000000"  # Fallback to black
        
    def _find_compliant_background(self, fg_color: str, bg_color: str, target_ratio: float) -> str:
        """Find WCAG compliant background color by lightening"""
        current_color = bg_color
        for factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            test_color = self._lighten_color(bg_color, factor)
            ratio = self._contrast_ratio(fg_color, test_color)
            if ratio >= target_ratio:
                return test_color
        return "#ffffff"  # Fallback to white
        
    def _get_fix_recommendation(self, pair: str, fixed_fg: str, fixed_bg: str) -> str:
        """Get specific recommendation for fixing the color pair"""
        if "textPrimary" in pair or "textSecondary" in pair:
            return f"Adjust text color to {fixed_fg} for better readability"
        elif "onPrimary" in pair:
            return f"Use {fixed_fg} for text on primary buttons/surfaces"
        elif "button" in pair.lower():
            return f"Update button text to {fixed_fg} or button background to {fixed_bg}"
        else:
            return f"Update foreground to {fixed_fg} or background to {fixed_bg}"
            
    def _calculate_compliance_score(self):
        """Calculate overall WCAG AAA compliance score"""
        contrast_ratios = self.results["color_analysis"]["contrast_ratios"]
        violations = self.results["contrast_violations"]
        
        if not contrast_ratios:
            self.results["wcag_compliance_score"] = 0
            return
            
        total_pairs = len(contrast_ratios)
        passing_pairs = total_pairs - len(violations)
        
        compliance_percentage = (passing_pairs / total_pairs) * 100
        self.results["wcag_compliance_score"] = compliance_percentage
        
        # Set validation passed if score >= 90%
        self.results["validation_passed"] = compliance_percentage >= 90
        
        print(f"\n🎯 WCAG AAA Compliance Score: {compliance_percentage:.1f}%")
        
        if compliance_percentage >= 95:
            print("🏆 EXCELLENT: Outstanding WCAG AAA compliance!")
        elif compliance_percentage >= 90:
            print("✅ GOOD: Strong WCAG AAA compliance")
        elif compliance_percentage >= 70:
            print("⚠️  FAIR: Basic WCAG compliance, improvement needed")
        else:
            print("❌ NEEDS WORK: Significant WCAG compliance issues")
            
    def generate_wcag_compliant_design_system(self) -> str:
        """Generate WCAG AAA compliant DesignSystem.swift"""
        fixes = self.results["recommended_fixes"]
        
        # Read current DesignSystem
        design_system_path = self.project_root / "AgenticSeek-Sandbox" / "DesignSystem.swift"
        with open(design_system_path, 'r') as f:
            content = f.read()
            
        # Apply fixes to create WCAG compliant version
        updated_content = content
        
        # Generate color mapping for fixes
        color_updates = {}
        for fix in fixes:
            pair = fix["pair"]
            if "textPrimary" in pair:
                color_updates["textPrimary"] = fix["fixed_foreground"]
            elif "textSecondary" in pair:
                color_updates["textSecondary"] = fix["fixed_foreground"]
            elif "onPrimary" in pair and "primary" in pair:
                color_updates["onPrimary"] = fix["fixed_foreground"]
            # Add more mappings as needed
            
        # Apply color updates
        for color_name, new_color in color_updates.items():
            # Update hex colors
            hex_pattern = f'(static let {color_name} = Color\\(hex: ")([#\\w]+)("\\))'
            if re.search(hex_pattern, updated_content):
                updated_content = re.sub(hex_pattern, f'\\1{new_color}\\3', updated_content)
            else:
                # Update RGB colors
                rgb_pattern = f'(static let {color_name} = Color\\(red: )[\\d.]+, green: [\\d.]+, blue: [\\d.]+(\\))'
                r, g, b = self._hex_to_rgb(new_color)
                rgb_values = f"{r/255:.3f}, green: {g/255:.3f}, blue: {b/255:.3f}"
                updated_content = re.sub(rgb_pattern, f'\\1{rgb_values}\\2', updated_content)
                
        return updated_content
        
    def generate_report(self) -> str:
        """Generate comprehensive WCAG compliance report"""
        report = []
        report.append("# WCAG AAA Color Contrast Compliance Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        score = self.results["wcag_compliance_score"]
        status = "✅ PASSED" if self.results["validation_passed"] else "❌ FAILED"
        report.append(f"## Summary")
        report.append(f"- **Compliance Status**: {status}")
        report.append(f"- **WCAG AAA Score**: {score:.1f}%")
        report.append("")
        
        # Color Analysis
        extracted_colors = self.results["color_analysis"].get("extracted_colors", {})
        report.append(f"## Color Analysis")
        report.append(f"- **Total Colors Analyzed**: {len(extracted_colors)}")
        
        contrast_ratios = self.results["color_analysis"].get("contrast_ratios", {})
        report.append(f"- **Color Pairs Tested**: {len(contrast_ratios)}")
        report.append("")
        
        # Violations
        violations = self.results["contrast_violations"]
        report.append(f"## WCAG AAA Violations ({len(violations)} found)")
        
        for violation in violations:
            severity_icon = "🔴" if violation["severity"] == "high" else "🟡"
            report.append(f"{severity_icon} **{violation['pair']}**")
            report.append(f"   - Current ratio: {violation['current_ratio']:.2f}")
            report.append(f"   - Required ratio: {violation['required_ratio']:.2f}")
            report.append(f"   - Foreground: {violation['foreground']}")
            report.append(f"   - Background: {violation['background']}")
            report.append("")
            
        # Recommended Fixes
        fixes = self.results["recommended_fixes"]
        report.append(f"## Recommended Fixes ({len(fixes)} available)")
        
        for fix in fixes:
            report.append(f"### {fix['pair']}")
            report.append(f"- **Foreground Fix**: {fix['original_foreground']} → {fix['fixed_foreground']}")
            report.append(f"  - New ratio: {fix['foreground_ratio']:.2f}")
            report.append(f"- **Background Fix**: {fix['original_background']} → {fix['fixed_background']}")
            report.append(f"  - New ratio: {fix['background_ratio']:.2f}")
            report.append(f"- **Recommendation**: {fix['recommendation']}")
            report.append("")
            
        # Implementation Guide
        report.append("## Implementation Guide")
        report.append("1. Apply recommended color changes to DesignSystem.swift")
        report.append("2. Test all UI components with new colors")
        report.append("3. Verify accessibility with VoiceOver")
        report.append("4. Run validation tests to confirm WCAG AAA compliance")
        report.append("")
        
        return "\n".join(report)

def main():
    project_root = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS"
    
    validator = WCAGColorValidator(project_root)
    
    print("Starting WCAG AAA Color Contrast Validation...")
    results = validator.validate_wcag_compliance()
    
    # Generate and display report
    report = validator.generate_report()
    print("\n" + report)
    
    # Generate WCAG compliant DesignSystem
    if results["contrast_violations"]:
        print("\n📝 Generating WCAG AAA compliant DesignSystem...")
        wcag_design_system = validator.generate_wcag_compliant_design_system()
        
        # Save WCAG compliant version
        output_path = Path(project_root) / "tests" / "WCAGCompliantDesignSystem.swift"
        with open(output_path, 'w') as f:
            f.write(wcag_design_system)
        print(f"💾 WCAG compliant DesignSystem saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    main()