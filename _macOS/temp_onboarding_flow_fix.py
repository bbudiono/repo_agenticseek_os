#!/usr/bin/env python3
"""
Remove duplicate OnboardingManager class from OnboardingFlow.swift
"""

def fix_onboarding_flow():
    file_path = 'AgenticSeek/OnboardingFlow.swift'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start and end of the duplicate OnboardingManager class
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if '// MARK: - Onboarding Manager' in line:
            start_line = i
        elif start_line is not None and line.strip() == '}' and 'class OnboardingManager' in ''.join(lines[start_line:i+5]):
            end_line = i
            break
    
    if start_line is not None and end_line is not None:
        print(f"🔧 Removing duplicate OnboardingManager class (lines {start_line+1}-{end_line+1})")
        
        # Remove the duplicate class lines
        new_lines = lines[:start_line] + lines[end_line+1:]
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        
        print("✅ Duplicate OnboardingManager class removed from OnboardingFlow.swift")
        print(f"📊 Removed {end_line - start_line + 1} lines")
    else:
        print("❌ Could not find duplicate OnboardingManager class boundaries")

if __name__ == "__main__":
    fix_onboarding_flow()