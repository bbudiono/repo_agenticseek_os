#!/usr/bin/env python3
"""
TDD Comment Block Validator for AgenticSeek
==========================================

* Purpose: Validates that all source files have required TDD comment blocks
* Usage: Called automatically by pre-commit hooks or run manually
* Exit Codes: 0 = success, 1 = validation failures
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

def validate_tdd_comment_block(file_path: str) -> Dict[str, Any]:
    """Validate TDD comment block in a file"""
    required_fields = [
        "Purpose:",
        "Issues & Complexity Summary:",
        "Key Complexity Drivers:",
        "AI Pre-Task Self-Assessment",
        "Final Code Complexity",
        "Overall Result Score"
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_fields = []
        for field in required_fields:
            if field not in content:
                missing_fields.append(field)
        
        # Check for complete comment structure
        has_header_block = "Purpose:" in content[:500]  # Should be near top
        
        return {
            "file": file_path,
            "valid": len(missing_fields) == 0 and has_header_block,
            "missing_fields": missing_fields,
            "has_header_block": has_header_block
        }
    
    except Exception as e:
        return {
            "file": file_path,
            "valid": False,
            "error": str(e),
            "missing_fields": required_fields,
            "has_header_block": False
        }

def main():
    """Main validation function"""
    if len(sys.argv) < 2:
        print("Usage: validate_tdd_comments.py <file1> [file2] ...")
        sys.exit(1)
    
    files_to_check = sys.argv[1:]
    validation_results = []
    failed_files = []
    
    for file_path in files_to_check:
        if not file_path.endswith('.py'):
            continue
            
        # Skip certain files
        skip_patterns = ['__init__.py', 'test_', '_test.py']
        if any(pattern in os.path.basename(file_path) for pattern in skip_patterns):
            continue
        
        result = validate_tdd_comment_block(file_path)
        validation_results.append(result)
        
        if not result["valid"]:
            failed_files.append(result)
    
    # Report results
    if failed_files:
        print("❌ TDD Comment Validation FAILED")
        print("="*50)
        
        for failure in failed_files:
            print(f"\n📁 File: {failure['file']}")
            
            if 'error' in failure:
                print(f"  ❌ Error: {failure['error']}")
            else:
                if not failure['has_header_block']:
                    print("  ❌ Missing TDD comment header block")
                
                if failure['missing_fields']:
                    print("  ❌ Missing required fields:")
                    for field in failure['missing_fields']:
                        print(f"     • {field}")
        
        print(f"\n💡 Fix by adding this comment block to each source file:")
        print("""
/*
* Purpose: [Brief description of file's purpose]
* Issues & Complexity Summary: [Overall complexity summary]
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): [~XXX]
  - Core Algorithm Complexity: [Low/Med/High/Very High]
  - Dependencies: [X New, Y Mod]
  - State Management Complexity: [Low/Med/High/Very High]
  - Novelty/Uncertainty Factor: [Low/Med/High]
* AI Pre-Task Self-Assessment (Est. Solution Difficulty %): [XX%]
* Problem Estimate (Inherent Problem Difficulty %): [XX%]
* Initial Code Complexity Estimate %: [XX%]
* Justification for Estimates: [brief reasoning]
* Final Code Complexity (Actual %): [XX%]
* Overall Result Score (Success & Quality %): [XX%]
* Key Variances/Learnings: [insights gained]
* Last Updated: [YYYY-MM-DD]
*/
        """)
        
        print(f"\n📊 Summary: {len(failed_files)} files failed validation")
        sys.exit(1)
    
    else:
        print(f"✅ TDD Comment Validation PASSED")
        print(f"📊 Validated {len(validation_results)} files successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()