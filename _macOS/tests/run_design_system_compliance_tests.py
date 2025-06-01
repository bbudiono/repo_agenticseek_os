import subprocess
import json
import os

# Define the root directory of the SwiftUI project to scan
SWIFTUI_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AgenticSeek'))

def run_grep_search(query, include_pattern='*.swift'):
    """
    Runs a grep search command within the SwiftUI project root.
    Returns a list of dictionaries, each representing a found match.
    """
    command = [
        'grep', '-r', '-n',  # -r for recursive, -n for line number
        '--include', include_pattern,
        '--exclude-dir', 'build',  # Exclude build directories
        '--exclude-dir', '.build',
        '--exclude-dir', 'DerivedData',
        '--exclude-dir', 'Pods',
        query,
        SWIFTUI_PROJECT_ROOT
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        results = []
        for line in process.stdout.splitlines():
            # Example line: /path/to/file.swift:123:  Color(red: 0.5, green: 0.6, blue: 0.7)
            parts = line.split(':', 2) # Split into filepath, line_number, rest
            if len(parts) >= 3:
                filepath = parts[0]
                line_number = int(parts[1])
                match_content = parts[2].strip()
                # Exclude the DesignSystem.swift file from results
                if not filepath.endswith('DesignSystem.swift') and not filepath.endswith('Strings.swift'):
                    results.append({
                        "filepath": filepath,
                        "line_number": line_number,
                        "content": match_content
                    })
        return results
    except subprocess.CalledProcessError as e:
        if e.returncode == 1: # grep returns 1 if no lines are selected
            return []
        print(f"Error running grep: {e}")
        print(f"Stderr: {e.stderr}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def main():
    print("Running Design System Compliance Tests...")

    hardcoded_colors = run_grep_search(r'#[0-9A-Fa-f]{6}|Color\(red: [0-9.]+, green: [0-9.]+, blue: [0-9.]+(, opacity: [0-9.]+)?\)')
    hardcoded_fonts = run_grep_search(r'Font\.(system|custom)\(.*?\)')
    hardcoded_spacing = run_grep_search(r'\.(padding|frame|cornerRadius|offset)\(.+?\)')
    hardcoded_strings = run_grep_search(r'"[^"]+"')

    # Filter out empty results and DesignSystem.swift/Strings.swift entries
    hardcoded_colors = [r for r in hardcoded_colors if not r['filepath'].endswith('DesignSystem.swift') and not r['filepath'].endswith('Strings.swift')]
    hardcoded_fonts = [r for r in hardcoded_fonts if not r['filepath'].endswith('DesignSystem.swift') and not r['filepath'].endswith('Strings.swift')]
    hardcoded_spacing = [r for r in hardcoded_spacing if not r['filepath'].endswith('DesignSystem.swift') and not r['filepath'].endswith('Strings.swift')]
    hardcoded_strings = [r for r in hardcoded_strings if not r['filepath'].endswith('DesignSystem.swift') and not r['filepath'].endswith('Strings.swift')]

    # Prepare results for consumption by the Swift test suite or reporting
    test_results = {
        "hardcoded_colors": hardcoded_colors,
        "hardcoded_fonts": hardcoded_fonts,
        "hardcoded_spacing": hardcoded_spacing,
        "hardcoded_strings": hardcoded_strings
    }

    # Output results to a JSON file or print for a Swift test runner to consume
    output_filename = os.path.join(os.path.dirname(__file__), 'design_system_compliance_results.json')
    with open(output_filename, 'w') as f:
        json.dump(test_results, f, indent=4)

    print(f"Design system compliance scan complete. Results saved to {output_filename}")
    
    # Provide a summary of findings
    summary = {
        "total_hardcoded_colors": len(hardcoded_colors),
        "total_hardcoded_fonts": len(hardcoded_fonts),
        "total_hardcoded_spacing": len(hardcoded_spacing),
        "total_hardcoded_strings": len(hardcoded_strings),
    }
    print("Summary of Hardcoded Value Findings:")
    for key, value in summary.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")

    # Determine overall compliance based on findings
    if all(len(v) == 0 for v in test_results.values()):
        print("\nOverall Compliance: ✅ Fully Compliant - No hardcoded values found outside DesignSystem/Strings.swift.")
        return 0 # Indicate success
    else:
        print("\nOverall Compliance: ⚠️ Warnings - Hardcoded values detected. Please address them using DesignSystem/Strings.swift.")
        return 1 # Indicate failure/warnings

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 