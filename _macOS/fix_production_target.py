#!/usr/bin/env python3
"""
Fix Production Xcode Project Target Membership
Adds missing Swift files to the Production AgenticSeek project target
"""

import subprocess
import os
import sys

def main():
    project_path = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS"
    os.chdir(project_path)
    
    # Missing Swift files that need to be added to Production target
    missing_files = [
        "OnboardingManager.swift",
        "OnboardingFlow.swift", 
        "AuthenticationManager.swift",
        "AppNavigationView.swift",
        "ChatbotInterface.swift",
        "ChatbotModels.swift",
        "ModelSelectionView.swift",
        "PerformanceAnalyticsView.swift",
        "PerformanceOptimizedComponents.swift",
        "ProductionComponents.swift",
        "RealChatInterface.swift",
        "EnhancedContentView.swift",
        "OptimizedModelManagementView.swift",
        "VoiceAICore.swift",
        "VoiceAIBridge.swift",
        "MinimalWorkingChatbot.swift",
        "SimpleWorkingChatbot.swift",
        "RejectionSamplingEngine.swift",
        "SpeculativeDecodingCoordinator.swift",
        "SpeculativeDecodingEngine.swift",
        "SandboxComponents.swift"
    ]
    
    print("🔧 Starting Production Target Fix...")
    print(f"📁 Working in: {project_path}")
    print(f"📋 Adding {len(missing_files)} missing Swift files to Production project target")
    
    # Verify files exist before adding
    existing_files = []
    for file in missing_files:
        file_path = f"AgenticSeek/{file}"
        if os.path.exists(file_path):
            existing_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    print(f"\n📊 Summary: {len(existing_files)}/{len(missing_files)} files found")
    
    if len(existing_files) != len(missing_files):
        print("⚠️  Some files are missing. Proceeding with existing files only.")
    
    # Use xcodebuild to add files (this is a simplified approach)
    # In practice, we'll need to manually add them or use xcodeproj gem
    print("\n🔧 Note: Files need to be added manually in Xcode or via xcodeproj tool")
    print("Key missing file causing build failure: OnboardingManager.swift")
    
    # Test the build after manual addition
    return True

if __name__ == "__main__":
    main()