#!/usr/bin/env ruby
require 'xcodeproj'

# Open the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first
puts "📋 Target: #{target.name}"

# Get the main group
main_group = project.main_group.find_subpath('AgenticSeek')
puts "📁 Main group: #{main_group.path}"

# Missing Swift files to add
missing_files = [
  'OnboardingManager.swift',
  'OnboardingFlow.swift', 
  'AuthenticationManager.swift',
  'AppNavigationView.swift',
  'ChatbotInterface.swift',
  'ChatbotModels.swift',
  'ModelSelectionView.swift',
  'PerformanceAnalyticsView.swift',
  'PerformanceOptimizedComponents.swift',
  'ProductionComponents.swift',
  'RealChatInterface.swift',
  'EnhancedContentView.swift',
  'OptimizedModelManagementView.swift',
  'VoiceAICore.swift',
  'VoiceAIBridge.swift',
  'MinimalWorkingChatbot.swift',
  'SimpleWorkingChatbot.swift',
  'RejectionSamplingEngine.swift',
  'SpeculativeDecodingCoordinator.swift',
  'SpeculativeDecodingEngine.swift',
  'SandboxComponents.swift'
]

puts "\n🔧 Adding #{missing_files.length} missing Swift files to target..."

missing_files.each do |filename|
  file_path = "AgenticSeek/#{filename}"
  
  # Check if file exists on disk
  unless File.exist?(file_path)
    puts "❌ File not found: #{file_path}"
    next
  end
  
  # Check if file is already in project
  existing_file = main_group.files.find { |f| f.path == filename }
  if existing_file
    puts "⚠️  File already in project (but not in target): #{filename}"
    file_ref = existing_file
  else
    # Add file reference to project
    file_ref = main_group.new_reference(filename)
    puts "➕ Added file reference: #{filename}"
  end
  
  # Add file to target build phases (Sources)
  build_file = target.add_file_references([file_ref]).first
  if build_file
    puts "✅ Added to target: #{filename}"
  else
    puts "❌ Failed to add to target: #{filename}"
  end
end

# Special handling for Core directory files
core_files = ['VoiceAICore.swift', 'VoiceAIBridge.swift']
core_group = main_group.find_subpath('Core') || main_group.new_group('Core')

core_files.each do |filename|
  file_path = "AgenticSeek/Core/#{filename}"
  
  if File.exist?(file_path)
    # Check if already exists in Core group
    existing_file = core_group.files.find { |f| f.path == filename }
    unless existing_file
      file_ref = core_group.new_reference(filename)
      file_ref.source_tree = 'SOURCE_ROOT'
      file_ref.path = "AgenticSeek/Core/#{filename}"
      target.add_file_references([file_ref])
      puts "✅ Added Core file: #{filename}"
    end
  end
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Try building now - OnboardingManager.swift should be available"