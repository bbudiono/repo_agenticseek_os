#!/usr/bin/env ruby

require 'xcodeproj'

# Path to the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Find the main target
target = project.targets.find { |t| t.name == 'AgenticSeek' }

if target.nil?
  puts "❌ Could not find AgenticSeek target"
  exit 1
end

puts "🔨 Fixing ALL file paths in Xcode project..."

# Single Agent Mode files that need path fixing
single_agent_files = [
  'SingleAgentModeView.swift',
  'GenericModelScanner.swift',
  'LMStudioDetector.swift',
  'LocalContextManager.swift',
  'MLACSModeManager.swift',
  'ModelCompatibilityAnalyzer.swift',
  'ModelPerformanceOptimizer.swift',
  'OfflineAgentCoordinator.swift',
  'OfflineQualityAssurance.swift',
  'OllamaDetector.swift',
  'SingleAgentBenchmark.swift',
  'SingleAgentUXValidator.swift',
  'SystemPerformanceAnalyzer.swift'
]

# Remove all incorrectly referenced files
single_agent_files.each do |file_name|
  project.files.each do |file_ref|
    if file_ref.path == file_name
      puts "🗑️ Removing incorrect Single Agent reference: #{file_name}"
      file_ref.remove_from_project
    end
  end
end

# Find the main group
main_group = project.main_group.find_subpath('AgenticSeek', true)

# Create SingleAgentMode group if it doesn't exist
single_agent_group = main_group.find_subpath('SingleAgentMode', true)

# Single Agent Mode files with correct paths
single_agent_file_mappings = [
  { file: 'SingleAgentMode/SingleAgentModeView.swift', name: 'SingleAgentModeView.swift' },
  { file: 'SingleAgentMode/GenericModelScanner.swift', name: 'GenericModelScanner.swift' },
  { file: 'SingleAgentMode/LMStudioDetector.swift', name: 'LMStudioDetector.swift' },
  { file: 'SingleAgentMode/LocalContextManager.swift', name: 'LocalContextManager.swift' },
  { file: 'SingleAgentMode/MLACSModeManager.swift', name: 'MLACSModeManager.swift' },
  { file: 'SingleAgentMode/ModelCompatibilityAnalyzer.swift', name: 'ModelCompatibilityAnalyzer.swift' },
  { file: 'SingleAgentMode/ModelPerformanceOptimizer.swift', name: 'ModelPerformanceOptimizer.swift' },
  { file: 'SingleAgentMode/OfflineAgentCoordinator.swift', name: 'OfflineAgentCoordinator.swift' },
  { file: 'SingleAgentMode/OfflineQualityAssurance.swift', name: 'OfflineQualityAssurance.swift' },
  { file: 'SingleAgentMode/OllamaDetector.swift', name: 'OllamaDetector.swift' },
  { file: 'SingleAgentMode/SingleAgentBenchmark.swift', name: 'SingleAgentBenchmark.swift' },
  { file: 'SingleAgentMode/SingleAgentUXValidator.swift', name: 'SingleAgentUXValidator.swift' },
  { file: 'SingleAgentMode/SystemPerformanceAnalyzer.swift', name: 'SystemPerformanceAnalyzer.swift' }
]

# Function to add files to group and target with correct paths
def add_files_to_group_with_correct_paths(files, group, target, project, base_path)
  files.each do |file_info|
    file_path = file_info[:file]
    file_name = file_info[:name]
    absolute_path = File.join(base_path, file_path)
    
    # Check if file exists
    unless File.exist?(absolute_path)
      puts "⚠️  File not found: #{absolute_path}"
      next
    end
    
    # Check if file is already in the group
    existing_file = group.files.find { |f| f.path == file_name }
    if existing_file
      puts "ℹ️  File already in group: #{file_name}"
      next
    end
    
    # Add file to group with correct relative path
    file_ref = group.new_reference(file_path)
    file_ref.last_known_file_type = 'sourcecode.swift'
    file_ref.source_tree = '<group>'
    
    # Add to build phase
    target.source_build_phase.add_file_reference(file_ref)
    
    puts "✅ Added with correct path: #{file_name} -> #{file_path}"
  end
end

base_path = File.join(Dir.pwd, 'AgenticSeek')

# Add Single Agent Mode files with correct paths
add_files_to_group_with_correct_paths(single_agent_file_mappings, single_agent_group, target, project, base_path)

# Save the project
project.save

puts "🎯 Successfully fixed all Xcode project file paths!"
puts "📋 Files added to target: #{target.name}"