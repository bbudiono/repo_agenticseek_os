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

# Find the main group
main_group = project.main_group.find_subpath('AgenticSeek', true)

# Create SingleAgentMode group if it doesn't exist
single_agent_group = main_group.find_subpath('SingleAgentMode', true)

# Single Agent Mode files to add
single_agent_files = [
  'SingleAgentMode/SingleAgentModeView.swift',
  'SingleAgentMode/GenericModelScanner.swift', 
  'SingleAgentMode/LMStudioDetector.swift',
  'SingleAgentMode/LocalContextManager.swift',
  'SingleAgentMode/MLACSModeManager.swift',
  'SingleAgentMode/ModelCompatibilityAnalyzer.swift',
  'SingleAgentMode/ModelPerformanceOptimizer.swift',
  'SingleAgentMode/OfflineAgentCoordinator.swift',
  'SingleAgentMode/OfflineQualityAssurance.swift',
  'SingleAgentMode/OllamaDetector.swift',
  'SingleAgentMode/SingleAgentBenchmark.swift',
  'SingleAgentMode/SingleAgentUXValidator.swift',
  'SingleAgentMode/SystemPerformanceAnalyzer.swift'
]

puts "🔨 Adding Single Agent Mode files to Xcode project..."

single_agent_files.each do |file_path|
  file_name = File.basename(file_path)
  absolute_path = File.join(Dir.pwd, 'AgenticSeek', file_path)
  
  # Check if file exists
  unless File.exist?(absolute_path)
    puts "⚠️  File not found: #{absolute_path}"
    next
  end
  
  # Check if file is already in project
  existing_file = single_agent_group.files.find { |f| f.path == file_name }
  if existing_file
    puts "ℹ️  File already in project: #{file_name}"
    next
  end
  
  # Add file to group
  file_ref = single_agent_group.new_reference(absolute_path)
  file_ref.last_known_file_type = 'sourcecode.swift'
  file_ref.path = file_name
  file_ref.source_tree = '<group>'
  
  # Add to build phase
  target.source_build_phase.add_file_reference(file_ref)
  
  puts "✅ Added: #{file_name}"
end

# Save the project
project.save

puts "🎯 Successfully updated Xcode project with Single Agent Mode files!"
puts "📋 Files added to target: #{target.name}"