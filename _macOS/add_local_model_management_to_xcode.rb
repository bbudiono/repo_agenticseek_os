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

puts "🔨 Adding Local Model Management files to Xcode project..."

# Find the main group
main_group = project.main_group.find_subpath('AgenticSeek', true)

# Create LocalModelManagement group if it doesn't exist
local_model_group = main_group.find_subpath('LocalModelManagement', true)

# Create Core subgroup
core_group = local_model_group.find_subpath('Core', true)

# Create Views subgroup  
views_group = local_model_group.find_subpath('Views', true)

# Core files with correct paths
core_files = [
  { file: 'LocalModelManagement/Core/LocalModelRegistry.swift', name: 'LocalModelRegistry.swift' },
  { file: 'LocalModelManagement/Core/OllamaIntegration.swift', name: 'OllamaIntegration.swift' },
  { file: 'LocalModelManagement/Core/LMStudioIntegration.swift', name: 'LMStudioIntegration.swift' },
  { file: 'LocalModelManagement/Core/ModelDownloadManager.swift', name: 'ModelDownloadManager.swift' },
  { file: 'LocalModelManagement/Core/ModelCapabilityAnalyzer.swift', name: 'ModelCapabilityAnalyzer.swift' },
  { file: 'LocalModelManagement/Core/IntelligentModelSelector.swift', name: 'IntelligentModelSelector.swift' },
  { file: 'LocalModelManagement/Core/ModelVersionManager.swift', name: 'ModelVersionManager.swift' },
  { file: 'LocalModelManagement/Core/ModelPerformanceMonitor.swift', name: 'ModelPerformanceMonitor.swift' }
]

# View files with correct paths
view_files = [
  { file: 'LocalModelManagement/Views/LocalModelManagementView.swift', name: 'LocalModelManagementView.swift' },
  { file: 'LocalModelManagement/Views/ModelDiscoveryView.swift', name: 'ModelDiscoveryView.swift' },
  { file: 'LocalModelManagement/Views/ModelPerformanceDashboard.swift', name: 'ModelPerformanceDashboard.swift' },
  { file: 'LocalModelManagement/Views/LocalModelConfigurationView.swift', name: 'LocalModelConfigurationView.swift' }
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

# Add core files with correct paths
add_files_to_group_with_correct_paths(core_files, core_group, target, project, base_path)

# Add view files with correct paths
add_files_to_group_with_correct_paths(view_files, views_group, target, project, base_path)

# Save the project
project.save

puts "🎯 Successfully added Local Model Management files to Xcode project!"
puts "📋 Files added to target: #{target.name}"