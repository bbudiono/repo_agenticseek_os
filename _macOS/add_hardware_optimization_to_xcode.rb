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

puts "🔨 Adding Hardware Optimization files to Xcode project..."

# Find the main group
main_group = project.main_group.find_subpath('AgenticSeek', true)

# Create HardwareOptimization group if it doesn't exist
hardware_group = main_group.find_subpath('HardwareOptimization', true)

# Create Core subgroup
core_group = hardware_group.find_subpath('Core', true)

# Create Views subgroup  
views_group = hardware_group.find_subpath('Views', true)

# Core files with correct paths
core_files = [
  { file: 'HardwareOptimization/Core/AppleSiliconProfiler.swift', name: 'AppleSiliconProfiler.swift' },
  { file: 'HardwareOptimization/Core/MemoryOptimizer.swift', name: 'MemoryOptimizer.swift' },
  { file: 'HardwareOptimization/Core/GPUAccelerationManager.swift', name: 'GPUAccelerationManager.swift' },
  { file: 'HardwareOptimization/Core/ThermalManagementSystem.swift', name: 'ThermalManagementSystem.swift' },
  { file: 'HardwareOptimization/Core/PowerManagementOptimizer.swift', name: 'PowerManagementOptimizer.swift' },
  { file: 'HardwareOptimization/Core/PerformanceProfiler.swift', name: 'PerformanceProfiler.swift' },
  { file: 'HardwareOptimization/Core/HardwareCapabilityDetector.swift', name: 'HardwareCapabilityDetector.swift' },
  { file: 'HardwareOptimization/Core/ModelHardwareOptimizer.swift', name: 'ModelHardwareOptimizer.swift' }
]

# View files with correct paths
view_files = [
  { file: 'HardwareOptimization/Views/HardwareOptimizationDashboard.swift', name: 'HardwareOptimizationDashboard.swift' },
  { file: 'HardwareOptimization/Views/PerformanceMonitoringView.swift', name: 'PerformanceMonitoringView.swift' },
  { file: 'HardwareOptimization/Views/ThermalManagementView.swift', name: 'ThermalManagementView.swift' },
  { file: 'HardwareOptimization/Views/HardwareConfigurationView.swift', name: 'HardwareConfigurationView.swift' }
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

puts "🎯 Successfully added Hardware Optimization files to Xcode project!"
puts "📋 Files added to target: #{target.name}"