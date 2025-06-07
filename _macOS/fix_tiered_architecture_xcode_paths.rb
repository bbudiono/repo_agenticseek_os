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

puts "🔨 Fixing Tiered Architecture file paths in Xcode project..."

# Remove incorrectly referenced files first
files_to_remove = [
  'TierManager.swift',
  'AgentLimitEnforcer.swift', 
  'UsageMonitor.swift',
  'TierUpgradeManager.swift',
  'DynamicAgentScaler.swift',
  'TierAnalytics.swift',
  'TieredAgentCoordinator.swift',
  'TieredArchitectureIntegration.swift',
  'TierConfigurationView.swift',
  'AgentDashboardView.swift',
  'TierUpgradeView.swift',
  'UsageAnalyticsView.swift'
]

files_to_remove.each do |file_name|
  project.files.each do |file_ref|
    if file_ref.path == file_name
      puts "🗑️ Removing incorrect reference: #{file_name}"
      file_ref.remove_from_project
    end
  end
end

# Find the main group
main_group = project.main_group.find_subpath('AgenticSeek', true)

# Create TieredArchitecture group if it doesn't exist
tiered_group = main_group.find_subpath('TieredArchitecture', true)

# Create Core subgroup
core_group = tiered_group.find_subpath('Core', true)

# Create Views subgroup  
views_group = tiered_group.find_subpath('Views', true)

# Core files with correct paths
core_files = [
  { file: 'TieredArchitecture/Core/TierManager.swift', name: 'TierManager.swift' },
  { file: 'TieredArchitecture/Core/AgentLimitEnforcer.swift', name: 'AgentLimitEnforcer.swift' },
  { file: 'TieredArchitecture/Core/UsageMonitor.swift', name: 'UsageMonitor.swift' },
  { file: 'TieredArchitecture/Core/TierUpgradeManager.swift', name: 'TierUpgradeManager.swift' },
  { file: 'TieredArchitecture/Core/DynamicAgentScaler.swift', name: 'DynamicAgentScaler.swift' },
  { file: 'TieredArchitecture/Core/TierAnalytics.swift', name: 'TierAnalytics.swift' },
  { file: 'TieredArchitecture/Core/TieredAgentCoordinator.swift', name: 'TieredAgentCoordinator.swift' },
  { file: 'TieredArchitecture/Core/TieredArchitectureIntegration.swift', name: 'TieredArchitectureIntegration.swift' }
]

# View files with correct paths
view_files = [
  { file: 'TieredArchitecture/Views/TierConfigurationView.swift', name: 'TierConfigurationView.swift' },
  { file: 'TieredArchitecture/Views/AgentDashboardView.swift', name: 'AgentDashboardView.swift' },
  { file: 'TieredArchitecture/Views/TierUpgradeView.swift', name: 'TierUpgradeView.swift' },
  { file: 'TieredArchitecture/Views/UsageAnalyticsView.swift', name: 'UsageAnalyticsView.swift' }
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

puts "🎯 Successfully fixed Xcode project file paths!"
puts "📋 Files added to target: #{target.name}"