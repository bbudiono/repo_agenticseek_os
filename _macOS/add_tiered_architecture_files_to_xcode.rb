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

# Create TieredArchitecture group if it doesn't exist
tiered_group = main_group.find_subpath('TieredArchitecture', true)

# Create Core subgroup
core_group = tiered_group.find_subpath('Core', true)

# Create Views subgroup  
views_group = tiered_group.find_subpath('Views', true)

# Create Tests subgroup
tests_group = tiered_group.find_subpath('Tests', true)

# Core files to add
core_files = [
  'TieredArchitecture/Core/TierManager.swift',
  'TieredArchitecture/Core/AgentLimitEnforcer.swift',
  'TieredArchitecture/Core/UsageMonitor.swift',
  'TieredArchitecture/Core/TierUpgradeManager.swift',
  'TieredArchitecture/Core/DynamicAgentScaler.swift',
  'TieredArchitecture/Core/TierAnalytics.swift',
  'TieredArchitecture/Core/TieredAgentCoordinator.swift',
  'TieredArchitecture/Core/TieredArchitectureIntegration.swift'
]

# View files to add
view_files = [
  'TieredArchitecture/Views/TierConfigurationView.swift',
  'TieredArchitecture/Views/AgentDashboardView.swift',
  'TieredArchitecture/Views/TierUpgradeView.swift',
  'TieredArchitecture/Views/UsageAnalyticsView.swift'
]

# Test files to add
test_files = [
  'Tests/TieredArchitectureTests/TierManagerTest.swift',
  'Tests/TieredArchitectureTests/AgentLimitEnforcerTest.swift',
  'Tests/TieredArchitectureTests/UsageMonitorTest.swift',
  'Tests/TieredArchitectureTests/TierUpgradeManagerTest.swift',
  'Tests/TieredArchitectureTests/DynamicAgentScalerTest.swift',
  'Tests/TieredArchitectureTests/TierAnalyticsTest.swift',
  'Tests/TieredArchitectureTests/TieredAgentCoordinatorTest.swift',
  'Tests/TieredArchitectureTests/TierConfigurationViewTest.swift',
  'Tests/TieredArchitectureTests/AgentDashboardViewTest.swift',
  'Tests/TieredArchitectureTests/TierUpgradeViewTest.swift',
  'Tests/TieredArchitectureTests/UsageAnalyticsViewTest.swift',
  'Tests/TieredArchitectureTests/TieredArchitectureIntegrationTest.swift'
]

puts "🔨 Adding Tiered Architecture files to Xcode project..."

# Function to add files to group and target
def add_files_to_group(files, group, target, project)
  files.each do |file_path|
    file_name = File.basename(file_path)
    absolute_path = File.join(Dir.pwd, 'AgenticSeek', file_path)
    
    # Check if file exists
    unless File.exist?(absolute_path)
      puts "⚠️  File not found: #{absolute_path}"
      next
    end
    
    # Check if file is already in project
    existing_file = group.files.find { |f| f.path == file_name }
    if existing_file
      puts "ℹ️  File already in project: #{file_name}"
      next
    end
    
    # Add file to group
    file_ref = group.new_reference(absolute_path)
    file_ref.last_known_file_type = 'sourcecode.swift'
    file_ref.path = file_name
    file_ref.source_tree = '<group>'
    
    # Add to build phase (only for non-test files)
    unless file_path.include?('Test')
      target.source_build_phase.add_file_reference(file_ref)
    end
    
    puts "✅ Added: #{file_name}"
  end
end

# Add core files
add_files_to_group(core_files, core_group, target, project)

# Add view files
add_files_to_group(view_files, views_group, target, project)

# Add test files
add_files_to_group(test_files, tests_group, target, project)

# Save the project
project.save

puts "🎯 Successfully updated Xcode project with Tiered Architecture files!"
puts "📋 Files added to target: #{target.name}"