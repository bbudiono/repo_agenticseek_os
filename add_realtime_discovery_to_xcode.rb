#!/usr/bin/env ruby

require 'xcodeproj'

# Path to the Xcode project
project_path = '/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/AgenticSeek.xcodeproj'

# Open the project
project = Xcodeproj::Project.open(project_path)

# Find the main target
target = project.targets.find { |t| t.name == 'AgenticSeek' }

if target.nil?
  puts "❌ Target 'AgenticSeek' not found"
  exit 1
end

# Get the main group
main_group = project.main_group

# Find or create RealtimeModelDiscovery group
realtime_discovery_group = main_group.groups.find { |group| group.name == 'RealtimeModelDiscovery' }
if realtime_discovery_group.nil?
  realtime_discovery_group = main_group.new_group('RealtimeModelDiscovery')
  puts "✅ Created RealtimeModelDiscovery group"
end

# Create Core subgroup
core_group = realtime_discovery_group.groups.find { |group| group.name == 'Core' }
if core_group.nil?
  core_group = realtime_discovery_group.new_group('Core')
  puts "✅ Created Core subgroup"
end

# Create Views subgroup  
views_group = realtime_discovery_group.groups.find { |group| group.name == 'Views' }
if views_group.nil?
  views_group = realtime_discovery_group.new_group('Views')
  puts "✅ Created Views subgroup"
end

# Create Tests subgroup
tests_group = realtime_discovery_group.groups.find { |group| group.name == 'Tests' }
if tests_group.nil?
  tests_group = realtime_discovery_group.new_group('Tests')
  puts "✅ Created Tests subgroup"
end

# Core files to add
core_files = [
  'ModelDiscoveryEngine.swift',
  'ModelRegistryManager.swift', 
  'CapabilityDetector.swift',
  'ProviderScanner.swift',
  'ModelRecommendationEngine.swift',
  'PerformanceProfiler.swift',
  'CompatibilityAnalyzer.swift',
  'ModelIndexer.swift',
  'DiscoveryScheduler.swift',
  'ModelValidator.swift',
  'DiscoveryModels.swift',
  'DiscoveryExtensions.swift',
  'DiscoveryUtilities.swift'
]

# View files to add
view_files = [
  'ModelDiscoveryDashboard.swift',
  'ModelBrowserView.swift',
  'RecommendationView.swift',
  'DiscoverySettingsView.swift'
]

# Test files to add
test_files = [
  'ModelDiscoveryEngineTest.swift',
  'ModelRegistryManagerTest.swift',
  'CapabilityDetectorTest.swift',
  'ProviderScannerTest.swift',
  'ModelRecommendationEngineTest.swift',
  'PerformanceProfilerTest.swift',
  'CompatibilityAnalyzerTest.swift',
  'ModelIndexerTest.swift',
  'DiscoverySchedulerTest.swift',
  'ModelValidatorTest.swift',
  'ModelDiscoveryDashboardTest.swift',
  'ModelBrowserViewTest.swift',
  'RecommendationViewTest.swift',
  'DiscoverySettingsViewTest.swift'
]

# Function to add files to target
def add_files_to_group_and_target(group, files, subfolder, target)
  files.each do |filename|
    file_path = "RealtimeModelDiscovery/#{subfolder}/#{filename}"
    
    # Check if file already exists in group
    existing_file = group.files.find { |f| f.path == filename }
    if existing_file.nil?
      # Add file reference to group
      file_ref = group.new_file(file_path)
      
      # Add to target's sources build phase if it's a Swift file
      if filename.end_with?('.swift')
        target.source_build_phase.add_file_reference(file_ref)
      end
      
      puts "✅ Added #{filename} to #{subfolder}"
    else
      puts "⚠️  #{filename} already exists in #{subfolder}"
    end
  end
end

# Add core files
add_files_to_group_and_target(core_group, core_files, 'Core', target)

# Add view files
add_files_to_group_and_target(views_group, view_files, 'Views', target)

# Add test files (these typically go to test target, but we'll add to main for now)
add_files_to_group_and_target(tests_group, test_files, 'Tests/RealtimeModelDiscoveryTests', target)

# Save the project
project.save

puts "\n🎉 Successfully added RealtimeModelDiscovery files to Xcode project!"
puts "📁 Core files: #{core_files.count}"
puts "📱 View files: #{view_files.count}" 
puts "🧪 Test files: #{test_files.count}"
puts "💾 Project saved successfully"