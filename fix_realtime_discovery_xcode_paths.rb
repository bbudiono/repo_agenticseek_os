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

# Remove the old PerformanceProfiler.swift file reference
old_file_ref = nil
project.files.each do |file_ref|
  if file_ref.path == 'RealtimeModelDiscovery/Core/PerformanceProfiler.swift'
    old_file_ref = file_ref
    break
  end
end

if old_file_ref
  # Remove from target's sources build phase
  target.source_build_phase.files.each do |build_file|
    if build_file.file_ref == old_file_ref
      target.source_build_phase.remove_file_reference(old_file_ref)
      puts "✅ Removed old PerformanceProfiler.swift from target"
      break
    end
  end
  
  # Remove file reference from project
  old_file_ref.remove_from_project
  puts "✅ Removed old PerformanceProfiler.swift file reference"
end

# Remove the old PerformanceProfilerTest.swift file reference
old_test_file_ref = nil
project.files.each do |file_ref|
  if file_ref.path == 'RealtimeModelDiscovery/Tests/RealtimeModelDiscoveryTests/PerformanceProfilerTest.swift'
    old_test_file_ref = file_ref
    break
  end
end

if old_test_file_ref
  # Remove from target's sources build phase
  target.source_build_phase.files.each do |build_file|
    if build_file.file_ref == old_test_file_ref
      target.source_build_phase.remove_file_reference(old_test_file_ref)
      puts "✅ Removed old PerformanceProfilerTest.swift from target"
      break
    end
  end
  
  # Remove file reference from project
  old_test_file_ref.remove_from_project
  puts "✅ Removed old PerformanceProfilerTest.swift file reference"
end

# Find RealtimeModelDiscovery group
realtime_discovery_group = project.main_group.groups.find { |group| group.name == 'RealtimeModelDiscovery' }
if realtime_discovery_group.nil?
  puts "❌ RealtimeModelDiscovery group not found"
  exit 1
end

# Find Core subgroup
core_group = realtime_discovery_group.groups.find { |group| group.name == 'Core' }
if core_group.nil?
  puts "❌ Core subgroup not found"
  exit 1
end

# Find Tests subgroup
tests_group = realtime_discovery_group.groups.find { |group| group.name == 'Tests' }
if tests_group.nil?
  puts "❌ Tests subgroup not found"
  exit 1
end

# Add the new DiscoveryPerformanceProfiler.swift file
discovery_profiler_path = 'RealtimeModelDiscovery/Core/DiscoveryPerformanceProfiler.swift'
discovery_profiler_ref = core_group.new_file(discovery_profiler_path)
target.source_build_phase.add_file_reference(discovery_profiler_ref)
puts "✅ Added DiscoveryPerformanceProfiler.swift to Core group"

# Add the new DiscoveryPerformanceProfilerTest.swift file
discovery_profiler_test_path = 'RealtimeModelDiscovery/Tests/RealtimeModelDiscoveryTests/DiscoveryPerformanceProfilerTest.swift'
discovery_profiler_test_ref = tests_group.new_file(discovery_profiler_test_path)
target.source_build_phase.add_file_reference(discovery_profiler_test_ref)
puts "✅ Added DiscoveryPerformanceProfilerTest.swift to Tests group"

# Save the project
project.save

puts "\n🎉 Successfully updated Xcode project with renamed files!"
puts "💾 Project saved successfully"