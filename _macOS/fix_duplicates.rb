#!/usr/bin/env ruby
require 'xcodeproj'

# Open the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first
puts "📋 Target: #{target.name}"

# Find and remove OnboardingFlow.swift from target (it has duplicate OnboardingManager class)
onboarding_flow_files = target.source_build_phase.files.select { |f| 
  f.file_ref&.path&.include?('OnboardingFlow.swift') 
}

puts "\n🔍 Found OnboardingFlow.swift entries: #{onboarding_flow_files.length}"
onboarding_flow_files.each do |build_file|
  puts "  ❌ Removing from target: #{build_file.file_ref.path}"
  target.source_build_phase.remove_file_reference(build_file.file_ref)
end

# Verify OnboardingManager.swift is in target
onboarding_manager_files = target.source_build_phase.files.select { |f| 
  f.file_ref&.path&.include?('OnboardingManager.swift') 
}

puts "\n✅ OnboardingManager.swift entries: #{onboarding_manager_files.length}"
onboarding_manager_files.each do |build_file|
  puts "  ✅ Keeping in target: #{build_file.file_ref.path}"
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Duplicate OnboardingManager class resolved - OnboardingFlow.swift removed from target"