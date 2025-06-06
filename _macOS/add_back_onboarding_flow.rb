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

# Find OnboardingFlow.swift file reference (it should exist but not be in target)
onboarding_flow_ref = main_group.files.find { |f| f.path == 'OnboardingFlow.swift' }

if onboarding_flow_ref
  puts "✅ Found OnboardingFlow.swift file reference"
  
  # Check if it's already in target
  existing_build_files = target.source_build_phase.files.select { |f| 
    f.file_ref&.path&.include?('OnboardingFlow.swift') 
  }
  
  if existing_build_files.empty?
    # Add it back to target
    build_file = target.add_file_references([onboarding_flow_ref]).first
    puts "✅ Added OnboardingFlow.swift back to target"
  else
    puts "⚠️  OnboardingFlow.swift already in target"
  end
else
  # File reference doesn't exist, create it
  onboarding_flow_ref = main_group.new_reference('OnboardingFlow.swift')
  target.add_file_references([onboarding_flow_ref])
  puts "✅ Created and added OnboardingFlow.swift file reference"
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 OnboardingFlow.swift added back to target"