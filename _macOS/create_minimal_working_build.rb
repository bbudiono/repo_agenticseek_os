#!/usr/bin/env ruby
require 'xcodeproj'

# Open the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first
puts "📋 Target: #{target.name}"

# Keep only essential files for a minimal working SwiftUI app
essential_files = [
  'AgenticSeekApp.swift',
  'ContentView.swift', 
  'DesignSystem.swift',
  'Strings.swift',
  'OnboardingManager.swift',
  'OnboardingFlow.swift'
]

puts "\n🧹 Creating minimal working build - keeping only essential files..."

# Get all current build files
all_build_files = target.source_build_phase.files.dup

# Remove all files except essential ones
all_build_files.each do |build_file|
  filename = build_file.file_ref&.path&.split('/')&.last
  
  unless essential_files.include?(filename)
    puts "  ❌ Removing: #{build_file.file_ref.path}"
    target.source_build_phase.remove_file_reference(build_file.file_ref)
  else
    puts "  ✅ Keeping: #{build_file.file_ref.path}"
  end
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Minimal working build with #{essential_files.length} essential files only"