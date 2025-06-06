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

# Add ProductionComponents.swift back
filename = 'ProductionComponents.swift'
file_ref = main_group.files.find { |f| f.path == filename }

if file_ref
  # Add to target if not already there
  existing_build_files = target.source_build_phase.files.select { |f| 
    f.file_ref&.path&.include?(filename) 
  }
  
  if existing_build_files.empty?
    target.add_file_references([file_ref])
    puts "✅ Added: #{filename}"
  else
    puts "✅ Already in target: #{filename}"
  end
else
  puts "❌ File not found: #{filename}"
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Added ProductionComponents.swift for ContentView UI"