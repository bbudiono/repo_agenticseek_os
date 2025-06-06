#!/usr/bin/env ruby
require 'xcodeproj'

# Open the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first
puts "📋 Target: #{target.name}"

# Remove RealChatInterface.swift which has duplicate ChatMessage struct
build_files = target.source_build_phase.files.select { |f| 
  f.file_ref&.path&.include?('RealChatInterface.swift') 
}

build_files.each do |build_file|
  puts "❌ Removing from target: #{build_file.file_ref.path}"
  target.source_build_phase.remove_file_reference(build_file.file_ref)
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Removed RealChatInterface.swift to fix ChatMessage duplicate"