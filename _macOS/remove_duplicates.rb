#!/usr/bin/env ruby
require 'xcodeproj'

# Open the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first
puts "📋 Target: #{target.name}"

# Get all build files
build_files = target.source_build_phase.files

# Find duplicate VoiceAI files
voiceai_core_files = build_files.select { |f| f.file_ref&.path&.include?('VoiceAICore.swift') }
voiceai_bridge_files = build_files.select { |f| f.file_ref&.path&.include?('VoiceAIBridge.swift') }

puts "\n🔍 Found VoiceAICore.swift entries: #{voiceai_core_files.length}"
voiceai_core_files.each_with_index do |f, i|
  puts "  #{i+1}. #{f.file_ref.path} (ref: #{f.file_ref.uuid})"
end

puts "\n🔍 Found VoiceAIBridge.swift entries: #{voiceai_bridge_files.length}"
voiceai_bridge_files.each_with_index do |f, i|
  puts "  #{i+1}. #{f.file_ref.path} (ref: #{f.file_ref.uuid})"
end

# Keep only the Core directory versions (they have SOURCE_ROOT source tree)
if voiceai_core_files.length > 1
  puts "\n🧹 Removing duplicate VoiceAICore.swift files..."
  voiceai_core_files.each do |build_file|
    # Keep the one with SOURCE_ROOT source tree (Core directory version)
    if build_file.file_ref.source_tree != 'SOURCE_ROOT'
      puts "  ❌ Removing: #{build_file.file_ref.path}"
      target.source_build_phase.remove_file_reference(build_file.file_ref)
    else
      puts "  ✅ Keeping: #{build_file.file_ref.path}"
    end
  end
end

if voiceai_bridge_files.length > 1
  puts "\n🧹 Removing duplicate VoiceAIBridge.swift files..."
  voiceai_bridge_files.each do |build_file|
    # Keep the one with SOURCE_ROOT source tree (Core directory version)
    if build_file.file_ref.source_tree != 'SOURCE_ROOT'
      puts "  ❌ Removing: #{build_file.file_ref.path}"
      target.source_build_phase.remove_file_reference(build_file.file_ref)
    else
      puts "  ✅ Keeping: #{build_file.file_ref.path}"
    end
  end
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Duplicates removed - build should work now"