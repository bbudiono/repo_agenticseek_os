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

puts "🔧 FIXING XCODE PROJECT FILE REFERENCES"
puts "======================================"
puts "🎯 CORRECTING MLACS COMPONENT PATHS TO INCLUDE 'AgenticSeek/' PREFIX"
puts ""

# Track changes
fixed_count = 0
total_files = 0

# MLACS component patterns that need fixing
mlacs_patterns = [
  'RealtimeModelDiscovery/',
  'IntelligentModelRecommendations/',
  'LocalModelCacheManagement/'
]

puts "🔍 ANALYZING PROJECT FILE REFERENCES"
puts "===================================="

# Go through all file references in the project
project.files.each do |file_ref|
  path = file_ref.path
  next unless path
  
  total_files += 1
  
  # Check if this file reference needs to be fixed
  mlacs_patterns.each do |pattern|
    if path.start_with?(pattern)
      # This file reference needs the AgenticSeek/ prefix
      new_path = "AgenticSeek/#{path}"
      
      puts "🔄 Fixing: #{path} → #{new_path}"
      
      # Update the file reference path
      file_ref.path = new_path
      fixed_count += 1
      
      break
    end
  end
end

puts ""
puts "📊 ANALYSIS COMPLETE"
puts "🔍 Total files examined: #{total_files}"
puts "✅ File references fixed: #{fixed_count}"
puts ""

if fixed_count > 0
  # Save the project
  project.save
  
  puts "🎉 XCODE PROJECT REFERENCES FIXED!"
  puts "=================================="
  puts "💾 Project saved successfully"
  puts "🔧 Fixed #{fixed_count} file references to include AgenticSeek/ prefix"
  puts "🚀 Build should now work correctly"
  puts ""
  puts "📋 NEXT STEPS:"
  puts "1. Test the build: xcodebuild -project AgenticSeek.xcodeproj -target AgenticSeek build"
  puts "2. Verify TestFlight deployment readiness"
  puts "3. Push to GitHub after successful build"
else
  puts "ℹ️ No file references needed fixing"
  puts "All MLACS component paths already correctly reference AgenticSeek/ prefix"
end