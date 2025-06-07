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

puts "🔧 REMOVING ALL TEST FILES FROM MAIN BUILD TARGET"
puts "================================================="
puts "🎯 ENSURING MAIN APP BUILDS WITHOUT TEST DEPENDENCIES"
puts ""

# Track changes
removed_count = 0
total_checked = 0

# Find all file references that are test files
test_file_refs = []

project.files.each do |file_ref|
  path = file_ref.path
  next unless path
  
  total_checked += 1
  
  # Check if this is any test file (including existing ones)
  if (path.include?('/Tests/') && path.end_with?('Test.swift')) || path.end_with?('Tests.swift')
    test_file_refs << file_ref
    puts "🧪 Found test file: #{path}"
  end
end

puts ""
puts "📊 ANALYSIS COMPLETE"
puts "🔍 Total files checked: #{total_checked}"
puts "🧪 Test files found: #{test_file_refs.length}"
puts ""

if test_file_refs.length > 0
  puts "🗑️ REMOVING ALL TEST FILES FROM MAIN BUILD TARGET"
  puts "=================================================="
  
  test_file_refs.each do |file_ref|
    # Remove from target's sources build phase
    target.source_build_phase.files.each do |build_file|
      if build_file.file_ref == file_ref
        target.source_build_phase.remove_file_reference(file_ref)
        removed_count += 1
        puts "✅ Removed from build: #{file_ref.path}"
        break
      end
    end
  end
  
  # Save the project
  project.save
  
  puts ""
  puts "🎉 COMPLETE TEST FILE CLEANUP SUCCESSFUL!"
  puts "========================================="
  puts "💾 Project saved successfully"
  puts "🗑️ Removed #{removed_count} test files from main app build"
  puts "🚀 Main app should now build without test dependencies"
  puts ""
  puts "📋 NEXT STEPS:"
  puts "1. Test the build: xcodebuild -project AgenticSeek.xcodeproj -target AgenticSeek build"
  puts "2. Verify TestFlight deployment readiness"
  puts "3. Push to GitHub after successful build"
  puts ""
  puts "ℹ️ NOTE: Test files remain in project for future test target creation"
  puts "ℹ️ They are just not compiled as part of the main application"
else
  puts "ℹ️ No test files found in main build target"
  puts "Main application should already build without test dependencies"
end