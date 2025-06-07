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

puts "🔧 REMOVING TEST FILES FROM BUILD TARGET"
puts "======================================="
puts "🎯 FIXING BUILD BY REMOVING MISSING TEST FILES"
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
  
  # Check if this is a test file that doesn't exist
  if path.include?('/Tests/') && path.end_with?('Test.swift')
    full_path = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/#{path}"
    
    unless File.exist?(full_path)
      test_file_refs << file_ref
      puts "❌ Missing test file: #{path}"
    end
  end
end

puts ""
puts "📊 ANALYSIS COMPLETE"
puts "🔍 Total files checked: #{total_checked}"
puts "❌ Missing test files found: #{test_file_refs.length}"
puts ""

if test_file_refs.length > 0
  puts "🗑️ REMOVING MISSING TEST FILES FROM BUILD"
  puts "=========================================="
  
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
    
    # Remove file reference from project entirely
    file_ref.remove_from_project
  end
  
  # Save the project
  project.save
  
  puts ""
  puts "🎉 BUILD CLEANUP COMPLETE!"
  puts "=========================="
  puts "💾 Project saved successfully"
  puts "🗑️ Removed #{removed_count} missing test files from build"
  puts "🚀 Build should now work correctly"
  puts ""
  puts "📋 NEXT STEPS:"
  puts "1. Test the build: xcodebuild -project AgenticSeek.xcodeproj -target AgenticSeek build"
  puts "2. Verify TestFlight deployment readiness"
  puts "3. Push to GitHub after successful build"
else
  puts "ℹ️ No missing test files found"
  puts "All referenced test files exist in the project"
end