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

# Remove old ModelCompatibilityAnalyzer file references from IntelligentModelRecommendations
old_file_refs = []
project.files.each do |file_ref|
  if file_ref.path&.include?('IntelligentModelRecommendations') && file_ref.path&.include?('ModelCompatibilityAnalyzer')
    old_file_refs << file_ref
  end
end

old_file_refs.each do |file_ref|
  # Remove from target's sources build phase
  target.source_build_phase.files.each do |build_file|
    if build_file.file_ref == file_ref
      target.source_build_phase.remove_file_reference(file_ref)
      puts "✅ Removed old #{file_ref.path} from target"
      break
    end
  end
  
  # Remove file reference from project
  file_ref.remove_from_project
  puts "✅ Removed old #{file_ref.path} file reference"
end

# Find IntelligentModelRecommendations group
intelligent_recommendations_group = project.main_group.groups.find { |group| group.name == 'IntelligentModelRecommendations' }
if intelligent_recommendations_group.nil?
  puts "❌ IntelligentModelRecommendations group not found"
  exit 1
end

# Find Core subgroup
core_group = intelligent_recommendations_group.groups.find { |group| group.name == 'Core' }
if core_group.nil?
  puts "❌ Core subgroup not found"
  exit 1
end

# Find Tests subgroup
tests_group = intelligent_recommendations_group.groups.find { |group| group.name == 'Tests' }
if tests_group.nil?
  puts "❌ Tests subgroup not found"
  exit 1
end

# Add the new IntelligentModelCompatibilityAnalyzer.swift file
intelligent_analyzer_path = 'IntelligentModelRecommendations/Core/IntelligentModelCompatibilityAnalyzer.swift'
intelligent_analyzer_ref = core_group.new_file(intelligent_analyzer_path)
target.source_build_phase.add_file_reference(intelligent_analyzer_ref)
puts "✅ Added IntelligentModelCompatibilityAnalyzer.swift to Core group"

# Add the new IntelligentModelCompatibilityAnalyzerTest.swift file
intelligent_analyzer_test_path = 'IntelligentModelRecommendations/Tests/IntelligentModelCompatibilityAnalyzerTest.swift'
intelligent_analyzer_test_ref = tests_group.new_file(intelligent_analyzer_test_path)
target.source_build_phase.add_file_reference(intelligent_analyzer_test_ref)
puts "✅ Added IntelligentModelCompatibilityAnalyzerTest.swift to Tests group"

# Save the project
project.save

puts "\n🎉 Successfully updated Xcode project with renamed files!"
puts "💾 Project saved successfully"