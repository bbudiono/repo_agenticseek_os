#!/usr/bin/env ruby
require 'xcodeproj'

# Open the Xcode project
project_path = 'AgenticSeek.xcodeproj'
project = Xcodeproj::Project.open(project_path)

# Get the main target
target = project.targets.first
puts "📋 Target: #{target.name}"

# Remove problematic files that have conflicts to get a minimal working build
problematic_files = [
  'PerformanceOptimizedComponents.swift',
  'SpeculativeDecodingCoordinator.swift',
  'SpeculativeDecodingEngine.swift',
  'RejectionSamplingEngine.swift',
  'EnhancedContentView.swift',
  'OptimizedModelManagementView.swift',
  'SandboxComponents.swift'
]

puts "\n🧹 Removing problematic files from target to achieve minimal working build..."

problematic_files.each do |filename|
  build_files = target.source_build_phase.files.select { |f| 
    f.file_ref&.path&.include?(filename) 
  }
  
  build_files.each do |build_file|
    puts "  ❌ Removing from target: #{build_file.file_ref.path}"
    target.source_build_phase.remove_file_reference(build_file.file_ref)
  end
end

# Save the project
project.save
puts "\n💾 Project saved successfully!"
puts "🎯 Minimal build configuration - removed conflicting files"
puts "📋 Remaining files should compile cleanly"