#!/usr/bin/env ruby

require 'xcodeproj'
require 'pathname'

# Model Performance Benchmarking Xcode Integration Script
puts "🔧 Adding Model Performance Benchmarking to Xcode Project"

project_path = Pathname.new(File.expand_path('.'))
xcodeproj_path = project_path / 'Sandbox-AgenticSeek.xcodeproj'

if !xcodeproj_path.exist?
  puts "❌ Error: Xcode project not found at #{xcodeproj_path}"
  exit 1
end

project = Xcodeproj::Project.open(xcodeproj_path)
target = project.targets.find { |t| t.name == 'AgenticSeek' }

if target.nil?
  puts "❌ Error: AgenticSeek target not found"
  exit 1
end

# Create ModelPerformanceBenchmarking group structure
main_group = project.main_group.find_subpath('AgenticSeek', true)
benchmarking_group = main_group.new_group('ModelPerformanceBenchmarking')
core_group = benchmarking_group.new_group('Core')
views_group = benchmarking_group.new_group('Views')

# Create test group
tests_group = project.main_group.find_subpath('Tests', true)
benchmarking_tests_group = tests_group.new_group('ModelPerformanceBenchmarkingTests')

# Core component files to add
core_files = [
  'ModelBenchmarkEngine.swift',
  'InferenceSpeedAnalyzer.swift',
  'QualityAssessmentEngine.swift',
  'ResourceMonitor.swift',
  'BenchmarkScheduler.swift',
  'BenchmarkDataManager.swift',
  'ModelComparator.swift',
  'PerformanceMetricsCalculator.swift',
  'BenchmarkModels.swift',
  'BenchmarkExtensions.swift',
  'BenchmarkUtilities.swift'
]

# View component files to add
view_files = [
  'BenchmarkDashboardView.swift',
  'BenchmarkConfigurationView.swift',
  'PerformanceVisualizationView.swift',
  'ModelComparisonView.swift'
]

# Test files to add
test_files = [
  'ModelBenchmarkEngineTest.swift',
  'InferenceSpeedAnalyzerTest.swift',
  'QualityAssessmentEngineTest.swift',
  'ResourceMonitorTest.swift',
  'BenchmarkSchedulerTest.swift',
  'BenchmarkDataManagerTest.swift',
  'ModelComparatorTest.swift',
  'PerformanceMetricsCalculatorTest.swift',
  'BenchmarkDashboardViewTest.swift',
  'BenchmarkConfigurationViewTest.swift',
  'PerformanceVisualizationViewTest.swift',
  'ModelComparisonViewTest.swift'
]

# Add core files
puts "📁 Adding Core components..."
core_files.each do |filename|
  file_path = "AgenticSeek/ModelPerformanceBenchmarking/Core/#{filename}"
  
  if File.exist?(file_path)
    file_ref = core_group.new_file(file_path)
    target.add_file_references([file_ref])
    puts "  ✅ Added #{filename}"
  else
    puts "  ⚠️  File not found: #{file_path}"
  end
end

# Add view files
puts "📁 Adding View components..."
view_files.each do |filename|
  file_path = "AgenticSeek/ModelPerformanceBenchmarking/Views/#{filename}"
  
  if File.exist?(file_path)
    file_ref = views_group.new_file(file_path)
    target.add_file_references([file_ref])
    puts "  ✅ Added #{filename}"
  else
    puts "  ⚠️  File not found: #{file_path}"
  end
end

# Add test files
puts "📁 Adding Test files..."
test_files.each do |filename|
  file_path = "AgenticSeek/Tests/ModelPerformanceBenchmarkingTests/#{filename}"
  
  if File.exist?(file_path)
    file_ref = benchmarking_tests_group.new_file(file_path)
    # Note: Test files typically don't need to be added to main target
    puts "  ✅ Added #{filename}"
  else
    puts "  ⚠️  File not found: #{file_path}"
  end
end

# Save the project
project.save

puts "\n🎯 Model Performance Benchmarking integration complete!"
puts "📊 Added #{core_files.length} core files"
puts "🖼️  Added #{view_files.length} view files"
puts "🧪 Added #{test_files.length} test files"
puts "\n✅ Xcode project updated successfully"