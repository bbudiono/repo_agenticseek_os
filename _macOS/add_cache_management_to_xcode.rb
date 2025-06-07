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

puts "🗂️ ADDING LOCAL MODEL CACHE MANAGEMENT TO XCODE PROJECT"
puts "======================================================="

# Find or create the LocalModelCacheManagement group
cache_group = project.main_group.groups.find { |group| group.name == 'LocalModelCacheManagement' }
if cache_group.nil?
  cache_group = project.main_group.new_group('LocalModelCacheManagement')
  puts "✅ Created LocalModelCacheManagement group"
end

# Create subgroups
core_group = cache_group.groups.find { |group| group.name == 'Core' }
if core_group.nil?
  core_group = cache_group.new_group('Core')
  puts "✅ Created LocalModelCacheManagement/Core group"
end

views_group = cache_group.groups.find { |group| group.name == 'Views' }
if views_group.nil?
  views_group = cache_group.new_group('Views')
  puts "✅ Created LocalModelCacheManagement/Views group"
end

integration_group = cache_group.groups.find { |group| group.name == 'Integration' }
if integration_group.nil?
  integration_group = cache_group.new_group('Integration')
  puts "✅ Created LocalModelCacheManagement/Integration group"
end

models_group = cache_group.groups.find { |group| group.name == 'Models' }
if models_group.nil?
  models_group = cache_group.new_group('Models')
  puts "✅ Created LocalModelCacheManagement/Models group"
end

tests_group = cache_group.groups.find { |group| group.name == 'Tests' }
if tests_group.nil?
  tests_group = cache_group.new_group('Tests')
  puts "✅ Created LocalModelCacheManagement/Tests group"
end

puts "\n🔄 ADDING CACHE MANAGEMENT FILES"
puts "================================"

# Core files
core_files = [
  'ModelWeightCacheManager.swift',
  'IntermediateActivationCache.swift',
  'ComputationResultCache.swift',
  'CacheEvictionEngine.swift',
  'CrossModelSharedParameterDetector.swift',
  'CacheCompressionEngine.swift',
  'CacheWarmingSystem.swift',
  'CachePerformanceAnalytics.swift',
  'CacheStorageOptimizer.swift',
  'CacheSecurityManager.swift'
]

core_files.each do |filename|
  file_path = "LocalModelCacheManagement/Core/#{filename}"
  file_ref = core_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to LocalModelCacheManagement/Core"
end

# Views files
views_files = [
  'CacheManagementDashboard.swift',
  'CacheConfigurationView.swift',
  'CacheAnalyticsView.swift'
]

views_files.each do |filename|
  file_path = "LocalModelCacheManagement/Views/#{filename}"
  file_ref = views_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to LocalModelCacheManagement/Views"
end

# Integration files
integration_files = [
  'MLACSCacheIntegration.swift'
]

integration_files.each do |filename|
  file_path = "LocalModelCacheManagement/Integration/#{filename}"
  file_ref = integration_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to LocalModelCacheManagement/Integration"
end

# Models files
models_files = [
  'CacheModels.swift'
]

models_files.each do |filename|
  file_path = "LocalModelCacheManagement/Models/#{filename}"
  file_ref = models_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to LocalModelCacheManagement/Models"
end

# Test files
test_files = [
  'ModelWeightCacheManagerTest.swift',
  'IntermediateActivationCacheTest.swift',
  'ComputationResultCacheTest.swift',
  'CacheEvictionEngineTest.swift',
  'CrossModelSharedParameterDetectorTest.swift',
  'CacheCompressionEngineTest.swift',
  'CacheWarmingSystemTest.swift',
  'CachePerformanceAnalyticsTest.swift',
  'CacheStorageOptimizerTest.swift',
  'CacheSecurityManagerTest.swift',
  'CacheManagementDashboardTest.swift',
  'CacheConfigurationViewTest.swift',
  'CacheAnalyticsViewTest.swift',
  'MLACSCacheIntegrationTest.swift',
  'CacheModelsTest.swift'
]

test_files.each do |filename|
  file_path = "LocalModelCacheManagement/Tests/#{filename}"
  file_ref = tests_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to LocalModelCacheManagement/Tests"
end

# Save the project
project.save

puts "\n🎉 SUCCESSFULLY ADDED LOCAL MODEL CACHE MANAGEMENT TO XCODE!"
puts "📁 Total files added: #{core_files.length + views_files.length + integration_files.length + models_files.length + test_files.length}"
puts "💾 Project saved successfully"
puts "🔧 Build should now include cache management components"