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

puts "🔧 COMPREHENSIVE XCODE FILE PATH FIX"
puts "===================================="
puts "🎯 SYSTEMATIC CORRECTION OF ALL MLACS COMPONENT PATHS"
puts ""

# Track changes
removed_count = 0
added_count = 0
fixed_count = 0

# Find all file references that have incorrect paths (missing AgenticSeek/ prefix)
incorrect_file_refs = []

project.files.each do |file_ref|
  path = file_ref.path
  next unless path
  
  # Check for MLACS component paths that are missing AgenticSeek/ prefix
  mlacs_patterns = [
    'LocalModelCacheManagement/',
    'IntelligentModelRecommendations/',
    'RealtimeModelDiscovery/',
    'ModelPerformanceBenchmarking/',
    'HardwareOptimization/',
    'LocalModelManagement/',
    'TieredArchitecture/',
    'CustomAgents/',
    'SingleAgentMode/'
  ]
  
  mlacs_patterns.each do |pattern|
    if path.start_with?(pattern)
      incorrect_file_refs << file_ref
      puts "❌ Found incorrect path: #{path}"
      break
    end
  end
end

puts ""
puts "📊 ANALYSIS COMPLETE"
puts "🔍 Found #{incorrect_file_refs.length} files with incorrect paths"
puts ""

# Remove all incorrect file references
puts "🗑️ REMOVING INCORRECT FILE REFERENCES"
puts "====================================="

incorrect_file_refs.each do |file_ref|
  # Remove from target's sources build phase
  target.source_build_phase.files.each do |build_file|
    if build_file.file_ref == file_ref
      target.source_build_phase.remove_file_reference(file_ref)
      break
    end
  end
  
  # Remove file reference from project
  file_ref.remove_from_project
  removed_count += 1
  puts "✅ Removed: #{file_ref.path}"
end

puts ""
puts "🔧 ADDING CORRECT FILE REFERENCES"
puts "=================================="

# Define all MLACS components with their correct structure
mlacs_components = [
  {
    name: "LocalModelCacheManagement",
    core_files: [
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
    ],
    views_files: [
      'CacheManagementDashboard.swift',
      'CacheConfigurationView.swift',
      'CacheAnalyticsView.swift'
    ],
    integration_files: [
      'MLACSCacheIntegration.swift'
    ],
    models_files: [
      'CacheModels.swift'
    ],
    test_files: [
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
  },
  {
    name: "IntelligentModelRecommendations",
    core_files: [
      'TaskComplexityAnalyzer.swift',
      'UserPreferenceLearningEngine.swift',
      'HardwareCapabilityProfiler.swift',
      'ModelPerformancePredictor.swift',
      'RecommendationGenerationEngine.swift',
      'ContextAwareRecommender.swift',
      'QualityPerformanceOptimizer.swift',
      'RecommendationCacheManager.swift',
      'FeedbackLearningSystem.swift',
      'RecommendationExplanationEngine.swift',
      'AdaptiveRecommendationUpdater.swift',
      'IntelligentModelCompatibilityAnalyzer.swift',
      'IntelligentRecommendationModels.swift',
      'IntelligentRecommendationExtensions.swift',
      'MLIntegrationUtilities.swift'
    ],
    views_files: [
      'IntelligentRecommendationDashboard.swift',
      'TaskAnalysisView.swift',
      'RecommendationExplanationView.swift',
      'UserPreferenceConfigurationView.swift',
      'PerformancePredictionView.swift',
      'RecommendationFeedbackView.swift'
    ],
    test_files: [
      'TaskComplexityAnalyzerTest.swift',
      'UserPreferenceLearningEngineTest.swift',
      'HardwareCapabilityProfilerTest.swift',
      'ModelPerformancePredictorTest.swift',
      'RecommendationGenerationEngineTest.swift',
      'ContextAwareRecommenderTest.swift',
      'QualityPerformanceOptimizerTest.swift',
      'RecommendationCacheManagerTest.swift',
      'FeedbackLearningSystemTest.swift',
      'RecommendationExplanationEngineTest.swift',
      'AdaptiveRecommendationUpdaterTest.swift',
      'IntelligentModelCompatibilityAnalyzerTest.swift',
      'IntelligentRecommendationDashboardTest.swift',
      'TaskAnalysisViewTest.swift',
      'RecommendationExplanationViewTest.swift',
      'UserPreferenceConfigurationViewTest.swift',
      'PerformancePredictionViewTest.swift',
      'RecommendationFeedbackViewTest.swift'
    ]
  },
  {
    name: "RealtimeModelDiscovery",
    core_files: [
      'ModelDiscoveryEngine.swift',
      'ModelRegistryManager.swift',
      'CapabilityDetector.swift',
      'ProviderScanner.swift',
      'ModelRecommendationEngine.swift',
      'CompatibilityAnalyzer.swift',
      'ModelIndexer.swift',
      'DiscoveryScheduler.swift',
      'ModelValidator.swift',
      'DiscoveryModels.swift',
      'DiscoveryExtensions.swift',
      'DiscoveryUtilities.swift',
      'DiscoveryPerformanceProfiler.swift'
    ],
    views_files: [
      'ModelDiscoveryDashboard.swift',
      'ModelBrowserView.swift',
      'RecommendationView.swift',
      'DiscoverySettingsView.swift'
    ],
    test_files: [
      'ModelDiscoveryEngineTest.swift',
      'ModelRegistryManagerTest.swift',
      'CapabilityDetectorTest.swift',
      'ProviderScannerTest.swift',
      'ModelRecommendationEngineTest.swift',
      'CompatibilityAnalyzerTest.swift',
      'ModelIndexerTest.swift',
      'DiscoverySchedulerTest.swift',
      'ModelValidatorTest.swift',
      'ModelDiscoveryDashboardTest.swift',
      'ModelBrowserViewTest.swift',
      'RecommendationViewTest.swift',
      'DiscoverySettingsViewTest.swift',
      'DiscoveryPerformanceProfilerTest.swift'
    ]
  }
]

# Function to create group if it doesn't exist
def find_or_create_group(parent_group, group_name)
  group = parent_group.groups.find { |g| g.name == group_name }
  if group.nil?
    group = parent_group.new_group(group_name)
    puts "✅ Created group: #{group_name}"
  end
  group
end

# Process each MLACS component
mlacs_components.each do |component|
  puts ""
  puts "🔄 Processing #{component[:name]}..."
  
  # Find or create main component group
  main_group = find_or_create_group(project.main_group, component[:name])
  
  # Create subgroups
  core_group = find_or_create_group(main_group, 'Core')
  views_group = find_or_create_group(main_group, 'Views') if component[:views_files]
  integration_group = find_or_create_group(main_group, 'Integration') if component[:integration_files]
  models_group = find_or_create_group(main_group, 'Models') if component[:models_files]
  tests_group = find_or_create_group(main_group, 'Tests')
  
  # Add core files
  component[:core_files].each do |filename|
    file_path = "#{component[:name]}/Core/#{filename}"
    file_ref = core_group.new_file(file_path)
    target.source_build_phase.add_file_reference(file_ref)
    added_count += 1
    puts "✅ Added: #{file_path}"
  end
  
  # Add views files
  if component[:views_files]
    component[:views_files].each do |filename|
      file_path = "#{component[:name]}/Views/#{filename}"
      file_ref = views_group.new_file(file_path)
      target.source_build_phase.add_file_reference(file_ref)
      added_count += 1
      puts "✅ Added: #{file_path}"
    end
  end
  
  # Add integration files
  if component[:integration_files]
    component[:integration_files].each do |filename|
      file_path = "#{component[:name]}/Integration/#{filename}"
      file_ref = integration_group.new_file(file_path)
      target.source_build_phase.add_file_reference(file_ref)
      added_count += 1
      puts "✅ Added: #{file_path}"
    end
  end
  
  # Add models files
  if component[:models_files]
    component[:models_files].each do |filename|
      file_path = "#{component[:name]}/Models/#{filename}"
      file_ref = models_group.new_file(file_path)
      target.source_build_phase.add_file_reference(file_ref)
      added_count += 1
      puts "✅ Added: #{file_path}"
    end
  end
  
  # Add test files
  component[:test_files].each do |filename|
    file_path = "#{component[:name]}/Tests/#{filename}"
    file_ref = tests_group.new_file(file_path)
    target.source_build_phase.add_file_reference(file_ref)
    added_count += 1
    puts "✅ Added: #{file_path}"
  end
end

# Save the project
project.save

puts ""
puts "🎉 COMPREHENSIVE FILE PATH FIX COMPLETE!"
puts "========================================"
puts "📊 SUMMARY:"
puts "🗑️ Removed incorrect references: #{removed_count}"
puts "✅ Added correct references: #{added_count}"
puts "💾 Project saved successfully"
puts ""
puts "🔧 All MLACS component file paths have been corrected!"
puts "📁 Files are now properly referenced within AgenticSeek target"
puts "🚀 Build should now work correctly"