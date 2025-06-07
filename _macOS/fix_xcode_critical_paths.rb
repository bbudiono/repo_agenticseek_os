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

puts "🔧 FIXING ALL XCODE FILE PATH REFERENCES"
puts "========================================="

# Remove all files with incorrect paths
files_to_remove = []
project.files.each do |file_ref|
  if file_ref.path&.start_with?('RealtimeModelDiscovery/') || 
     file_ref.path&.start_with?('IntelligentModelRecommendations/')
    files_to_remove << file_ref
    puts "❌ Found incorrect path: #{file_ref.path}"
  end
end

# Remove from target's sources build phase and from project
files_to_remove.each do |file_ref|
  # Remove from target's sources build phase
  target.source_build_phase.files.each do |build_file|
    if build_file.file_ref == file_ref
      target.source_build_phase.remove_file_reference(file_ref)
      puts "✅ Removed #{file_ref.path} from target"
      break
    end
  end
  
  # Remove file reference from project
  file_ref.remove_from_project
  puts "✅ Removed #{file_ref.path} file reference"
end

# Find or create the RealtimeModelDiscovery group
realtime_discovery_group = project.main_group.groups.find { |group| group.name == 'RealtimeModelDiscovery' }
if realtime_discovery_group.nil?
  realtime_discovery_group = project.main_group.new_group('RealtimeModelDiscovery')
  puts "✅ Created RealtimeModelDiscovery group"
end

# Find or create the IntelligentModelRecommendations group
intelligent_recommendations_group = project.main_group.groups.find { |group| group.name == 'IntelligentModelRecommendations' }
if intelligent_recommendations_group.nil?
  intelligent_recommendations_group = project.main_group.new_group('IntelligentModelRecommendations')
  puts "✅ Created IntelligentModelRecommendations group"
end

# Create subgroups for RealtimeModelDiscovery
rt_core_group = realtime_discovery_group.groups.find { |group| group.name == 'Core' }
if rt_core_group.nil?
  rt_core_group = realtime_discovery_group.new_group('Core')
  puts "✅ Created RealtimeModelDiscovery/Core group"
end

rt_views_group = realtime_discovery_group.groups.find { |group| group.name == 'Views' }
if rt_views_group.nil?
  rt_views_group = realtime_discovery_group.new_group('Views')
  puts "✅ Created RealtimeModelDiscovery/Views group"
end

rt_tests_group = realtime_discovery_group.groups.find { |group| group.name == 'Tests' }
if rt_tests_group.nil?
  rt_tests_group = realtime_discovery_group.new_group('Tests')
  puts "✅ Created RealtimeModelDiscovery/Tests group"
end

# Create subgroups for IntelligentModelRecommendations
ir_core_group = intelligent_recommendations_group.groups.find { |group| group.name == 'Core' }
if ir_core_group.nil?
  ir_core_group = intelligent_recommendations_group.new_group('Core')
  puts "✅ Created IntelligentModelRecommendations/Core group"
end

ir_views_group = intelligent_recommendations_group.groups.find { |group| group.name == 'Views' }
if ir_views_group.nil?
  ir_views_group = intelligent_recommendations_group.new_group('Views')
  puts "✅ Created IntelligentModelRecommendations/Views group"
end

ir_tests_group = intelligent_recommendations_group.groups.find { |group| group.name == 'Tests' }
if ir_tests_group.nil?
  ir_tests_group = intelligent_recommendations_group.new_group('Tests')
  puts "✅ Created IntelligentModelRecommendations/Tests group"
end

puts "\n🔄 ADDING FILES WITH CORRECT PATHS"
puts "=================================="

# RealtimeModelDiscovery Core files
rt_core_files = [
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
]

rt_core_files.each do |filename|
  file_path = "RealtimeModelDiscovery/Core/#{filename}"
  file_ref = rt_core_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to RealtimeModelDiscovery/Core"
end

# RealtimeModelDiscovery Views files
rt_views_files = [
  'ModelDiscoveryDashboard.swift',
  'ModelBrowserView.swift',
  'RecommendationView.swift',
  'DiscoverySettingsView.swift'
]

rt_views_files.each do |filename|
  file_path = "RealtimeModelDiscovery/Views/#{filename}"
  file_ref = rt_views_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to RealtimeModelDiscovery/Views"
end

# RealtimeModelDiscovery Test files
rt_test_files = [
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

rt_test_files.each do |filename|
  file_path = "RealtimeModelDiscovery/Tests/#{filename}"
  file_ref = rt_tests_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to RealtimeModelDiscovery/Tests"
end

# IntelligentModelRecommendations Core files
ir_core_files = [
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
]

ir_core_files.each do |filename|
  file_path = "IntelligentModelRecommendations/Core/#{filename}"
  file_ref = ir_core_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to IntelligentModelRecommendations/Core"
end

# IntelligentModelRecommendations Views files
ir_views_files = [
  'IntelligentRecommendationDashboard.swift',
  'TaskAnalysisView.swift',
  'RecommendationExplanationView.swift',
  'UserPreferenceConfigurationView.swift',
  'PerformancePredictionView.swift',
  'RecommendationFeedbackView.swift'
]

ir_views_files.each do |filename|
  file_path = "IntelligentModelRecommendations/Views/#{filename}"
  file_ref = ir_views_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to IntelligentModelRecommendations/Views"
end

# IntelligentModelRecommendations Test files
ir_test_files = [
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

ir_test_files.each do |filename|
  file_path = "IntelligentModelRecommendations/Tests/#{filename}"
  file_ref = ir_tests_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to IntelligentModelRecommendations/Tests"
end

# Save the project
project.save

puts "\n🎉 SUCCESSFULLY FIXED ALL XCODE FILE PATH REFERENCES!"
puts "📁 Total files added: #{rt_core_files.length + rt_views_files.length + rt_test_files.length + ir_core_files.length + ir_views_files.length + ir_test_files.length}"
puts "💾 Project saved successfully"
puts "🔧 Build should now work correctly"