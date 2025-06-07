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

# Create the IntelligentModelRecommendations group
intelligent_recommendations_group = project.main_group.new_group('IntelligentModelRecommendations')

# Create subgroups
core_group = intelligent_recommendations_group.new_group('Core')
views_group = intelligent_recommendations_group.new_group('Views')
tests_group = intelligent_recommendations_group.new_group('Tests')

puts "✅ Created IntelligentModelRecommendations group structure"

# Core files to add
core_files = [
  'TaskComplexityAnalyzer.swift',
  'UserPreferenceLearningEngine.swift',
  'HardwareCapabilityProfiler.swift',
  'ModelPerformancePredictor.swift',
  'RecommendationGenerationEngine.swift',
  'ContextAwareRecommender.swift',
  'QualityPerformanceOptimizer.swift',
  'RecommendationCacheManager.swift',
  'FeedbackLearningSystem.swift',
  'ModelCompatibilityAnalyzer.swift',
  'RecommendationExplanationEngine.swift',
  'AdaptiveRecommendationUpdater.swift',
  'IntelligentRecommendationModels.swift',
  'IntelligentRecommendationExtensions.swift',
  'MLIntegrationUtilities.swift'
]

# Views files to add
views_files = [
  'IntelligentRecommendationDashboard.swift',
  'TaskAnalysisView.swift',
  'RecommendationExplanationView.swift',
  'UserPreferenceConfigurationView.swift',
  'PerformancePredictionView.swift',
  'RecommendationFeedbackView.swift'
]

# Test files to add
test_files = [
  'TaskComplexityAnalyzerTest.swift',
  'UserPreferenceLearningEngineTest.swift',
  'HardwareCapabilityProfilerTest.swift',
  'ModelPerformancePredictorTest.swift',
  'RecommendationGenerationEngineTest.swift',
  'ContextAwareRecommenderTest.swift',
  'QualityPerformanceOptimizerTest.swift',
  'RecommendationCacheManagerTest.swift',
  'FeedbackLearningSystemTest.swift',
  'ModelCompatibilityAnalyzerTest.swift',
  'RecommendationExplanationEngineTest.swift',
  'AdaptiveRecommendationUpdaterTest.swift',
  'IntelligentRecommendationDashboardTest.swift',
  'TaskAnalysisViewTest.swift',
  'RecommendationExplanationViewTest.swift',
  'UserPreferenceConfigurationViewTest.swift',
  'PerformancePredictionViewTest.swift',
  'RecommendationFeedbackViewTest.swift'
]

# Add Core files
core_files.each do |filename|
  file_path = "IntelligentModelRecommendations/Core/#{filename}"
  file_ref = core_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to Core group"
end

# Add Views files
views_files.each do |filename|
  file_path = "IntelligentModelRecommendations/Views/#{filename}"
  file_ref = views_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to Views group"
end

# Add Test files
test_files.each do |filename|
  file_path = "IntelligentModelRecommendations/Tests/#{filename}"
  file_ref = tests_group.new_file(file_path)
  target.source_build_phase.add_file_reference(file_ref)
  puts "✅ Added #{filename} to Tests group"
end

# Save the project
project.save

puts "\n🎉 Successfully added all Intelligent Model Recommendations files to Xcode project!"
puts "📁 Total files added: #{core_files.length + views_files.length + test_files.length}"
puts "💾 Project saved successfully"