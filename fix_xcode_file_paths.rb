#!/usr/bin/env ruby
require 'xcodeproj'

project_path = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/AgenticSeek.xcodeproj"
project = Xcodeproj::Project.open(project_path)

target = project.targets.find { |t| t.name == "AgenticSeek" }
if target.nil?
    puts "Error: Target 'AgenticSeek' not found"
    exit 1
end

# Fix file paths for MLACS files
files_to_fix = [
    "MLACSInfoDissemination.swift",
    "MLACSInfoFlowView.swift",
    "MLACSEnhancedInterface.swift",
    "MLACSCoordinator.swift"
]

files_to_fix.each do |filename|
    # Find the file reference
    file_ref = project.files.find { |f| f.display_name == filename }
    
    if file_ref
        # Update the path to be relative to AgenticSeek folder
        file_ref.path = filename
        file_ref.source_tree = "<group>"
        puts "Fixed path for #{filename}"
    else
        puts "File reference not found: #{filename}"
    end
end

# Move MLACS files to the AgenticSeek group instead of root
agentic_seek_group = project.main_group.find_subpath("AgenticSeek", true)

files_to_fix.each do |filename|
    file_ref = project.files.find { |f| f.display_name == filename }
    
    if file_ref
        # Remove from current group
        file_ref.parent.children.delete(file_ref) if file_ref.parent
        
        # Add to AgenticSeek group
        agentic_seek_group.children << file_ref
        puts "Moved #{filename} to AgenticSeek group"
    end
end

project.save
puts "Project updated successfully"