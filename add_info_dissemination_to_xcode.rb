
require 'xcodeproj'

project_path = "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/AgenticSeek.xcodeproj"
project = Xcodeproj::Project.open(project_path)

target = project.targets.find { |t| t.name == "AgenticSeek" }
if target.nil?
    puts "Error: Target 'AgenticSeek' not found"
    exit 1
end

files_to_add = [
    "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/AgenticSeek/MLACSInfoDissemination.swift",
    "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/AgenticSeek/MLACSInfoFlowView.swift"
]

files_to_add.each do |file_path|
    if File.exist?(file_path)
        relative_path = file_path.gsub("/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS/AgenticSeek/", "")
        file_ref = project.main_group.find_file_by_path(relative_path)
        
        if file_ref.nil?
            file_ref = project.main_group.new_reference(relative_path)
            file_ref.last_known_file_type = "sourcecode.swift"
            target.source_build_phase.add_file_reference(file_ref)
            puts "Added #{relative_path} to target"
        else
            puts "#{relative_path} already in project"
        end
    else
        puts "File not found: #{file_path}"
    end
end

project.save
puts "Project saved successfully"
