#\!/usr/bin/env python3
import os
import uuid
import re

def add_file_to_pbxproj(pbxproj_path, swift_file_path):
    """Add a Swift file to the Xcode project.pbxproj file"""
    
    # Generate unique IDs for Xcode
    file_ref_id = str(uuid.uuid4().hex[:24]).upper()
    build_file_id = str(uuid.uuid4().hex[:24]).upper()
    
    filename = os.path.basename(swift_file_path)
    
    with open(pbxproj_path, 'r') as f:
        content = f.read()
    
    # Add PBXBuildFile entry
    build_file_entry = f'\t\t{build_file_id} /* {filename} in Sources */ = {{isa = PBXBuildFile; fileRef = {file_ref_id} /* {filename} */; }};'
    
    # Find the PBXBuildFile section and add our entry
    build_file_section = re.search(r'(/* Begin PBXBuildFile section \*/.*?)(/* End PBXBuildFile section \*/)', content, re.DOTALL)
    if build_file_section:
        new_build_section = build_file_section.group(1) + build_file_entry + '\n\t\t' + build_file_section.group(2)
        content = content.replace(build_file_section.group(0), new_build_section)
    
    # Add PBXFileReference entry
    file_ref_entry = f'\t\t{file_ref_id} /* {filename} */ = {{isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = {filename}; sourceTree = "<group>"; }};'
    
    # Find the PBXFileReference section and add our entry
    file_ref_section = re.search(r'(/* Begin PBXFileReference section \*/.*?)(/* End PBXFileReference section \*/)', content, re.DOTALL)
    if file_ref_section:
        new_file_ref_section = file_ref_section.group(1) + file_ref_entry + '\n\t\t' + file_ref_section.group(2)
        content = content.replace(file_ref_section.group(0), new_file_ref_section)
    
    # Add to sources build phase
    sources_build_phase = re.search(r'(files = \(\n)(.*?)(\s+\);)', content, re.DOTALL)
    if sources_build_phase:
        build_file_line = f'\t\t\t\t{build_file_id} /* {filename} in Sources */,\n'
        new_sources = sources_build_phase.group(1) + build_file_line + sources_build_phase.group(2) + sources_build_phase.group(3)
        content = content.replace(sources_build_phase.group(0), new_sources)
    
    # Add to group (children section)
    group_children = re.search(r'(children = \(\n)(.*?E52D5E9B2C5B1A0000000001 /\* ContentView\.swift \*/,\n)(.*?)(\s+\);)', content, re.DOTALL)
    if group_children:
        file_ref_line = f'\t\t\t\t{file_ref_id} /* {filename} */,\n'
        new_children = group_children.group(1) + group_children.group(2) + file_ref_line + group_children.group(3) + group_children.group(4)
        content = content.replace(group_children.group(0), new_children)
    
    with open(pbxproj_path, 'w') as f:
        f.write(content)
    
    print(f"Added {filename} to Xcode project")

# Add the missing files
pbxproj_path = "AgenticSeek.xcodeproj/project.pbxproj"
files_to_add = [
    "AgenticSeek/ChatbotModels.swift",
    "AgenticSeek/AuthenticationManager.swift"
]

for file_path in files_to_add:
    if os.path.exists(file_path):
        add_file_to_pbxproj(pbxproj_path, file_path)
    else:
        print(f"File not found: {file_path}")
