#!/bin/bash

echo "🔨 Building Working AgenticSeek App..."

# Compile Swift app
swiftc -o WorkingAgenticSeekApp main.swift -framework SwiftUI -framework Cocoa

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Launching app..."
    ./WorkingAgenticSeekApp &
    echo "✅ App launched!"
else
    echo "❌ Build failed"
    exit 1
fi
