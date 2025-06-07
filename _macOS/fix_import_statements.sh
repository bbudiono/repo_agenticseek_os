#!/bin/bash

echo "🔧 FIXING IMPORT STATEMENTS IN MLACS COMPONENTS"
echo "=============================================="
echo "🎯 REMOVING DUPLICATE IMPORTS AND INVALID MODULES"
echo ""

cd "/Users/bernhardbudiono/Library/CloudStorage/Dropbox/_Documents - Apps (Working)/repos_github/Working/_repo_agenticseek/_macOS"

fixed_count=0

# Fix duplicate Foundation imports
echo "📝 Fixing duplicate Foundation imports..."
find AgenticSeek/RealtimeModelDiscovery/Core/ -name "*.swift" -exec sed -i '' '/^import Foundation$/N;s/^import Foundation\nimport Foundation$/import Foundation/' {} \;
find AgenticSeek/IntelligentModelRecommendations/Core/ -name "*.swift" -exec sed -i '' '/^import Foundation$/N;s/^import Foundation\nimport Foundation$/import Foundation/' {} \;
find AgenticSeek/LocalModelCacheManagement/Core/ -name "*.swift" -exec sed -i '' '/^import Foundation$/N;s/^import Foundation\nimport Foundation$/import Foundation/' {} \;

# Remove invalid FileManager imports
echo "🗑️ Removing invalid FileManager imports..."
find AgenticSeek/ -name "*.swift" -exec sed -i '' '/^import FileManager$/d' {} \;

echo ""
echo "✅ Import statement fixes completed!"
echo "🔍 Checking for remaining issues..."

# Check for any remaining duplicate imports
duplicate_count=$(grep -r "import Foundation" AgenticSeek/RealtimeModelDiscovery/Core/ AgenticSeek/IntelligentModelRecommendations/Core/ AgenticSeek/LocalModelCacheManagement/Core/ 2>/dev/null | grep -v "View" | wc -l)
echo "📊 Remaining Foundation imports: $duplicate_count"

# Check for invalid imports
invalid_count=$(grep -r "import FileManager" AgenticSeek/ 2>/dev/null | wc -l)
echo "❌ Invalid FileManager imports: $invalid_count"

echo ""
echo "🎉 IMPORT FIXES COMPLETE!"
echo "========================"
echo "Ready to test build"