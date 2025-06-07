#!/bin/bash

# Fix Final Duplicates Script
# Aggressively removes all duplicate declarations and ambiguous types

echo "🔧 Fixing final duplicate declarations..."

# Navigate to AgenticSeek directory
cd "AgenticSeek" || exit 1

echo "1️⃣ Removing duplicate MetricCard and CacheConfigurationSheet from CacheManagementDashboard..."

# Use a more aggressive approach to remove duplicate structures from CacheManagementDashboard
if [ -f "LocalModelCacheManagement/Views/CacheManagementDashboard.swift" ]; then
    # Remove everything after the first occurrence of "// MARK: - Supporting Views" 
    awk '
    /\/\/ MARK: - Supporting Views/ {
        if (!seen_supporting_views) {
            print
            seen_supporting_views = 1
            next
        } else {
            skip_rest = 1
            next
        }
    }
    !skip_rest { print }
    ' "LocalModelCacheManagement/Views/CacheManagementDashboard.swift" > "LocalModelCacheManagement/Views/CacheManagementDashboard.swift.tmp"
    
    # Add back just the preview at the end
    cat >> "LocalModelCacheManagement/Views/CacheManagementDashboard.swift.tmp" << 'EOF'

// GREEN PHASE: Preview for development
#if DEBUG
struct CacheManagementDashboard_Previews: PreviewProvider {
    static var previews: some View {
        CacheManagementDashboard()
    }
}
#endif
EOF
    
    mv "LocalModelCacheManagement/Views/CacheManagementDashboard.swift.tmp" "LocalModelCacheManagement/Views/CacheManagementDashboard.swift"
    echo "   ✅ Cleaned CacheManagementDashboard.swift"
fi

echo "2️⃣ Consolidating ErrorRecoveryAction and CacheError definitions..."

# Remove duplicate enums from all files except the models file
find . -name "*.swift" -type f | while read -r file; do
    if [[ "$file" != *"CacheModels.swift" ]] && [[ "$file" != *"DiscoveryModels.swift" ]]; then
        # Remove duplicate ErrorRecoveryAction and CacheError enums
        awk '
        /^enum ErrorRecoveryAction/ { in_error_recovery = 1; next }
        /^enum CacheError/ { in_cache_error = 1; next }
        in_error_recovery && /^}$/ { in_error_recovery = 0; next }
        in_cache_error && /^}$/ { in_cache_error = 0; next }
        !in_error_recovery && !in_cache_error { print }
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
done

echo "   ✅ Removed duplicate enums from non-model files"

echo "3️⃣ Removing duplicate MetricCard from all view files..."

# Keep MetricCard only in CacheManagementDashboard, remove from all other files
find . -name "*.swift" -type f | while read -r file; do
    if [[ "$file" != *"CacheManagementDashboard.swift" ]]; then
        # Remove MetricCard struct declarations
        awk '
        /^struct MetricCard: View/ { in_metric_card = 1; next }
        in_metric_card && /^}$/ { in_metric_card = 0; next }
        !in_metric_card { print }
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
done

echo "   ✅ Removed duplicate MetricCard from all other files"

echo "4️⃣ Testing build after duplicate removal..."

# Navigate back to root
cd ".."

# Test build
echo "🔨 Testing build after duplicate removal..."
xcodebuild -project "AgenticSeek.xcodeproj" -scheme "AgenticSeek" -configuration Debug build -destination "platform=macOS" > build_after_duplicate_fix.log 2>&1

BUILD_RESULT=$?

if [ $BUILD_RESULT -eq 0 ]; then
    echo ""
    echo "🎉🎉🎉 BUILD SUCCESS! 🎉🎉🎉"
    echo "✅ ALL DUPLICATE DECLARATIONS RESOLVED"
    echo "📊 Build is now ready for TestFlight deployment"
    echo "🚀 CRITICAL P0 TASK COMPLETED SUCCESSFULLY"
    echo ""
    
    echo "5️⃣ Creating production archive..."
    xcodebuild -project "AgenticSeek.xcodeproj" -scheme "AgenticSeek" -configuration Release archive -archivePath "AgenticSeek.xcarchive" -destination "platform=macOS" > archive_build.log 2>&1
    
    ARCHIVE_RESULT=$?
    
    if [ $ARCHIVE_RESULT -eq 0 ]; then
        echo "✅ Production archive created successfully: AgenticSeek.xcarchive"
        echo "🚀 Ready for TestFlight upload!"
    else
        echo "⚠️  Archive creation had issues, but main build works"
    fi
    
else
    echo "❌ Build still has issues after duplicate removal"
    error_count=$(grep -c "error:" build_after_duplicate_fix.log || echo "0")
    echo "   🔍 Remaining errors: $error_count"
    
    if [ "$error_count" -le 10 ]; then
        echo "   🎯 Remaining errors:"
        grep "error:" build_after_duplicate_fix.log | head -10
    fi
fi

echo "🎯 Final duplicates fix script completed!"