#!/bin/bash

# Comprehensive Build Issues Fix Script
# Addresses duplicate types, broken templates, and import issues

echo "🔧 Starting comprehensive build issues fix..."

# Navigate to AgenticSeek directory
cd "_macOS/AgenticSeek" || exit 1

echo "📝 Step 1: Fixing duplicate import statements..."

# Fix duplicate imports across all MLACS files
find . -name "*.swift" -type f | while read -r file; do
    # Remove duplicate import Foundation statements
    if grep -q "import Foundation" "$file"; then
        # Remove all import Foundation lines except the first one
        awk '!seen["import Foundation"]++ { if ($0 == "import Foundation" && seen["import Foundation"] > 1) next } 1' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
    
    # Remove duplicate import SwiftUI statements
    if grep -q "import SwiftUI" "$file"; then
        awk '!seen["import SwiftUI"]++ { if ($0 == "import SwiftUI" && seen["import SwiftUI"] > 1) next } 1' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
    
    # Remove invalid import statements
    sed -i '' '/import FileManager/d' "$file" 2>/dev/null || true
    sed -i '' '/import URL/d' "$file" 2>/dev/null || true
    sed -i '' '/import Data/d' "$file" 2>/dev/null || true
done

echo "🔧 Step 2: Fixing template variables in button labels..."

# Fix template variables in CacheAnalyticsView
if [ -f "LocalModelCacheManagement/Views/CacheAnalyticsView.swift" ]; then
    sed -i '' 's/Button("Displayanalytics")/Button("Display Analytics")/g' "LocalModelCacheManagement/Views/CacheAnalyticsView.swift"
    sed -i '' 's/Button("Generateinsights")/Button("Generate Insights")/g' "LocalModelCacheManagement/Views/CacheAnalyticsView.swift"
    sed -i '' 's/Button("Createperformancereports")/Button("Create Performance Reports")/g' "LocalModelCacheManagement/Views/CacheAnalyticsView.swift"
fi

# Fix template variables in CacheManagementDashboard
if [ -f "LocalModelCacheManagement/Views/CacheManagementDashboard.swift" ]; then
    sed -i '' 's/Button("Displaystatus")/Button("Display Status")/g' "LocalModelCacheManagement/Views/CacheManagementDashboard.swift"
    sed -i '' 's/Button("Showperformancemetrics")/Button("Show Performance Metrics")/g' "LocalModelCacheManagement/Views/CacheManagementDashboard.swift"
    sed -i '' 's/Button("Providecontrols")/Button("Provide Controls")/g' "LocalModelCacheManagement/Views/CacheManagementDashboard.swift"
fi

echo "🔧 Step 3: Fixing duplicate struct declarations..."

# Fix duplicate MetricCard and CacheConfigurationSheet in CacheAnalyticsView
if [ -f "LocalModelCacheManagement/Views/CacheAnalyticsView.swift" ]; then
    # Remove duplicate MetricCard and CacheConfigurationSheet from CacheAnalyticsView
    # Keep only the definitions in CacheManagementDashboard
    awk '
    /^struct MetricCard: View {/ { in_metric_card = 1; next }
    /^struct CacheConfigurationSheet: View {/ { in_config_sheet = 1; next }
    in_metric_card && /^}$/ { in_metric_card = 0; next }
    in_config_sheet && /^}$/ { in_config_sheet = 0; next }
    !in_metric_card && !in_config_sheet { print }
    ' "LocalModelCacheManagement/Views/CacheAnalyticsView.swift" > "LocalModelCacheManagement/Views/CacheAnalyticsView.swift.tmp"
    mv "LocalModelCacheManagement/Views/CacheAnalyticsView.swift.tmp" "LocalModelCacheManagement/Views/CacheAnalyticsView.swift"
fi

echo "🔧 Step 4: Fixing missing property references..."

# Fix undefined isInitialized property
find . -name "*.swift" -type f -exec grep -l "isInitialized" {} \; | while read -r file; do
    # Add @State private var isInitialized = false at the top of the struct
    if ! grep -q "@State private var isInitialized" "$file"; then
        sed -i '' '/struct.*View {/a\
    @State private var isInitialized = false
' "$file"
    fi
done

echo "🔧 Step 5: Fixing DiscoveryResult property access..."

# Fix availability_status property reference in DiscoveryModels.swift
if [ -f "RealtimeModelDiscovery/Core/DiscoveryModels.swift" ]; then
    sed -i '' 's/\$0.availability_status == "available"/\$0.isAvailable/g' "RealtimeModelDiscovery/Core/DiscoveryModels.swift"
fi

echo "🔧 Step 6: Adding missing NSFetchRequest import..."

# Add CoreData import where NSFetchRequest is used
find . -name "*.swift" -type f -exec grep -l "NSFetchRequest" {} \; | while read -r file; do
    if ! grep -q "import CoreData" "$file"; then
        sed -i '' '/import Foundation/a\
import CoreData
' "$file"
    fi
done

echo "🔧 Step 7: Creating missing ModelEntity class..."

# Create ModelEntity class if it doesn't exist
if ! find . -name "*.swift" -exec grep -l "class ModelEntity" {} \; | head -1 > /dev/null; then
    cat > "RealtimeModelDiscovery/Core/ModelEntity.swift" << 'EOF'
import Foundation
import CoreData

@objc(ModelEntity)
public class ModelEntity: NSManagedObject {
    @NSManaged public var id: String
    @NSManaged public var name: String
    @NSManaged public var provider: String
    @NSManaged public var modelType: String
    @NSManaged public var size: String
    @NSManaged public var endpoint: String
    @NSManaged public var discovered: Date
    @NSManaged public var isAvailable: Bool
}

extension ModelEntity {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<ModelEntity> {
        return NSFetchRequest<ModelEntity>(entityName: "ModelEntity")
    }
}
EOF
fi

echo "🔧 Step 8: Fixing CacheType enum conflicts..."

# Remove duplicate CacheType enums from view files, keep only in CacheModels.swift
find . -name "*.swift" -type f | while read -r file; do
    if [[ "$file" != *"CacheModels.swift" ]] && grep -q "enum CacheType" "$file"; then
        # Remove the enum CacheType definition from this file
        awk '
        /^enum CacheType/ { in_enum = 1; next }
        in_enum && /^}$/ { in_enum = 0; next }
        !in_enum { print }
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
done

echo "🔧 Step 9: Adding CacheType enum to CacheModels.swift if missing..."

# Ensure CacheType is properly defined in CacheModels.swift
if [ -f "LocalModelCacheManagement/Models/CacheModels.swift" ] && ! grep -q "enum CacheType" "LocalModelCacheManagement/Models/CacheModels.swift"; then
    cat >> "LocalModelCacheManagement/Models/CacheModels.swift" << 'EOF'

// MARK: - Cache Type Enum
enum CacheType: String, Codable, CaseIterable {
    case modelWeights = "model_weights"
    case activations = "activations" 
    case computationResults = "computation_results"
    case sharedParameters = "shared_parameters"
    case compressedData = "compressed_data"
    
    var displayName: String {
        switch self {
        case .modelWeights: return "Model Weights"
        case .activations: return "Intermediate Activations"
        case .computationResults: return "Computation Results"
        case .sharedParameters: return "Shared Parameters"
        case .compressedData: return "Compressed Data"
        }
    }
}
EOF
fi

echo "🔧 Step 10: Testing build compilation..."

# Navigate back to root Xcode project
cd "../.."

# Test if the build compiles now
echo "🔨 Testing Xcode build..."
xcodebuild -workspace "_macOS/AgenticSeek.xcworkspace" -scheme "AgenticSeek" -configuration Debug build -destination "platform=macOS" > build_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ BUILD SUCCESS! All issues have been resolved."
    echo "📊 Build completed successfully and ready for TestFlight deployment."
else
    echo "❌ Build still has issues. Check build_test.log for details."
    echo "🔍 Top 10 build errors:"
    grep -i "error:" build_test.log | head -10
fi

echo "🎯 Comprehensive fix script completed!"