#!/bin/bash

# AgenticSeek macOS Native App Build Script
# This script builds the native macOS application wrapper for AgenticSeek

set -e  # Exit on any error

echo "🔧 Building AgenticSeek Native macOS App..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}Project root: ${PROJECT_ROOT}${NC}"
echo -e "${BLUE}macOS app path: ${SCRIPT_DIR}${NC}"

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo -e "${RED}❌ Error: Xcode command line tools not found${NC}"
    echo "Please install Xcode and run: xcode-select --install"
    exit 1
fi

echo -e "${GREEN}✅ Xcode found${NC}"

# Check if Docker is running (needed for AgenticSeek services)
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Docker is not running${NC}"
    echo "The app will still build, but you'll need Docker running to use AgenticSeek services"
fi

# Clean previous builds
echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
cd "$SCRIPT_DIR"
rm -rf build/
rm -rf DerivedData/

# Build the app
echo -e "${BLUE}🔨 Building AgenticSeek.app...${NC}"

xcodebuild \
    -project AgenticSeek.xcodeproj \
    -scheme AgenticSeek \
    -configuration Release \
    -derivedDataPath ./DerivedData \
    -destination "platform=macOS,arch=arm64" \
    clean build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Find the built app
BUILT_APP="./DerivedData/Build/Products/Release/AgenticSeek.app"

if [ ! -d "$BUILT_APP" ]; then
    echo -e "${RED}❌ Built app not found at expected location${NC}"
    exit 1
fi

# Create build output directory
mkdir -p ./build

# Copy the app to build directory
echo -e "${BLUE}📦 Copying built app...${NC}"
cp -R "$BUILT_APP" ./build/

# Check app signature (for distribution readiness)
echo -e "${BLUE}🔍 Checking app signature...${NC}"
codesign -dv --verbose=4 "./build/AgenticSeek.app" 2>&1 | grep -E "(Authority|TeamIdentifier|Identifier)" || true

echo ""
echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo -e "${GREEN}📍 App location: ${SCRIPT_DIR}/build/AgenticSeek.app${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Double-click the app to run it"
echo "2. Make sure Docker is running for AgenticSeek services"
echo "3. The app will automatically start backend services"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo "• Use ⌘+N for new chat"
echo "• Use ⌘+⇧+K to clear conversation"
echo "• Check menu bar for service status"
echo "• Use Settings (gear icon) to manage services"
echo ""

# Optional: Open the build directory
read -p "Open build directory in Finder? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open ./build
fi