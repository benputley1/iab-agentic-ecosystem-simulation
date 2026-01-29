#!/bin/bash
# Install IAB Tech Lab repositories as dependencies
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_ROOT/vendor/iab"

echo "Installing IAB repositories to $VENDOR_DIR"

mkdir -p "$VENDOR_DIR"

# Clone buyer-agent
if [ ! -d "$VENDOR_DIR/buyer-agent" ]; then
    echo "Cloning buyer-agent..."
    git clone https://github.com/IABTechLab/buyer-agent "$VENDOR_DIR/buyer-agent"
else
    echo "buyer-agent already exists, pulling latest..."
    (cd "$VENDOR_DIR/buyer-agent" && git pull)
fi

# Clone seller-agent
if [ ! -d "$VENDOR_DIR/seller-agent" ]; then
    echo "Cloning seller-agent..."
    git clone https://github.com/IABTechLab/seller-agent "$VENDOR_DIR/seller-agent"
else
    echo "seller-agent already exists, pulling latest..."
    (cd "$VENDOR_DIR/seller-agent" && git pull)
fi

# Clone agentic-direct (MCP server)
if [ ! -d "$VENDOR_DIR/agentic-direct" ]; then
    echo "Cloning agentic-direct..."
    git clone https://github.com/IABTechLab/agentic-direct "$VENDOR_DIR/agentic-direct"
else
    echo "agentic-direct already exists, pulling latest..."
    (cd "$VENDOR_DIR/agentic-direct" && git pull)
fi

# Clone agentic-rtb-framework
if [ ! -d "$VENDOR_DIR/agentic-rtb-framework" ]; then
    echo "Cloning agentic-rtb-framework..."
    git clone https://github.com/IABTechLab/agentic-rtb-framework "$VENDOR_DIR/agentic-rtb-framework"
else
    echo "agentic-rtb-framework already exists, pulling latest..."
    (cd "$VENDOR_DIR/agentic-rtb-framework" && git pull)
fi

# Clone agentic-audiences
if [ ! -d "$VENDOR_DIR/agentic-audiences" ]; then
    echo "Cloning agentic-audiences..."
    git clone https://github.com/IABTechLab/agentic-audiences "$VENDOR_DIR/agentic-audiences"
else
    echo "agentic-audiences already exists, pulling latest..."
    (cd "$VENDOR_DIR/agentic-audiences" && git pull)
fi

echo ""
echo "IAB repos installed successfully!"
echo ""
echo "To use in Python, add to PYTHONPATH:"
echo "  export PYTHONPATH=\"\${PYTHONPATH}:$VENDOR_DIR/buyer-agent/src\""
echo "  export PYTHONPATH=\"\${PYTHONPATH}:$VENDOR_DIR/seller-agent/src\""
