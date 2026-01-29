#!/bin/bash
set -e

# Install IAB repos as dependencies
# Run from project root: ./scripts/install_iab_repos.sh

echo "Installing IAB repositories..."

mkdir -p vendor/iab

# Clone buyer-agent
if [ ! -d "vendor/iab/buyer-agent" ]; then
    echo "Cloning IAB buyer-agent..."
    git clone https://github.com/IABTechLab/buyer-agent vendor/iab/buyer-agent
else
    echo "buyer-agent already exists, pulling latest..."
    (cd vendor/iab/buyer-agent && git pull)
fi

# Clone seller-agent
if [ ! -d "vendor/iab/seller-agent" ]; then
    echo "Cloning IAB seller-agent..."
    git clone https://github.com/IABTechLab/seller-agent vendor/iab/seller-agent
else
    echo "seller-agent already exists, pulling latest..."
    (cd vendor/iab/seller-agent && git pull)
fi

# Clone agentic-direct (MCP server)
if [ ! -d "vendor/iab/agentic-direct" ]; then
    echo "Cloning IAB agentic-direct..."
    git clone https://github.com/IABTechLab/agentic-direct vendor/iab/agentic-direct
else
    echo "agentic-direct already exists, pulling latest..."
    (cd vendor/iab/agentic-direct && git pull)
fi

# Clone agentic-rtb-framework
if [ ! -d "vendor/iab/agentic-rtb-framework" ]; then
    echo "Cloning IAB agentic-rtb-framework..."
    git clone https://github.com/IABTechLab/agentic-rtb-framework vendor/iab/agentic-rtb-framework
else
    echo "agentic-rtb-framework already exists, pulling latest..."
    (cd vendor/iab/agentic-rtb-framework && git pull)
fi

# Clone agentic-audiences
if [ ! -d "vendor/iab/agentic-audiences" ]; then
    echo "Cloning IAB agentic-audiences..."
    git clone https://github.com/IABTechLab/agentic-audiences vendor/iab/agentic-audiences
else
    echo "agentic-audiences already exists, pulling latest..."
    (cd vendor/iab/agentic-audiences && git pull)
fi

echo ""
echo "IAB repos installed successfully!"
echo ""
echo "Add to your PYTHONPATH:"
echo "  export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)/vendor/iab/buyer-agent/src\""
echo "  export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)/vendor/iab/seller-agent/src\""
