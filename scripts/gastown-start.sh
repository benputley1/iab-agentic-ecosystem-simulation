#!/bin/bash
# Gas Town Quick Start for RTB Simulation
# Usage: ./scripts/gastown-start.sh [--full|--mock|--reset]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 IAB Agentic RTB Simulation - Gas Town Deployment${NC}"
echo ""

# Parse arguments
MODE="mock"
RESET=false

for arg in "$@"; do
    case $arg in
        --full)
            MODE="full"
            shift
            ;;
        --mock)
            MODE="mock"
            shift
            ;;
        --reset)
            RESET=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--full|--mock|--reset]"
            echo ""
            echo "Options:"
            echo "  --full   Run with real LLM (costs ~\$8-15)"
            echo "  --mock   Run with mock LLM (free, default)"
            echo "  --reset  Reset database and start fresh"
            exit 0
            ;;
    esac
done

# Step 1: Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.11+.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Step 2: Reset if requested
if [ "$RESET" = true ]; then
    echo -e "${YELLOW}🔄 Resetting database...${NC}"
    cd "$PROJECT_DIR/docker"
    docker-compose down -v 2>/dev/null || true
    echo -e "${GREEN}✅ Database reset${NC}"
fi

# Step 3: Start Docker services
echo -e "${YELLOW}🐳 Starting Docker services...${NC}"
cd "$PROJECT_DIR/docker"

# Use parallel compose if available
if [ -f "docker-compose.parallel.yml" ]; then
    docker-compose -f docker-compose.parallel.yml up -d
else
    docker-compose up -d
fi

# Step 4: Wait for services
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"

MAX_RETRIES=30
RETRY_COUNT=0

until docker exec rtb_postgres pg_isready -U rtb_sim > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}❌ PostgreSQL failed to start${NC}"
        exit 1
    fi
    echo "  Waiting for PostgreSQL... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo -e "${GREEN}  ✅ PostgreSQL ready${NC}"

until docker exec rtb_redis redis-cli ping > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}❌ Redis failed to start${NC}"
        exit 1
    fi
    echo "  Waiting for Redis... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo -e "${GREEN}  ✅ Redis ready${NC}"

until curl -sf http://localhost:8086/health > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}❌ InfluxDB failed to start${NC}"
        exit 1
    fi
    echo "  Waiting for InfluxDB... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo -e "${GREEN}  ✅ InfluxDB ready${NC}"

until curl -sf http://localhost:3000/api/health > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}❌ Grafana failed to start${NC}"
        exit 1
    fi
    echo "  Waiting for Grafana... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done
echo -e "${GREEN}  ✅ Grafana ready${NC}"

# Step 5: Set up Python environment if needed
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}  ✅ Created virtual environment${NC}"
fi

source .venv/bin/activate
pip install -q -e ".[full]" 2>/dev/null || pip install -q -e .
echo -e "${GREEN}  ✅ Dependencies installed${NC}"

# Step 6: Check for .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}  ⚠️  Created .env from example. Please add your ANTHROPIC_API_KEY.${NC}"
fi

# Step 7: Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Gas Town Ready for RTB Simulation!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Services:${NC}"
echo "  • PostgreSQL: localhost:5432"
echo "  • Redis:      localhost:6379"
echo "  • InfluxDB:   http://localhost:8086"
echo "  • Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo -e "${BLUE}Quick Commands:${NC}"
echo ""
if [ "$MODE" = "mock" ]; then
    echo "  # Run simulation (mock LLM - FREE)"
    echo "  source .venv/bin/activate"
    echo "  rtb-sim run --days 30 --mock-llm --scenario a,b,c"
else
    echo "  # Run simulation (real LLM - ~\$8-15)"
    echo "  source .venv/bin/activate"
    echo "  rtb-sim run --days 30 --real-llm --scenario a,b,c"
fi
echo ""
echo "  # Test single scenario"
echo "  rtb-sim test-scenario --scenario c --mock-llm"
echo ""
echo "  # Compare results"
echo "  rtb-sim compare results/run.json --format markdown"
echo ""
echo -e "${BLUE}Grafana Dashboards:${NC}"
echo "  Open http://localhost:3000 → Login: admin/admin"
echo "  • RTB Overview"
echo "  • Scenario Comparison"
echo "  • Context Rot Analysis"
echo ""
echo -e "${YELLOW}To stop: cd docker && docker-compose down${NC}"
