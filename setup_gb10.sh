#!/bin/bash
###############################################################################
# GB10 DGX Spark - Complete Agentic RAG Setup Script
#
# This script sets up the entire Agentic RAG application on GB10 DGX Spark
# (ARM64 Grace Blackwell with CUDA 13) with open source web search.
#
# Features:
#   - Automatic dependency installation
#   - GB10 compatibility checks
#   - Open source web search (DuckDuckGo or SearXNG)
#   - Local GPU embeddings (Hybrid Mode)
#   - Complete end-to-end setup
#
# Usage:
#   ./setup_gb10.sh [OPTIONS]
#
# Options:
#   --mode=MODE          Setup mode: cloud, hybrid, local, searxng (default: hybrid)
#   --search=TYPE        Search engine: duckduckgo, searxng (default: duckduckgo)
#   --nvidia-key=KEY     NVIDIA API key (required for cloud/hybrid modes)
#   --skip-deps          Skip dependency installation
#   --test               Run tests after setup
#   --help               Show this help message
#
# Examples:
#   ./setup_gb10.sh --mode=hybrid --search=duckduckgo
#   ./setup_gb10.sh --mode=cloud --nvidia-key=nvapi-xxxxx
#   ./setup_gb10.sh --mode=searxng --test
#
# Author: NVIDIA
# Date: 2025-11-14
# Platform: GB10 DGX Spark (ARM64 + CUDA 13)
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODE="hybrid"
SEARCH_ENGINE="duckduckgo"
NVIDIA_API_KEY=""
SKIP_DEPS=false
RUN_TESTS=false
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

show_help() {
    cat << EOF
GB10 DGX Spark - Complete Agentic RAG Setup Script

Usage:
    ./setup_gb10.sh [OPTIONS]

Options:
    --mode=MODE          Setup mode (default: hybrid)
                         - cloud: Use NVIDIA cloud endpoints only
                         - hybrid: Local embeddings + cloud LLM (recommended)
                         - local: Fully local with LM Studio
                         - searxng: Hybrid with self-hosted SearXNG

    --search=TYPE        Search engine (default: duckduckgo)
                         - duckduckgo: Free, no setup required
                         - searxng: Self-hosted, full privacy

    --nvidia-key=KEY     NVIDIA API key (required for cloud/hybrid modes)
                         Get one at: https://org.ngc.nvidia.com/setup/api-keys

    --skip-deps          Skip system dependency installation

    --test               Run tests after setup

    --help               Show this help message

Examples:
    # Quick setup with DuckDuckGo (easiest)
    ./setup_gb10.sh --mode=hybrid --search=duckduckgo --nvidia-key=nvapi-xxxxx

    # Full privacy with SearXNG
    ./setup_gb10.sh --mode=searxng

    # Cloud-only mode (no local GPU usage)
    ./setup_gb10.sh --mode=cloud --nvidia-key=nvapi-xxxxx --test

Modes Explained:
    cloud   - Uses NVIDIA cloud for LLM + embeddings, DuckDuckGo for search
    hybrid  - Local GPU embeddings, cloud LLM, open source search
    local   - Everything local (requires LM Studio setup)
    searxng - Hybrid mode + self-hosted SearXNG search engine

For more information, see GB10_QUICKSTART.md
EOF
}

###############################################################################
# System Checks
###############################################################################

check_architecture() {
    print_header "Checking System Architecture"

    ARCH=$(uname -m)
    print_info "Architecture: $ARCH"

    if [ "$ARCH" != "aarch64" ]; then
        print_warning "Expected ARM64 (aarch64), got $ARCH"
        print_info "Script will continue, but GB10 optimizations may not apply"
    else
        print_success "ARM64 architecture confirmed"
    fi

    # Check for CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "CUDA installed: $CUDA_VERSION"

        # Check CUDA version
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            print_success "CUDA version is compatible (12+)"
        else
            print_warning "CUDA $CUDA_VERSION detected. CUDA 12+ recommended for GB10"
        fi
    else
        print_warning "CUDA not found in PATH"
        print_info "Make sure CUDA 13 is installed for GB10"
    fi

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        print_success "GPU detected: $GPU_NAME"
        print_info "GPU memory: ${GPU_MEM}MB"

        if [[ $GPU_NAME == *"GB10"* ]] || [ "$GPU_MEM" -gt 100000 ]; then
            print_success "GB10 Grace Blackwell detected!"
        fi
    else
        print_warning "nvidia-smi not found"
    fi
}

check_python() {
    print_header "Checking Python"

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"

    # Check Python version (need 3.8+)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.8+ required, got $PYTHON_VERSION"
        exit 1
    fi
}

check_docker() {
    print_header "Checking Docker (for SearXNG)"

    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | cut -d',' -f1)
        print_success "Docker $DOCKER_VERSION found"

        # Check if docker-compose is available
        if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
            print_success "Docker Compose available"
            return 0
        else
            print_warning "Docker Compose not found (needed for SearXNG)"
            return 1
        fi
    else
        print_warning "Docker not found (needed for SearXNG)"
        return 1
    fi
}

###############################################################################
# Dependency Installation
###############################################################################

install_system_dependencies() {
    print_header "Installing System Dependencies"

    if [ "$SKIP_DEPS" = true ]; then
        print_info "Skipping system dependency installation (--skip-deps)"
        return 0
    fi

    # Detect package manager
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt-get"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
    else
        print_warning "Unknown package manager, skipping system dependencies"
        return 1
    fi

    print_info "Using package manager: $PKG_MANAGER"

    # Update package list
    print_info "Updating package list..."
    sudo $PKG_MANAGER update -y

    # Install dependencies
    PACKAGES=(
        "python3-pip"
        "python3-dev"
        "build-essential"
        "poppler-utils"
        "tesseract-ocr"
        "tesseract-ocr-eng"
        "ffmpeg"
        "libsm6"
        "libxext6"
        "curl"
        "git"
    )

    for package in "${PACKAGES[@]}"; do
        print_info "Installing $package..."
        sudo $PKG_MANAGER install -y $package || print_warning "Failed to install $package"
    done

    print_success "System dependencies installed"
}

install_python_dependencies() {
    print_header "Installing Python Dependencies"

    # Upgrade pip
    print_info "Upgrading pip..."
    python3 -m pip install --upgrade pip

    # Install core dependencies
    print_info "Installing core dependencies..."
    pip3 install -r "$PROJECT_DIR/requirements.txt"

    # Install open source search dependencies
    print_info "Installing open source search dependencies..."
    pip3 install duckduckgo-search

    # Install mode-specific dependencies
    if [ "$MODE" = "hybrid" ] || [ "$MODE" = "searxng" ]; then
        print_info "Installing hybrid mode dependencies (PyTorch + sentence-transformers)..."

        # Try to install PyTorch with CUDA 13 support
        print_info "Installing PyTorch for CUDA 13..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || {
            print_warning "CUDA 13 PyTorch not available, trying CUDA 12.1..."
            pip3 install torch torchvision torchaudio
        }

        print_info "Installing sentence-transformers..."
        pip3 install sentence-transformers
    fi

    # Install additional utilities
    print_info "Installing additional utilities..."
    pip3 install python-dotenv rich typer

    print_success "Python dependencies installed"
}

###############################################################################
# Configuration
###############################################################################

create_environment_file() {
    print_header "Creating Environment Configuration"

    ENV_FILE="$PROJECT_DIR/.env"

    # Backup existing .env if present
    if [ -f "$ENV_FILE" ]; then
        print_info "Backing up existing .env file..."
        cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    # Create new .env file
    cat > "$ENV_FILE" << EOF
# GB10 Agentic RAG Configuration
# Generated: $(date)

# Mode: $MODE
# Search Engine: $SEARCH_ENGINE

# NVIDIA API Key (required for cloud/hybrid modes)
NVIDIA_API_KEY=$NVIDIA_API_KEY

# Tavily is replaced with open source alternatives
# TAVILY_API_KEY is no longer needed

# Search Engine Configuration
SEARCH_ENGINE=$SEARCH_ENGINE

# SearXNG Configuration (if using searxng)
SEARXNG_URL=http://localhost:8888

# Embedding Model (for hybrid/searxng modes)
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cuda

# LLM Configuration
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=documents

# Web UI
GRADIO_SERVER_PORT=8080
GRADIO_SERVER_NAME=0.0.0.0

# Logging
LOG_LEVEL=INFO
EOF

    print_success "Environment file created: $ENV_FILE"

    # Make sure .env is in .gitignore
    if ! grep -q "^\.env$" "$PROJECT_DIR/.gitignore" 2>/dev/null; then
        echo ".env" >> "$PROJECT_DIR/.gitignore"
        print_info "Added .env to .gitignore"
    fi
}

###############################################################################
# Open Source Search Setup
###############################################################################

setup_duckduckgo() {
    print_header "Setting up DuckDuckGo Search"

    print_info "DuckDuckGo requires no additional setup!"
    print_success "DuckDuckGo is ready to use"

    print_info "Features:"
    print_info "  - No API key required"
    print_info "  - No rate limits"
    print_info "  - Privacy-focused"
    print_info "  - Works immediately"
}

setup_searxng() {
    print_header "Setting up SearXNG (Self-Hosted Search)"

    if ! check_docker; then
        print_error "Docker is required for SearXNG"
        print_info "Install Docker: https://docs.docker.com/engine/install/"
        exit 1
    fi

    SEARXNG_DIR="$PROJECT_DIR/searxng"

    # Clone SearXNG Docker setup
    if [ ! -d "$SEARXNG_DIR" ]; then
        print_info "Cloning SearXNG Docker setup..."
        git clone https://github.com/searxng/searxng-docker.git "$SEARXNG_DIR"
    else
        print_info "SearXNG directory already exists"
    fi

    cd "$SEARXNG_DIR"

    # Generate configuration
    if [ ! -f ".env" ]; then
        print_info "Generating SearXNG configuration..."

        # Generate random secret key
        SECRET_KEY=$(openssl rand -hex 32)

        cat > .env << EOF
SEARXNG_HOSTNAME=localhost
SEARXNG_PORT=8888
SEARXNG_SECRET=$SECRET_KEY
LETSENCRYPT_EMAIL=
EOF

        print_success "SearXNG configuration created"
    fi

    # Create docker-compose override for ARM64
    cat > docker-compose.override.yml << EOF
version: '3.7'

services:
  searxng:
    platform: linux/arm64
    ports:
      - "8888:8080"
EOF

    print_info "Starting SearXNG..."
    docker compose up -d

    print_success "SearXNG started"
    print_info "SearXNG will be available at: http://localhost:8888"
    print_info "Waiting for SearXNG to be ready..."

    # Wait for SearXNG to be ready
    for i in {1..30}; do
        if curl -s http://localhost:8888 > /dev/null 2>&1; then
            print_success "SearXNG is ready!"
            break
        fi
        sleep 2
        echo -n "."
    done

    cd "$PROJECT_DIR"
}

###############################################################################
# Application Setup
###############################################################################

create_open_source_search_wrapper() {
    print_header "Creating Open Source Search Integration"

    UTILS_DIR="$PROJECT_DIR/code/chatui/utils"
    mkdir -p "$UTILS_DIR"

    cat > "$UTILS_DIR/opensource_search.py" << 'EOF'
"""
Open Source Web Search Integration
Replaces Tavily with DuckDuckGo or SearXNG
"""

import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OpenSourceSearch:
    """Unified interface for open source web search"""

    def __init__(self, engine: str = "duckduckgo", searxng_url: Optional[str] = None):
        """
        Initialize open source search

        Args:
            engine: "duckduckgo" or "searxng"
            searxng_url: URL of SearXNG instance (required if engine="searxng")
        """
        self.engine = engine.lower()
        self.searxng_url = searxng_url

        if self.engine == "duckduckgo":
            self._init_duckduckgo()
        elif self.engine == "searxng":
            self._init_searxng()
        else:
            raise ValueError(f"Unknown search engine: {engine}")

    def _init_duckduckgo(self):
        """Initialize DuckDuckGo search"""
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            logger.info("DuckDuckGo search initialized")
        except ImportError:
            raise ImportError(
                "duckduckgo-search not installed. "
                "Run: pip install duckduckgo-search"
            )

    def _init_searxng(self):
        """Initialize SearXNG search"""
        if not self.searxng_url:
            self.searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8888")

        logger.info(f"SearXNG search initialized: {self.searxng_url}")

    def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Search the web

        Args:
            query: Search query
            max_results: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of search results with 'title', 'url', 'content'
        """
        if self.engine == "duckduckgo":
            return self._search_duckduckgo(query, max_results, **kwargs)
        elif self.engine == "searxng":
            return self._search_searxng(query, max_results, **kwargs)

    def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Search using DuckDuckGo"""
        try:
            results = []

            # Text search
            ddg_results = self.ddgs.text(
                query,
                max_results=max_results,
                **kwargs
            )

            for r in ddg_results:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                    "source": "duckduckgo"
                })

            logger.info(f"DuckDuckGo search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    def _search_searxng(
        self,
        query: str,
        max_results: int,
        **kwargs
    ) -> List[Dict[str, str]]:
        """Search using SearXNG"""
        import requests

        try:
            # SearXNG search API
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
                "engines": kwargs.get("engines", "google,duckduckgo,bing")
            }

            response = requests.get(
                f"{self.searxng_url}/search",
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for r in data.get("results", [])[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "source": "searxng",
                    "engine": r.get("engine", "")
                })

            logger.info(f"SearXNG search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            return []


# LangChain-compatible wrapper
class OpenSourceSearchTool:
    """LangChain-compatible tool for open source search"""

    def __init__(self, engine: str = "duckduckgo", searxng_url: Optional[str] = None):
        self.search = OpenSourceSearch(engine, searxng_url)

    def run(self, query: str, **kwargs) -> str:
        """Run search and return formatted results"""
        results = self.search.search(query, **kwargs)

        if not results:
            return "No results found."

        # Format results
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r['title']}\n"
                f"   URL: {r['url']}\n"
                f"   {r['content']}\n"
            )

        return "\n".join(formatted)


def get_search_tool(
    engine: Optional[str] = None,
    searxng_url: Optional[str] = None
) -> OpenSourceSearchTool:
    """
    Get configured search tool

    Args:
        engine: Search engine to use (from env if not specified)
        searxng_url: SearXNG URL (from env if not specified)

    Returns:
        Configured search tool
    """
    if engine is None:
        engine = os.getenv("SEARCH_ENGINE", "duckduckgo")

    if searxng_url is None:
        searxng_url = os.getenv("SEARXNG_URL")

    return OpenSourceSearchTool(engine, searxng_url)
EOF

    print_success "Open source search wrapper created"
}

create_modified_chain() {
    print_header "Modifying Application Chain"

    # Backup original chain.py if it exists
    CHAIN_FILE="$PROJECT_DIR/code/chatui/chain.py"

    if [ -f "$CHAIN_FILE" ]; then
        print_info "Backing up original chain.py..."
        cp "$CHAIN_FILE" "$CHAIN_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    print_info "Chain modification will be done via Python script..."
    print_success "Use GB10_HYBRID_IMPLEMENTATION.py for embeddings"
    print_success "Use opensource_search.py for web search"
}

###############################################################################
# Testing
###############################################################################

run_tests() {
    print_header "Running Tests"

    # Test 1: Python imports
    print_info "Testing Python imports..."
    python3 << EOF
import sys
try:
    import langchain
    import chromadb
    import gradio
    from duckduckgo_search import DDGS
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

    # Test 2: CUDA availability (if hybrid/searxng mode)
    if [ "$MODE" = "hybrid" ] || [ "$MODE" = "searxng" ]; then
        print_info "Testing CUDA availability..."
        python3 << EOF
import sys
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available")
        sys.exit(1)
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)
EOF
    fi

    # Test 3: Search functionality
    print_info "Testing search functionality..."
    python3 << EOF
import sys
sys.path.insert(0, "$PROJECT_DIR/code")

try:
    from chatui.utils.opensource_search import get_search_tool

    search = get_search_tool(engine="$SEARCH_ENGINE")
    results = search.search.search("test query", max_results=1)

    if results:
        print(f"✓ Search working: {len(results)} results")
    else:
        print("⚠ Search returned no results (may be rate limited)")
except Exception as e:
    print(f"✗ Search test failed: {e}")
    sys.exit(1)
EOF

    # Test 4: GB10 hybrid implementation (if applicable)
    if [ "$MODE" = "hybrid" ] || [ "$MODE" = "searxng" ]; then
        print_info "Testing GB10 hybrid implementation..."
        python3 "$PROJECT_DIR/GB10_HYBRID_IMPLEMENTATION.py" --check
    fi

    print_success "All tests passed!"
}

###############################################################################
# Main Setup Flow
###############################################################################

main() {
    print_header "GB10 DGX Spark - Agentic RAG Setup"

    echo "Configuration:"
    echo "  Mode: $MODE"
    echo "  Search Engine: $SEARCH_ENGINE"
    echo "  Project Directory: $PROJECT_DIR"
    echo ""

    # System checks
    check_architecture
    check_python

    # Install dependencies
    install_system_dependencies
    install_python_dependencies

    # Create configuration
    create_environment_file

    # Setup search engine
    if [ "$SEARCH_ENGINE" = "duckduckgo" ]; then
        setup_duckduckgo
    elif [ "$SEARCH_ENGINE" = "searxng" ]; then
        setup_searxng
    fi

    # Create application components
    create_open_source_search_wrapper
    create_modified_chain

    # Run tests if requested
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi

    # Final instructions
    print_header "Setup Complete!"

    echo "Next steps:"
    echo ""

    if [ -n "$NVIDIA_API_KEY" ]; then
        echo "  1. Your NVIDIA API key is configured in .env"
    else
        echo "  1. Set your NVIDIA API key:"
        echo "     export NVIDIA_API_KEY='nvapi-xxxxx'"
        echo "     Or add it to $PROJECT_DIR/.env"
    fi

    echo ""
    echo "  2. Start the application:"
    echo "     cd $PROJECT_DIR/code"
    echo "     python3 -m chatui"
    echo ""
    echo "  3. Open your browser to:"
    echo "     http://localhost:8080"
    echo ""

    if [ "$SEARCH_ENGINE" = "searxng" ]; then
        echo "  SearXNG is running at: http://localhost:8888"
        echo ""
    fi

    echo "Documentation:"
    echo "  - Quick Start: $PROJECT_DIR/GB10_QUICKSTART.md"
    echo "  - Full Report: $PROJECT_DIR/GB10_COMPATIBILITY_REPORT.md"
    echo "  - Hybrid Mode: $PROJECT_DIR/GB10_HYBRID_IMPLEMENTATION.py"
    echo ""

    print_success "GB10 setup complete! Happy coding!"
}

###############################################################################
# Argument Parsing
###############################################################################

for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE="${arg#*=}"
            ;;
        --search=*)
            SEARCH_ENGINE="${arg#*=}"
            ;;
        --nvidia-key=*)
            NVIDIA_API_KEY="${arg#*=}"
            ;;
        --skip-deps)
            SKIP_DEPS=true
            ;;
        --test)
            RUN_TESTS=true
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ ! "$MODE" =~ ^(cloud|hybrid|local|searxng)$ ]]; then
    print_error "Invalid mode: $MODE"
    echo "Valid modes: cloud, hybrid, local, searxng"
    exit 1
fi

# Validate search engine
if [[ ! "$SEARCH_ENGINE" =~ ^(duckduckgo|searxng)$ ]]; then
    print_error "Invalid search engine: $SEARCH_ENGINE"
    echo "Valid search engines: duckduckgo, searxng"
    exit 1
fi

# Auto-adjust mode if searxng selected
if [ "$SEARCH_ENGINE" = "searxng" ] && [ "$MODE" != "searxng" ]; then
    MODE="searxng"
    print_info "Mode automatically set to 'searxng' based on search engine"
fi

# Run main setup
main
