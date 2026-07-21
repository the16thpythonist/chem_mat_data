#!/usr/bin/env bash
# release.sh - Build & publish script for chem_mat_data
#
# Usage: ./release.sh
#
# This script:
#   1. Validates preconditions (tools available)
#   2. Builds the package via `uv build`
#   3. Publishes the package via `uv publish`
#
# Note: version bumping is handled separately and is NOT part of this script.

set -euo pipefail

# ── Colors & helpers ─────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Precondition checks ─────────────────────────────────────────────

step "Checking preconditions"

# Check required tools
for cmd in uv; do
    if ! command -v "$cmd" &>/dev/null; then
        error "Required tool '$cmd' is not installed or not in PATH."
    fi
done
ok "All required tools are available"

# Read current version (used to scope the publish artifacts)
VERSION=$(cat chem_mat_data/VERSION)
info "Current version: ${BOLD}$VERSION${NC}"

# ── Build ────────────────────────────────────────────────────────────

step "Building package"

info "Cleaning previous build artifacts..."
rm -rf dist/

info "Executing: uv build"
uv build
ok "Package built"

# ── Publish ──────────────────────────────────────────────────────────

step "Publishing package"

info "Executing: uv publish"
uv publish dist/*"${VERSION}"*
ok "Package published"

# ── Done ─────────────────────────────────────────────────────────────

step "Release ${VERSION} complete!"
echo ""
