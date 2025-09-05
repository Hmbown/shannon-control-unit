#!/bin/bash

# Shannon Control Unit Visual QA Checklist Script
# Run this after any changes to ensure visual consistency

echo "🔍 Shannon Control Unit Visual QA Check"
echo "======================================="

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check for wrong number formats
echo -n "Checking for -0.244 instances... "
if grep -r "0\.244\|−0\.244" --include="*.md" --include="*.html" --include="*.py" --exclude-dir=".git" --exclude="visual_*" --exclude="*REPORT*" . > /dev/null 2>&1; then
    echo -e "${RED}FOUND - Fix needed!${NC}"
    grep -r "0\.244\|−0\.244" --include="*.md" --include="*.html" --include="*.py" --exclude-dir=".git" --exclude="visual_*" --exclude="*REPORT*" . | head -5
else
    echo -e "${GREEN}✓ None found${NC}"
fi

# Check for correct percentage formatting
echo -n "Checking 1B model shows -6.2%... "
if grep -q "\-6\.2%" README.md; then
    echo -e "${GREEN}✓ Correct${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

echo -n "Checking 3B model shows -10.6%... "
if grep -q "\-10\.6%" README.md; then
    echo -e "${GREEN}✓ Correct${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check S* targets
echo -n "Checking 1B uses S*=1%... "
if grep -q "S\*=1%" README.md; then
    echo -e "${GREEN}✓ Correct${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

echo -n "Checking 3B uses S*=3%... "
if grep -q "S\*=3%" README.md; then
    echo -e "${GREEN}✓ Correct${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check for missing images
echo -n "Checking for missing images... "
MISSING_IMAGES=0
for img in web/images/*.png web/assets/figures/*.png web/photos/*.jpg; do
    if [ ! -f "$img" ] && [[ "$img" != *"*"* ]]; then
        echo -e "${RED}Missing: $img${NC}"
        MISSING_IMAGES=$((MISSING_IMAGES + 1))
    fi
done
if [ $MISSING_IMAGES -eq 0 ]; then
    echo -e "${GREEN}✓ All images present${NC}"
fi

# Check for TODO comments
echo -n "Checking for TODO/FIXME comments... "
if grep -r "TODO\|FIXME\|XXX\|HACK" --include="*.py" --include="*.js" --include="*.html" --exclude-dir=".git" . > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Found - Review needed${NC}"
    grep -r "TODO\|FIXME\|XXX\|HACK" --include="*.py" --include="*.js" --include="*.html" --exclude-dir=".git" . | head -3
else
    echo -e "${GREEN}✓ None found${NC}"
fi

# Check mobile breakpoints
echo -n "Checking mobile CSS breakpoints... "
BREAKPOINT_COUNT=$(grep -r "@media.*max-width.*768px" web/*.html 2>/dev/null | wc -l)
if [ $BREAKPOINT_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Found $BREAKPOINT_COUNT mobile breakpoints${NC}"
else
    echo -e "${YELLOW}⚠ No mobile breakpoints found${NC}"
fi

# Summary
echo ""
echo "======================================="
echo "Visual QA Check Complete!"
echo ""
echo "Key metrics to verify:"
echo "• 1B: -6.2% BPT, -15.6% perplexity (S*=1%)"
echo "• 3B: -10.6% BPT, -12.6% perplexity (S*=3%)"
echo ""
echo "If any issues found above, fix before pushing!"