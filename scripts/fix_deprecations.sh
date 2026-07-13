#!/bin/bash
#
# Fix Mojo 0.26.3 deprecation warnings across all .mojo files
#
# This script applies:
# 1. fn → def (function definitions only)
# 2. from memory import → from std.memory import
# 3. from collections import → from std.collections import
# 4. from algorithm import → from std.algorithm import
# 5. from itertools import → from std.itertools import

set -e

BASE_DIR="/home/mvillmow/Odyssey"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: $BASE_DIR does not exist"
    exit 1
fi

echo "Finding all .mojo files..."
MOJO_FILES=$(find "$BASE_DIR" -type f -name "*.mojo" ! -path '*/__pycache__/*' ! -path '*/.git/*')
FILE_COUNT=$(echo "$MOJO_FILES" | wc -l)
echo "Found $FILE_COUNT .mojo files"
echo

MODIFIED=0
for file in $MOJO_FILES; do
    CHANGED=0

    # 1. Replace fn → def at function definitions
    if grep -q "^fn \|^    fn \|^        fn \|^            fn " "$file"; then
        sed -i 's/^\([[:space:]]*\)fn \([a-zA-Z_]\)/\1def \2/g' "$file"
        CHANGED=1
    fi

    # 2. Replace from memory import → from std.memory import
    if grep -q "from memory import" "$file"; then
        sed -i 's/from memory import/from std.memory import/g' "$file"
        CHANGED=1
    fi

    # 3. Replace from collections import → from std.collections import
    if grep -q "from collections import" "$file"; then
        sed -i 's/from collections import/from std.collections import/g' "$file"
        CHANGED=1
    fi

    # 4. Replace from algorithm import → from std.algorithm import
    if grep -q "from algorithm import" "$file"; then
        sed -i 's/from algorithm import/from std.algorithm import/g' "$file"
        CHANGED=1
    fi

    # 5. Replace from itertools import → from std.itertools import
    if grep -q "from itertools import" "$file"; then
        sed -i 's/from itertools import/from std.itertools import/g' "$file"
        CHANGED=1
    fi

    if [ $CHANGED -eq 1 ]; then
        MODIFIED=$((MODIFIED + 1))
        REL_PATH="${file#$BASE_DIR/}"
        echo "✓ $REL_PATH"
    fi
done

echo
echo "========================================================================"
echo "Fixed $MODIFIED files"
echo "========================================================================"
