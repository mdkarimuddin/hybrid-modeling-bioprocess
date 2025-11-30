#!/bin/bash
# Test script to check module availability
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    echo "Module system initialized"
    module avail python 2>&1 | head -5
else
    echo "Module system not found at /etc/profile.d/modules.sh"
fi
