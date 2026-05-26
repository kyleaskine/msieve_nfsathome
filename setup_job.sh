#!/bin/bash

gz=$(ls *.gz 2>/dev/null | head -1)
fb=$(ls *.fb 2>/dev/null | head -1)
ini=$(ls *.ini 2>/dev/null | head -1)

if [ -z "$gz" ] || [ -z "$fb" ] || [ -z "$ini" ]; then
    echo "Error: could not find all three required files (.gz, .fb, .ini)"
    [ -z "$gz" ]  && echo "  missing: .gz"
    [ -z "$fb" ]  && echo "  missing: .fb"
    [ -z "$ini" ] && echo "  missing: .ini"
    exit 1
fi

echo "Found: $gz, $fb, $ini"

mv "$gz"  msieve.dat.gz
mv "$fb"  msieve.fb
mv "$ini" worktodo.ini

echo "Decompressing msieve.dat.gz..."
gzip -d msieve.dat.gz

find . -name "*:Zone.Identifier" -delete
echo "Cleaned up Zone.Identifier files"

echo "Done."
