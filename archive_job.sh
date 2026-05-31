#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <density>"
    exit 1
fi

density=$1

mv "msieve.dat.cyc.${density}" msieve.dat.cyc
mv "msieve.dat.mat.${density}" msieve.dat.mat

tar -cf "archive_${density}.tar.zst" -I 'zstd -22 --ultra -T0' \
    msieve.dat.cyc msieve.dat.mat msieve.fb msieve.log worktodo.ini
