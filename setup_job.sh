#!/bin/bash

# Ignore files a previous run already renamed, so a leftover msieve.fb can't be
# mistaken for the job's own .fb (which would put "msieve" in jobname.txt).
gz=$(ls *.gz 2>/dev/null  | grep -vx 'msieve\.dat\.gz' | head -1)
fb=$(ls *.fb 2>/dev/null  | grep -vx 'msieve\.fb'       | head -1)
ini=$(ls *.ini 2>/dev/null | grep -vx 'worktodo\.ini'   | head -1)

if [ -z "$gz" ] || [ -z "$fb" ] || [ -z "$ini" ]; then
    echo "Error: could not find all three required files (.gz, .fb, .ini)"
    [ -z "$gz" ]  && echo "  missing: .gz"
    [ -z "$fb" ]  && echo "  missing: .fb"
    [ -z "$ini" ] && echo "  missing: .ini"
    exit 1
fi

echo "Found: $gz, $fb, $ini"

# Remember the NFS@Home job name (e.g. C170_2664_1354) before we rename
# everything to msieve.*; report_job.sh reads it back out.
jobname="${fb%.fb}"
echo "$jobname" > jobname.txt
echo "Job name: $jobname (saved to jobname.txt)"

mv "$gz"  msieve.dat.gz
mv "$fb"  msieve.fb
mv "$ini" worktodo.ini

echo "Decompressing msieve.dat.gz..."
gzip -d msieve.dat.gz

find . -name "*:Zone.Identifier" -delete
echo "Cleaned up Zone.Identifier files"

echo "Done."
