#!/bin/bash
# Time GPU Lanczos settings on synthetic matrices, to compare cards and
# settings without real job files.
#
# Builds msieve at each VBITS from this source tree, generates matrices with
# the structure of a real NFS matrix (tools/synthmat.c, scaled from the
# profile of a C168 TD 110 matrix in tools/la_profile_td110.txt) and times
# each setting with bench_la.sh: warmup, a measurement window, then SIGTERM,
# which runs the Lanczos integrity check. Everything stays in one work
# directory; no job files are read or written.
#
# Usage:   tools/tune_la.sh [options]
#   -s SIZES  matrix sizes in millions of columns, each optionally SIZE@W
#             with W the sparse nonzeros per column in the matrix file
#             (default: the profile's 103.6, about TD 110; TD 70/90/130
#             give about 68/86/122). Default "7 14 25 35 45"
#   -v LIST   VBITS values (default "128 256")
#   -c LIST   settings (default "single single-col24 two two-col24")
#   -A N=ARGS add setting N with -nc2 arguments ARGS (repeatable)
#   -T SECS   measurement window per run (default 90)
#   -W SECS   warmup per run (default 30)
#   -g N      GPU index (default 0)
#   -t N      msieve threads (default 8)
#   -o DIR    work directory (default bench/tune-<host>-<stamp>)
#   -B DIR    use DIR/v<VBITS>/msieve instead of building
#   -k        keep the generated matrices
#
# Sizes that need more RAM or disk than is available are skipped, and a
# run whose matrix cannot fit on the card even when streamed is stopped and
# reported as NOFIT. At the default density a size needs about 0.45 GB of
# disk per million columns for the matrix (45M columns: ~20 GB) plus a
# checkpoint at shutdown, and about 1.5x the matrix file in RAM.
#
# Example: tools/tune_la.sh -s "7 14" -v 128 -T 60
# Results: DIR/results.tsv (one row per run) and DIR/report.txt. The run
# directories under DIR/job/bench keep each msieve log; the matrices are
# deleted after each size, or when the tune is interrupted, unless -k.

sizes="7 14 25 35 45"
vbits="128 256"
configs="single single-col24 two two-col24"
window=90
warmup=30
gpu=0
threads=8
out=""
bindir=""
keep=0
extra=""
declare -A SETTING=(
    [single]="single_copy=1"
    [single-col24]="single_copy=1 col24=1"
    [two]=""
    [two-col24]="col24=1"
)

while getopts "s:v:c:A:T:W:g:t:o:B:kh" opt; do
    case $opt in
        s) sizes=$OPTARG ;;
        v) vbits=$OPTARG ;;
        c) configs=$OPTARG ;;
        A) SETTING[${OPTARG%%=*}]=${OPTARG#*=}
           extra="$extra ${OPTARG%%=*}" ;;
        T) window=$OPTARG ;;
        W) warmup=$OPTARG ;;
        g) gpu=$OPTARG ;;
        t) threads=$OPTARG ;;
        o) out=$OPTARG ;;
        B) bindir=$(realpath -- "$OPTARG") ;;
        k) keep=1 ;;
        *) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 1 ;;
    esac
done
configs="$configs$extra"
for c in $configs; do
    [ -n "${SETTING[$c]+x}" ] || { echo "error: unknown setting $c"; exit 1; }
done

repo=$(cd "$(dirname "$0")/.." && pwd)
profile=$repo/tools/la_profile_td110.txt
host=$(hostname -s)
out=${out:-$repo/bench/tune-$host-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out/job" || exit 1
out=$(cd "$out" && pwd)
log() { echo "$(date +%T) $*" | tee -a "$out/tune.log"; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# bench_la.sh's run directories (bench/NAME-YYYYMMDD-HHMMSS) for setting
# name $1, relative to the job directory
stamp_glob='[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]'
run_dirs() {
    (cd "$out/job" && for d in bench/"$1"-$stamp_glob; do [ -d "$d" ] && echo "$d"; done)
}
# send signal $1 to the current run's msieve, identified by its working
# directory and run directory, so other tunes and msieve jobs are untouched
stop_run() {
    local p
    [ -n "$run" ] || return
    for p in /proc/[0-9]*; do
        [ "$p/cwd" -ef "$out/job" ] &&
            tr '\0' '\n' 2>/dev/null < "$p/cmdline" | grep -qxF -- "$run/msieve.dat" &&
            kill -"$1" "${p#/proc/}" 2>/dev/null
    done
}
run="" partial="" cur_mat=""
on_exit() {
    stop_run TERM
    [ -z "$partial" ] || rm -f "$partial"
    [ "$keep" = 1 ] || [ -z "$cur_mat" ] || rm -f "$cur_mat"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
    echo "host: $host"
    date
    echo "source: $(git -C "$repo" describe --always --dirty 2>/dev/null)"
    nvidia-smi -i "$gpu" --query-gpu=name,memory.total,memory.used,driver_version,compute_cap,pcie.link.gen.max,pcie.link.width.max --format=csv
    if [ -e /dev/dxg ]; then echo "WDDM: yes (/dev/dxg)"; else echo "WDDM: no"; fi
    grep -E 'MemTotal|MemAvailable' /proc/meminfo
    lscpu | grep -E 'Model name|^CPU\(s\)'
    uname -r
    echo "sizes: $sizes; VBITS: $vbits; window ${window}s, warmup ${warmup}s, -t $threads"
    for c in $configs; do echo "setting $c: ${SETTING[$c]:-(defaults)}"; done
} > "$out/environment.txt" 2>&1
cat "$out/environment.txt"

# binaries
declare -A BIN
arch=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
[ -d /usr/lib/wsl/lib ] && export LIBRARY_PATH=/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
for v in $vbits; do
    if [ -n "$bindir" ]; then
        BIN[$v]=$bindir/v$v/msieve
    else
        b=$out/build/v$v
        mkdir -p "$b"
        (cd "$repo" && git ls-files -z | tar --null -T - -cf -) | tar -xf - -C "$b" || exit 1
        log "building VBITS=$v for sm_$arch"
        (cd "$b" && make -j"$(nproc)" all CUDA="$arch" VBITS="$v") > "$b/build.log" 2>&1 || {
            log "build failed, see $b/build.log"; exit 1; }
        BIN[$v]=$b/msieve
    fi
    [ -x "${BIN[$v]}" ] || { log "error: no executable ${BIN[$v]}"; exit 1; }
done
gcc -O2 -fopenmp -o "$out/synthmat" "$repo/tools/synthmat.c" -lm || exit 1

# an LA-only job: the polynomials are never used, only parsed
n=1167984798111281975972139931059274579165801700195500732513291383783133049588151975645370374287852614884146888067442512219413748768010657572575384986457405973985247465176041951676954461208131403777
printf '%s\n' "$n" > "$out/job/worktodo.ini"
printf 'N %s\nSKEW 1\nR0 -2\nR1 1\nA0 %s\nA1 0\nA2 0\nA3 1\n' "$n" \
    1167984798111281975972139931059274579165801700195500732513291383783133049588151975645370374287852614884146888067442512219413748768010657572575384986457405973985247465176041951676954461208131403769 \
    > "$out/job/msieve.fb"

results=$out/results.tsv
[ -f "$results" ] || printf "size\tncols\tweight\tvbits\tsetting\tstatus\tdims_per_sec\teta_full\tms_per_iter\tsparse_mb\tstreamed_mb\tcopied_mib_per_iter\tpeak_vram_mib\tbase_vram_mib\trun_dir\n" > "$results"
gthreads=$(nproc); [ "$gthreads" -gt 16 ] && gthreads=16
pweight=$(awk 'NR == 2 { printf "%.2f", $4 / $3 }' "$profile")

for spec in $sizes; do
    m=${spec%@*}
    w=""; [ "$spec" != "$m" ] && w=${spec#*@}
    weight=${w:-$pweight}
    ncols=$(awk -v m="$m" 'BEGIN { printf "%d", m * 1e6 + 0.5 }')
    label=${m}M${w:+w$w}
    read -r file_mb gen_mb run_mb <<< "$(awk -v c="$ncols" -v w="$weight" -v t="$gthreads" 'BEGIN {
        file = (c * w * 4 + c * 20) / 1048576
        printf "%d %d %d", file, (c * w * 4 + c * (4 * t + 64)) / 1048576, file * 1.5 }')"
    avail=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo)
    disk=$(df -Pm "$out" | awk 'NR == 2 { print $4 }')
    need=$(( gen_mb > run_mb ? gen_mb : run_mb ))
    reason=""
    [ "$need" -gt "$avail" ] && reason="needs ~$need MB RAM, $avail MB available"
    [ $(( file_mb * 3 / 2 )) -gt "$disk" ] &&
        reason="${reason:+$reason; }needs ~$(( file_mb * 3 / 2 )) MB disk, $disk MB free"
    mat=$out/job/msieve.dat.mat.$label
    if [ -z "$reason" ] && [ ! -s "$mat" ]; then
        log "generating $label: $ncols columns, $weight nonzeros/column (~$file_mb MB)"
        # synthmat writes $mat.part and renames it when complete
        cur_mat=$mat partial=$mat.part
        OMP_NUM_THREADS=$gthreads "$out/synthmat" gen "$profile" "$mat" "$ncols" "${w:-0}" 1 \
            >> "$out/tune.log" 2>&1 || { reason="matrix generation failed"; rm -f "$partial"; }
        partial=""
    fi
    if [ -n "$reason" ]; then
        log "skipping $label: $reason"
        for v in $vbits; do for c in $configs; do
            printf "%s\t%s\t%s\t%s\t%s\tSKIP\t-\t-\t-\t-\t-\t-\t-\t-\t%s\n" "$label" "$ncols" "$weight" "$v" "$c" "$reason" >> "$results"
        done; done
        continue
    fi
    cur_mat=$mat
    for v in $vbits; do
        for c in $configs; do
            name=$label-v$v-$c
            log "$label VBITS=$v $c: ${SETTING[$c]:-(defaults)}"
            # run directories left by an earlier tune in this -o
            before=$(run_dirs "$name")
            (cd "$out/job" && "$repo/bench_la.sh" -d "$label" -b "${BIN[$v]}" \
                -a "${SETTING[$c]}" -T "$window" -W "$warmup" -g "$gpu" \
                -n "$name" -- -t "$threads") >> "$out/runs.log" 2>&1 &
            bpid=$!
            # with no layout that fits, msieve tries to load the whole
            # matrix; under WDDM that pages GPU memory instead of failing.
            # The log is checked once more after the run has ended.
            nofit=0 run=""
            while :; do
                alive=0; kill -0 $bpid 2>/dev/null && alive=1
                [ -n "$run" ] || run=$(run_dirs "$name" | grep -vxF -e "$before" | tail -1)
                if [ -n "$run" ] && grep -qs 'cannot stream the matrix' "$out/job/$run/msieve.log"; then
                    nofit=1
                    stop_run KILL
                    break
                fi
                [ $alive = 1 ] || break
                sleep 2
            done
            wait $bpid
            # this run's row, matched by name and run directory
            row=""
            [ -n "$run" ] && row=$(awk -F'\t' -v n="$name" -v d="$run" '$2 == n && $13 == d' \
                "$out/job/bench/results.tsv" 2>/dev/null | tail -1)
            [ -n "$row" ] ||
                row=$(printf 'x\t%s\tx\tx\tFAIL\t-\t-\t-\t-\t-\t-\tx\t%s\t-\t-' "$name" "${run:--}")
            st=$(printf '%s\n' "$row" | cut -f5)
            [ "$nofit" = 1 ] && st=NOFIT
            printf '%s\n' "$row" | awk -F'\t' -v OFS='\t' -v lab="$label" -v nc="$ncols" \
                -v w="$weight" -v vb="$v" -v c="$c" -v st="$st" -v base="$out/job/" '
                { print lab, nc, w, vb, c, st, $6, $7, $8, $9, $14, $15, $10, $11,
                        ($13 == "-" ? "-" : base $13) }' >> "$results"
            rate=$(printf '%s\n' "$row" | cut -f6); eta=$(printf '%s\n' "$row" | cut -f7)
            log "  $st: $rate dims/s, full run $eta"
        done
    done
    [ "$keep" = 1 ] || rm -f "$mat"
    cur_mat=""
done

# report: per size, every run sorted by speed
{
    cat "$out/environment.txt"
    echo
    awk -F'\t' 'NR > 1 {
        key = $1; if (!(key in seen)) { seen[key] = 1; order[++n] = key }
        line[key] = line[key] sprintf("%s\t%s\t%s\t%s\t%s\n", $7 == "-" ? 0 : $7, $4, $5, $6, $8)
    } END {
        for (i = 1; i <= n; i++) {
            printf "size %s\n", order[i]
            fflush()
            cmd = "sort -t\"\t\" -k1,1gr"
            printf "%s", line[order[i]] | cmd
            close(cmd)
        }
    }' "$results" | awk -F'\t' '/^size/ { print; best = ""; next }
        { if (best == "" && $1 > 0) { best = $1; mark = " <- best" } else mark = ""
          printf "  VBITS %-4s %-14s %-5s %9s dims/s  full run %s%s\n", $2, $3, $4,
                 ($1 > 0 ? $1 : "-"), $5, mark }'
} > "$out/report.txt"
log "done; report in $out/report.txt"
cat "$out/report.txt"
