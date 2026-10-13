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
# checkpoint at shutdown, and at least 1.5x the matrix file in RAM. More is
# needed when the card is small, because blocks streamed from host memory
# are pinned there. Once msieve has planned its GPU layout, each run's peak
# (matrix, pinned streamed blocks and host vectors) is estimated. A run
# that would not fit with 2 GB to spare is stopped before it pins anything,
# and a run the kernel kills for lack of memory anyway is also reported as
# NORAM. Available RAM is the smaller of MemAvailable and the headroom under
# any cgroup (container) memory limit.
#
# Example: tools/tune_la.sh -s "7 14" -v 128 -T 60
# Results: DIR/results.tsv (one row per run) and DIR/report.txt. Each run
# records msieve's peak resident memory and the host memory it took (the
# largest drop in available RAM during the run, which includes pinned
# buffers; other programs' use shows up there too). The run directories
# under DIR/job/bench keep each msieve log; the matrices are deleted after
# each size, or when the tune is interrupted, unless -k.

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
# the current run's msieve, identified by its working directory and run
# directory, so other tunes and msieve jobs are never touched
run_pids() {
    local p
    [ -n "$run" ] || return
    for p in /proc/[0-9]*; do
        [ "$p/cwd" -ef "$out/job" ] &&
            tr '\0' '\n' 2>/dev/null < "$p/cmdline" | grep -qxF -- "$run/msieve.dat" &&
            echo "${p#/proc/}"
    done
}
stop_run() {
    local p
    for p in $(run_pids); do kill -"$1" "$p" 2>/dev/null; done
}

# This process's cgroup directory: cgroup v2, else the v1 memory hierarchy.
# Prints "DIR ROOT V" (V = 2 or 1), nothing if neither is readable.
cg_dir() {
    local p
    p=$(awk -F: '$1 == "0" && $2 == "" { print $3; exit }' /proc/self/cgroup 2>/dev/null)
    if [ -n "$p" ] && [ -r /sys/fs/cgroup/cgroup.controllers ]; then
        echo "/sys/fs/cgroup${p%/} /sys/fs/cgroup 2"
        return
    fi
    p=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ { print $3; exit }' /proc/self/cgroup 2>/dev/null)
    [ -n "$p" ] && [ -d /sys/fs/cgroup/memory ] &&
        echo "/sys/fs/cgroup/memory${p%/} /sys/fs/cgroup/memory 1"
}
# MB that can still be used under cgroup memory limits (the tightest of
# this cgroup and its ancestors, page cache counted as free); empty if
# there is no limit. In a container the root is the container's cgroup.
cg_headroom() {
    local d root v lim cur rec h best=""
    read -r d root v <<< "$(cg_dir)"
    [ -n "$d" ] || return
    while :; do
        if [ "$v" = 2 ]; then
            lim=$(cat "$d/memory.max" 2>/dev/null)
            [ "$lim" = max ] && lim=""
            cur=$(cat "$d/memory.current" 2>/dev/null)
            rec=$(awk '$1 == "active_file" || $1 == "inactive_file" { s += $2 } END { print s + 0 }' \
                "$d/memory.stat" 2>/dev/null)
        else
            lim=$(cat "$d/memory.limit_in_bytes" 2>/dev/null)
            # no limit reads as about 2^63
            [ -n "$lim" ] && [ "${#lim}" -ge 19 ] && lim=""
            cur=$(cat "$d/memory.usage_in_bytes" 2>/dev/null)
            rec=$(awk '$1 == "total_active_file" || $1 == "total_inactive_file" { s += $2 } END { print s + 0 }' \
                "$d/memory.stat" 2>/dev/null)
        fi
        if [ -n "$lim" ] && [ -n "$cur" ]; then
            h=$(( (lim - cur + rec) / 1048576 ))
            if [ -z "$best" ] || [ "$h" -lt "$best" ]; then best=$h; fi
        fi
        [ "$d" = "$root" ] || [ "${d#"$root"/}" = "$d" ] && break
        d=${d%/*}
    done
    echo "$best"
}
# MB of RAM available: MemAvailable, or less under a cgroup limit
mem_avail() {
    local a h
    a=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo)
    h=$(cg_headroom)
    if [ -n "$h" ] && [ "$h" -lt "$a" ]; then a=$h; fi
    echo "$a"
}
# MB msieve keeps free on the card while streaming, for nc2 arguments $1
gpu_margin() {
    case " $1 " in
        *" max_gpu_mem="*) echo 0 ;;
        *) if [ -e /dev/dxg ]; then echo 1536; else echo 256; fi ;;
    esac
}
# Peak host memory of a run in MB, once msieve has logged its GPU layout
# ("GPU memory:"); nothing before that. Prints "TOTAL MATRIX STREAMED
# VECTORS": the matrix as loaded, the blocks it will pin in host memory
# (what does not fit in the GPU budget once up to three staging buffers
# are set aside), and host vectors and work space, which measured 9.6-10.2
# bytes per column per VBITS/8. Calibrated on the 5070 at 25M and 35M
# (LANCZOS_OPTIMIZATION_NOTES.md, 2026-10-12), where it ran 0.5-1 GB high.
# Arguments: LOG VBITS MARGIN_MB
ram_estimate() {
    awk -v vb="$2" -v margin="$3" '
        /matrix is [0-9]+ x [0-9]+ \(/ && m == "" {
            s = $0; sub(/ MB\).*/, "", s); sub(/.*\(/, "", s); m = s + 0
            s = $0; sub(/.* x /, "", s); nc = s + 0 }
        /nonzeros per matrix block:/ { bnz = $NF + 0 }
        /GPU memory:/ { g = $0 }
        END {
            if (g == "" || m == "") exit
            s = g; sub(/.*vectors, /, "", s); avail = s + 0
            s = g; sub(/.*blocks \(/, "", s); need = s + 0
            streamed = 0
            if (need > avail) {
                budget = avail > margin ? avail - margin : 0
                streamed = need - budget + 3 * 1.1 * (bnz * 4 + nc * 4) / 1048576
                if (streamed > need) streamed = need
            }
            vec = 10.5 * nc * vb / 8 / 1048576
            printf "%d %d %d %d\n", m + streamed + vec, m, streamed, vec
        }' "$1" 2>/dev/null
}
# processes killed for lack of memory so far: this cgroup's count (which
# includes its descendants), else the whole system's; empty if unknown
oom_count() {
    local d root v n=""
    read -r d root v <<< "$(cg_dir)"
    if [ "$v" = 2 ]; then
        n=$(awk '$1 == "oom_kill" { print $2 }' "$d/memory.events" 2>/dev/null)
        # a container may only show its own cgroup, at the root
        [ -n "$n" ] || n=$(awk '$1 == "oom_kill" { print $2 }' "$root/memory.events" 2>/dev/null)
    elif [ "$v" = 1 ]; then
        n=$(awk '$1 == "oom_kill" { print $2 }' "$d/memory.oom_control" 2>/dev/null)
    fi
    [ -n "$n" ] || n=$(awk '$1 == "oom_kill" { print $2 }' /proc/vmstat 2>/dev/null)
    echo "$n"
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
    h=$(cg_headroom)
    [ -n "$h" ] && echo "cgroup memory limit: $h MB still free under it"
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
        # tracked files, less any deleted from this checkout
        ( set -o pipefail
          cd "$repo" && git ls-files -z | while IFS= read -r -d '' f; do
              if [ -e "$f" ] || [ -L "$f" ]; then printf '%s\0' "$f"; fi
          done | tar --null -T - -cf - | tar -xf - -C "$b" ) || {
            log "copying the source tree failed"; exit 1; }
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
if [ ! -f "$results" ]; then
    printf "size\tncols\tweight\tvbits\tsetting\tstatus\tdims_per_sec\teta_full\tms_per_iter\tsparse_mb\tstreamed_mb\tcopied_mib_per_iter\tpeak_vram_mib\tbase_vram_mib\trun_dir\tpeak_rss_mib\thost_mem_mib\n" > "$results"
elif ! head -1 "$results" | grep -q host_mem_mib; then
    # from before the memory columns
    sed -i '1s/$/\tpeak_rss_mib\thost_mem_mib/' "$results"
fi
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
    avail=$(mem_avail)
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
            printf "%s\t%s\t%s\t%s\t%s\tSKIP\t-\t-\t-\t-\t-\t-\t-\t-\t%s\t-\t-\n" "$label" "$ncols" "$weight" "$v" "$c" "$reason" >> "$results"
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
            oom0=$(oom_count) avail0=$(mem_avail)
            (cd "$out/job" && "$repo/bench_la.sh" -d "$label" -b "${BIN[$v]}" \
                -a "${SETTING[$c]}" -T "$window" -W "$warmup" -g "$gpu" \
                -n "$name" -- -t "$threads") >> "$out/runs.log" 2>&1 &
            bpid=$!
            # with no layout that fits, msieve tries to load the whole
            # matrix; under WDDM that pages GPU memory instead of failing.
            # The log is checked once more after the run has ended.
            nofit=0 noram="" est="" run="" mpid="" rss=0 amin=$avail0
            margin=$(gpu_margin "${SETTING[$c]}")
            while :; do
                alive=0; kill -0 $bpid 2>/dev/null && alive=1
                [ -n "$run" ] || run=$(run_dirs "$name" | grep -vxF -e "$before" | tail -1)
                # peak memory: msieve's own (VmHWM), and the drop in
                # available RAM, which also counts its pinned buffers
                if [ -z "$mpid" ] || [ ! -d "/proc/$mpid" ]; then mpid=$(run_pids | head -1); fi
                if [ -n "$mpid" ]; then
                    h=$(awk '/^VmHWM:/ { print int($2 / 1024) }' "/proc/$mpid/status" 2>/dev/null)
                    if [ -n "$h" ] && [ "$h" -gt "$rss" ]; then rss=$h; fi
                fi
                a=$(mem_avail)
                if [ "$a" -lt "$amin" ]; then amin=$a; fi
                if [ -n "$run" ] && grep -qs 'cannot stream the matrix' "$out/job/$run/msieve.log"; then
                    nofit=1
                    stop_run KILL
                    break
                fi
                # once msieve has planned its GPU layout, and before it
                # pins the streamed blocks, stop a run whose peak would not
                # fit in the RAM that was available when it started, with
                # 2 GB to spare (the drop in available RAM ran up to 1.2 GB
                # above the estimate in calibration)
                if [ -z "$est" ] && [ -n "$run" ]; then
                    read -r est est_m est_s est_v <<< \
                        "$(ram_estimate "$out/job/$run/msieve.log" "$v" "$margin")"
                    if [ -n "$est" ] && [ $(( est + 2048 )) -gt "$avail0" ]; then
                        # a layout that cannot fit on the card at all is NOFIT
                        sleep 1
                        if grep -qs 'cannot stream the matrix' "$out/job/$run/msieve.log"; then
                            nofit=1
                        else
                            noram="needs ~$est MB at its peak (matrix $est_m, streamed blocks $est_s, vectors $est_v), $avail0 MB available"
                        fi
                        stop_run KILL
                        break
                    fi
                fi
                [ $alive = 1 ] || break
                sleep 1
            done
            wait $bpid
            host_mem=$(( avail0 - amin ))
            [ "$host_mem" -ge 0 ] || host_mem=0
            oom1=$(oom_count)
            # this run's row, matched by name and run directory
            row=""
            [ -n "$run" ] && row=$(awk -F'\t' -v n="$name" -v d="$run" '$2 == n && $13 == d' \
                "$out/job/bench/results.tsv" 2>/dev/null | tail -1)
            [ -n "$row" ] ||
                row=$(printf 'x\t%s\tx\tx\tFAIL\t-\t-\t-\t-\t-\t-\tx\t%s\t-\t-' "$name" "${run:--}")
            st=$(printf '%s\n' "$row" | cut -f5)
            [ "$nofit" = 1 ] && st=NOFIT
            if [ -n "$noram" ]; then
                log "  stopped before pinning host memory: $noram"
                st=NORAM
            fi
            # killed for lack of host memory (by the kernel, not this script)
            if [ -n "$oom0" ] && [ -n "$oom1" ] && [ "$oom1" -gt "$oom0" ]; then
                log "  the kernel killed $(( oom1 - oom0 )) process(es) for lack of memory during this run"
                [ "$st" = FAIL ] && st=NORAM
            fi
            printf '%s\n' "$row" | awk -F'\t' -v OFS='\t' -v lab="$label" -v nc="$ncols" \
                -v w="$weight" -v vb="$v" -v c="$c" -v st="$st" -v base="$out/job/" \
                -v rss="$rss" -v hm="$host_mem" '
                { print lab, nc, w, vb, c, st, $6, $7, $8, $9, $14, $15, $10, $11,
                        ($13 == "-" ? "-" : base $13), (rss > 0 ? rss : "-"), hm }' >> "$results"
            rate=$(printf '%s\n' "$row" | cut -f6); eta=$(printf '%s\n' "$row" | cut -f7)
            log "  $st: $rate dims/s, full run $eta, host memory ~$host_mem MB (msieve resident $rss MB${est:+, estimated $est MB})"
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
        line[key] = line[key] sprintf("%s\t%s\t%s\t%s\t%s\t%s\n", $7 == "-" ? 0 : $7, $4, $5, $6, $8,
                                      $17 == "" ? "-" : $17)
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
          printf "  VBITS %-4s %-14s %-5s %9s dims/s  full run %-7s host RAM %6s MB%s\n", $2, $3, $4,
                 ($1 > 0 ? $1 : "-"), $5, $6, mark }'
} > "$out/report.txt"
log "done; report in $out/report.txt"
cat "$out/report.txt"
