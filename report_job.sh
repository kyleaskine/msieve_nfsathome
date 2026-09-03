#!/usr/bin/env bash
#
# report_job.sh -- build the NFS@Home forum post for a finished job and
#                  upload msieve.log to pastebin.
#
# Usage:  ./report_job.sh [JOBNAME] [options]
#
#   JOBNAME        e.g. C170_2664_1354.  If omitted, read from ./jobname.txt
#                  (written by setup_job.sh), else the directory name.
#   -f FILE        log file to parse/paste (default msieve.log).  May be given
#                  more than once; files are concatenated in order -- use this
#                  when the Lanczos run happened on a remote box and you have
#                  its log separately.
#   -g NAME        override the GPU name (default: parsed from the log)
#   -n             don't upload anything, just print the post
#   -t             test the pastebin credentials (login only, no paste) and exit
#   -u URL         use this pastebin URL instead of uploading
#   -h             help
#
# Pastebin credentials (optional) live in ~/.pastebin_credentials or
# ./.pastebin_credentials, mode 600.  The file is parsed, not sourced: write the
# values literally, no shell escaping, no spaces around the '='.
#
#   PASTEBIN_API_KEY=<dev key from https://pastebin.com/doc_api>
#   PASTEBIN_USER=<your pastebin username>      # omit for an anonymous paste
#   PASTEBIN_PASS=<your pastebin password>
#
# or, if you'd rather not store the password, the long-lived key that
# api_login.php hands back:
#
#   PASTEBIN_USER_KEY=<user key>
#
# With user+pass the paste is unlisted and owned by your account; with only the
# dev key it is an unlisted guest paste; with neither, upload is skipped and the
# post is printed with a placeholder.

set -uo pipefail

LOGS=()
GPU_OVERRIDE=""
NO_PASTE=0
TEST_LOGIN=0
PASTE_URL=""
JOB=""

usage() { awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"; exit 0; }

need_arg() { [ "$1" -ge 2 ] || { echo "$2 needs an argument (see -h)" >&2; exit 1; }; }

while [ $# -gt 0 ]; do
    case "$1" in
        -f) need_arg $# -f; LOGS+=("$2"); shift 2 ;;
        -g) need_arg $# -g; GPU_OVERRIDE="$2"; shift 2 ;;
        -u) need_arg $# -u; PASTE_URL="$2"; shift 2 ;;
        -n) NO_PASTE=1; shift ;;
        -t) TEST_LOGIN=1; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; exit 1 ;;
        *)  JOB="$1"; shift ;;
    esac
done

[ ${#LOGS[@]} -eq 0 ] && LOGS=(msieve.log)
for f in "${LOGS[@]}"; do
    [ -r "$f" ] || { echo "cannot read $f" >&2; exit 1; }
done

# ---- job name -------------------------------------------------------------
if [ -z "$JOB" ]; then
    if [ -r jobname.txt ]; then
        JOB=$(tr -d ' \t\n\r' < jobname.txt)
    else
        JOB=$(basename "$PWD")
        echo "note: no jobname.txt and no name given, using '$JOB'" >&2
    fi
fi

# ---- assemble the log we parse and paste ----------------------------------
LOG=$(mktemp); trap 'rm -f "$LOG"' EXIT
cat "${LOGS[@]}" > "$LOG"

# ---- parse ----------------------------------------------------------------
# The log accumulates every msieve invocation, so anchor on the LAST linear
# algebra run that actually iterated (a bare all_matbuild=1 run also emits a
# BLanczosTime, and multi-density runs emit several "matrix is" lines).
eval "$(awk '
    function num(s) { gsub(/[^0-9]/, "", s); return s+0 }
    # first number in s, decimal point kept ("to 100.0" -> 100)
    function fnum(s) { match(s, /[0-9]+(\.[0-9]+)?/); return substr(s, RSTART, RLENGTH) + 0 }

    /commencing linear algebra/          { la = NR }
    /commencing Lanczos iteration/       { la_final = la }

    # relation counts (first occurrence: the filtering run)
    /hash collisions in .* relations/ && !total {
        for (i = 1; i <= NF; i++) if ($i == "in") total = num($(i+1))
    }
    /duplicates and .* unique relations/ && !uniq {
        for (i = 1; i <= NF; i++) if ($i == "and") uniq = num($(i+1))
    }

    { line[NR] = $0 }
    END {
        n = NR
        start = (la_final ? la_final : (la ? la : 1))
        # matrix / gpu / lanczos time: first match after the final LA start
        for (i = start; i <= n; i++) {
            s = line[i]
            if (!rows && s ~ /matrix is [0-9]+ x [0-9]+/) {
                match(s, /matrix is [0-9]+/); rows = num(substr(s, RSTART, RLENGTH))
            }
            if (!gpu && s ~ /using GPU [0-9]+ \(/) {
                gpu = substr(s, index(s, "(") + 1); sub(/\).*$/, "", gpu)
            }
            if (!dens && s ~ /selecting density [0-9]+/) {       # select_density=N
                match(s, /density [0-9]+/); dens = fnum(substr(s, RSTART, RLENGTH))
            }
            if (s ~ /BLanczosTime: [0-9]+/) { match(s, /: [0-9]+/); lt = num(substr(s, RSTART, RLENGTH)) }
        }

        # No select_density line.  If this was a multi-density build, identify the
        # density by the matrix that all_matbuild=1 logged with our row count:
        #   "building matrix for density N" ... "matrix is <rows> x ..."
        if (!dens && rows) {
            for (i = 1; i <= n; i++) {
                if (line[i] ~ /building matrix for density [0-9]+/) {
                    match(line[i], /density [0-9]+/); d = fnum(substr(line[i], RSTART, RLENGTH))
                } else if (d && line[i] ~ /matrix is [0-9]+ x [0-9]+/) {
                    match(line[i], /matrix is [0-9]+/)
                    if (num(substr(line[i], RSTART, RLENGTH)) == rows) { dens = d; break }
                }
            }
        }
        # Ordinary single-density job: filtering logs it, before the LA run.
        if (!dens) {
            for (i = n; i >= 1; i--)
                if (line[i] ~ /target matrix density to [0-9.]+/) {
                    match(line[i], /to [0-9.]+/); dens = fnum(substr(line[i], RSTART, RLENGTH)); break
                }
        }
        # Last resort: the highest density the merge attempted, which is not
        # necessarily the one the matrix was built at -- flag it as a guess.
        if (!dens) {
            for (i = n; i >= 1; i--)
                if (line[i] ~ /trying target density [0-9.]+/) {
                    match(line[i], /density [0-9.]+/); dens = fnum(substr(line[i], RSTART, RLENGTH))
                    guess = 1; break
                }
        }

        # factors: the trailing block.  "c" = composite cofactor (driver.c), i.e.
        # the number is NOT fully factored -- keep the line and say so.
        f = ""
        for (i = n; i >= 1; i--) {
            if (line[i] ~ / (p|c|prp)[0-9]+ factor: [0-9]+/) {
                s = line[i]; sub(/^.*  /, "", s)
                if (s ~ /^c[0-9]+ /) composite = 1
                f = (f == "" ? s : s "\n" f)
            } else if (f != "") break
        }
        gsub(/\n/, "\\n", f); gsub(/"/, "", f)

        printf "TOTAL=%d\nUNIQ=%d\nROWS=%d\nDENS=%d\nLANTIME=%d\nGPU=\"%s\"\nFACTORS=\"%s\"\nDENS_GUESS=%d\nCOMPOSITE=%d\n",
               total, uniq, rows, dens, lt, gpu, f, guess, composite
    }
' "$LOG")"

# ---- format ---------------------------------------------------------------
fmt_m()  { awk -v n="$1" 'BEGIN { printf "%.0fM", n/1e6 }'; }
fmt_mat(){ awk -v n="$1" 'BEGIN { printf "%.1fM", n/1e6 }'; }
fmt_hm() { awk -v s="$1" 'BEGIN { h=int(s/3600); m=int((s%3600)/60)
                                  if (h) printf "%dh%02dm", h, m; else printf "%dm", m }'; }

# "NVIDIA GeForce RTX 5070" -> "5070";  "Tesla V100-SXM2-16GB" -> "V100"
short_gpu() {
    local g="$1"
    g=${g#NVIDIA }; g=${g#GeForce }; g=${g#Tesla }; g=${g#RTX }; g=${g#GTX }
    g=${g%%-*}
    echo "$g"
}
[ -n "$GPU_OVERRIDE" ] && GPU="$GPU_OVERRIDE" || GPU=$(short_gpu "$GPU")

warn=0
for v in TOTAL UNIQ ROWS DENS LANTIME; do
    [ "${!v}" -eq 0 ] 2>/dev/null && { echo "warning: could not parse $v from the log" >&2; warn=1; }
done
[ -z "$GPU" ] && { echo "warning: no 'using GPU' line in the log (CPU or MPI run?) -- use -g" >&2; warn=1; }
[ -z "$FACTORS" ] && { echo "warning: no factors found -- did -nc3 finish?" >&2; warn=1; }
[ "$DENS_GUESS" -eq 1 ] && { echo "warning: TD=$DENS is the last density tried, not necessarily the one used" >&2; warn=1; }
[ "$COMPOSITE" -eq 1 ] && { echo "warning: a composite (c) cofactor is in the output -- NOT fully factored" >&2; warn=1; }

# ---- pastebin -------------------------------------------------------------
# Read KEY=VALUE lines out of the credentials file *without* sourcing it, so
# that shell metacharacters in a password ($ ` ! " ' \ space #) are literal.
# One optional layer of matching surrounding quotes is stripped, so both
#   PASTEBIN_PASS=my$ecret!
#   PASTEBIN_PASS='my$ecret!'
# give the same password.  Don't put spaces around the '='.
CRED_API_KEY=""; CRED_USER=""; CRED_PASS=""; CRED_USER_KEY=""
read_creds() {
    local line k v
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%$'\r'}
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        k=${line%%=*}
        v=${line#*=}
        k=${k#"${k%%[![:space:]]*}"}          # trim whitespace around the key
        k=${k%"${k##*[![:space:]]}"}
        case "$v" in
            \"*\") v=${v#\"}; v=${v%\"} ;;
            \'*\') v=${v#\'}; v=${v%\'} ;;
        esac
        case "$k" in
            PASTEBIN_API_KEY)  CRED_API_KEY=$v ;;
            PASTEBIN_USER)     CRED_USER=$v ;;
            PASTEBIN_PASS)     CRED_PASS=$v ;;
            PASTEBIN_USER_KEY) CRED_USER_KEY=$v ;;
        esac
    done < "$1"
}

find_creds() {
    local c
    for c in ./.pastebin_credentials "$HOME/.pastebin_credentials"; do
        [ -r "$c" ] && { printf '%s' "$c"; return 0; }
    done
    return 1
}

upload() {
    local creds
    creds=$(find_creds) || {
        echo "note: no .pastebin_credentials found, skipping upload" >&2
        return 1
    }
    # NB: the file is parsed, never sourced -- a password full of $ ` ! " ' or
    # spaces is taken exactly as written.
    read_creds "$creds"
    local key="$CRED_API_KEY"
    [ -z "$key" ] && { echo "note: PASTEBIN_API_KEY not set in $creds" >&2; return 1; }

    # cache is per-account: changing PASTEBIN_USER must not reuse the old key
    local userkey="$CRED_USER_KEY" cache="$HOME/.cache/pastebin_user_key_${CRED_USER//[^A-Za-z0-9_.-]/_}"
    if [ -z "$userkey" ] && [ -n "$CRED_USER" ] && [ -n "$CRED_PASS" ]; then
        [ -r "$cache" ] && userkey=$(cat "$cache")
        if [ -z "$userkey" ]; then
            userkey=$(curl -s --max-time 30 https://pastebin.com/api/api_login.php \
                        -d "api_dev_key=$key" \
                        --data-urlencode "api_user_name=$CRED_USER" \
                        --data-urlencode "api_user_password=$CRED_PASS")
            case "$userkey" in
                Bad*|*"error"*|"") echo "pastebin login failed: $userkey" >&2; userkey="" ;;
                *) mkdir -p "$(dirname "$cache")"
                   ( umask 077; printf '%s' "$userkey" > "$cache" ) ;;
            esac
        fi
    fi

    local out
    out=$(curl -s --max-time 120 https://pastebin.com/api/api_post.php \
            -d "api_dev_key=$key" \
            -d "api_option=paste" \
            -d "api_paste_private=1" \
            -d "api_paste_expire_date=N" \
            -d "api_paste_format=text" \
            ${userkey:+-d "api_user_key=$userkey"} \
            --data-urlencode "api_paste_name=$JOB msieve.log" \
            --data-urlencode "api_paste_code@$LOG")
    case "$out" in
        https://pastebin.com/*) printf '%s' "$out"; return 0 ;;
        *) echo "pastebin upload failed: $out" >&2
           # a stale cached user key is the usual cause
           [ -n "$userkey" ] && rm -f "$cache"
           return 1 ;;
    esac
}

if [ "$TEST_LOGIN" -eq 1 ]; then
    creds=$(find_creds) || { echo "no .pastebin_credentials found" >&2; exit 1; }
    read_creds "$creds"
    echo "credentials file: $creds"
    [ -z "$CRED_API_KEY" ] && { echo "PASTEBIN_API_KEY is not set" >&2; exit 1; }
    echo "api key:  ${#CRED_API_KEY} chars"
    echo "user:     ${CRED_USER:-<none, will paste anonymously>}"
    echo "password: ${#CRED_PASS} chars"
    if [ -n "$CRED_USER_KEY" ]; then
        echo "user key given directly in the file, nothing to test"
        exit 0
    fi
    [ -z "$CRED_USER" ] && exit 0
    resp=$(curl -s --max-time 30 https://pastebin.com/api/api_login.php \
             -d "api_dev_key=$CRED_API_KEY" \
             --data-urlencode "api_user_name=$CRED_USER" \
             --data-urlencode "api_user_password=$CRED_PASS")
    case "$resp" in
        Bad*|*error*|"") echo "login FAILED: $resp" >&2; exit 1 ;;
        *) echo "login ok, user key: ${resp:0:4}...${resp: -4} (${#resp} chars)"
           rm -f "$HOME/.cache/pastebin_user_key_${CRED_USER//[^A-Za-z0-9_.-]/_}"
           exit 0 ;;
    esac
fi

if [ -z "$PASTE_URL" ] && [ "$NO_PASTE" -eq 0 ]; then
    PASTE_URL=$(upload) || PASTE_URL="<paste failed -- upload msieve.log manually>"
fi

# ---- emit -----------------------------------------------------------------
POST=$(cat <<EOF
$JOB is complete at $(fmt_m "$TOTAL") ($(fmt_m "$UNIQ") unique) relations. Lanczos was $(fmt_hm "$LANTIME") on a $(fmt_mat "$ROWS") matrix on a $GPU at TD=$DENS.

[CODE]$(printf '%b' "$FACTORS")[/CODE]
EOF
)
[ -n "$PASTE_URL" ] && POST=$(printf '%s\n\n%s' "$POST" "$PASTE_URL")

printf '%s\n' "$POST"
if command -v clip.exe >/dev/null 2>&1; then
    printf '%s\n' "$POST" | clip.exe && echo >&2 && echo "(copied to Windows clipboard)" >&2
fi
exit $warn
