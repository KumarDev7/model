#!/usr/bin/env bash
# Train the memory-pool LM on Ultra-FineWeb streamed from Hugging Face.
# The same command starts a new run or resumes it (same VM or a new one).
#
#   scripts/train_stream.sh            start or resume, in the foreground
#   nohup scripts/train_stream.sh > /dev/null 2>&1 &     same, detached (logs in $RUN_DIR/logs)
#   scripts/train_stream.sh status     step, loss, data position, waits for data, disk
#   scripts/train_stream.sh stop       stop training, the tokenizer and the dashboard (keeps the last checkpoint)
#   scripts/train_stream.sh url        print the dashboard link
#
# Logs ($RUN_DIR): logs/train.log (console), model.msgpack.metrics.jsonl (one JSON line per
# step: every metric, timings, MFU, data position, memory; eval/checkpoint/sample events),
# model.msgpack.samples.jsonl (text generated on the CPU). Full analysis at any time:
#   python -m memory_pool_model.metrics_report $RUN_DIR/model.msgpack.metrics.jsonl --plot run.png
#
# What it does:
#   1. tokenizer.json + val.npy in $DATA_DIR (built once by prepare_ultrafineweb; reused after)
#   2. the stream producer (experiments.stream_ultrafineweb) in the background, restarted if it dies
#   3. training (--task stream --resume), restarted from its last checkpoint if it crashes
#
# The run settings are saved in $RUN_DIR/run.conf on the first start and read
# from there on every resume (STEPS sets the learning-rate schedule, so it
# must not change silently). Edit run.conf to change them on purpose.
#
# Settings (environment variables, first start only):
#   RUN_NAME      run name (default pool_768x12)
#   BASE          where runs and data live (default /kaggle/working if it exists, else <repo>/runs)
#   RUN_DIR       checkpoint + logs; keep it on a disk that survives the VM (default $BASE/runs/$RUN_NAME)
#   DATA_DIR      tokenizer.json + val.npy (default $BASE/data/ufw)
#   STREAM_DIR    token shards; fast local disk, may be lost (default /dev/shm/stream_$RUN_NAME)
#   TOKENIZER     existing tokenizer.json to reuse when DATA_DIR has none (else a new one is trained)
#   STEPS BATCH LR WARMUP EVAL_EVERY CKPT_EVERY LOG_EVERY MODEL_FLAGS
#   SAMPLE_EVERY  generate text on the CPU every N steps (default 5000, 0 = off); SAMPLE_CPUS cores for it
#   PARTS MAX_READY_GB WORKERS SHARD_TOKENS   (stream producer)
#   MAX_RESTARTS  training restarts after a crash before giving up (default 5)
#   DASHBOARD     1 = live web dashboard of the run (default), 0 = off; DASHBOARD_PORT (default 8765)
#   TUNNEL        cloudflared = public https link through a Cloudflare quick tunnel (default), none = local only
#   WANDB_API_KEY set it to mirror metrics and samples to Weights & Biases (WANDB_PROJECT, WANDB_ENTITY)
#   PYTHON        python interpreter (default python3)
#   POLL_SECONDS  how often the supervisor checks training and the producer (default 30)

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
RUN_NAME=${RUN_NAME:-pool_768x12}
if [ -z "${BASE:-}" ]; then
    if [ -d /kaggle/working ]; then BASE=/kaggle/working; else BASE=$REPO/runs; fi
fi
RUN_DIR=${RUN_DIR:-$BASE/runs/$RUN_NAME}
LOG_DIR=$RUN_DIR/logs
CONF=$RUN_DIR/run.conf
CKPT=$RUN_DIR/model.msgpack          # final params; resumable state is $CKPT.state(.json)
STATE_JSON=$CKPT.state.json

log() { echo "[$(date -u '+%F %T')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# a pid file is trusted only if that process still runs the expected command
alive() {  # alive <pidfile> <pattern>
    local pid
    [ -f "$1" ] || return 1
    pid=$(cat "$1")
    [ -n "$pid" ] && ps -p "$pid" -o args= 2>/dev/null | grep -q -- "$2"
}

stop_pid() {  # stop_pid <pidfile> <pattern> <name>
    if alive "$1" "$2"; then
        local pid
        pid=$(cat "$1")
        log "stopping $3 (pid $pid)"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 30); do alive "$1" "$2" || break; sleep 1; done
        alive "$1" "$2" && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; }
    fi
    rm -f "$1"
}

# ------------------------------------------------------------------ settings
write_conf() {
    mkdir -p "$RUN_DIR"
    cat > "$CONF" <<EOF
# written on the first start of $RUN_NAME ($(date -u '+%F %T') UTC); read on every resume
DATA_DIR=${DATA_DIR:-$BASE/data/ufw}
STREAM_DIR=${STREAM_DIR:-/dev/shm/stream_$RUN_NAME}
# 100,000 steps x 256 x 256 tokens = 6.55B tokens (~4.5 h at 160 ms/step on v5e-8)
STEPS=${STEPS:-100000}
BATCH=${BATCH:-256}
LR=${LR:-6e-4}
WARMUP=${WARMUP:-2000}
LOG_EVERY=${LOG_EVERY:-250}
EVAL_EVERY=${EVAL_EVERY:-5000}
# checkpoints are written in the background; training waits ~1 s for the copy to host memory
CKPT_EVERY=${CKPT_EVERY:-2000}
# text from the current weights, generated on the CPU in its own process (training doesn't wait)
SAMPLE_EVERY=${SAMPLE_EVERY:-5000}
SAMPLE_CPUS=${SAMPLE_CPUS:-16}
# the validated 368M model: d768 x 12 backbone, 1M-vector pool read in layers 4 and 8
MODEL_FLAGS="${MODEL_FLAGS:---d_model 768 --n_layers 12 --n_heads 12 --ffn_mult 4 --max_len 256 --compute_dtype bfloat16 --memory_layers 4,8 --n_sub_keys 1024 --d_key 128 --d_value 256 --memory_ffn true --top_k 8 --pool_sharding false}"
# Ultra-FineWeb English files in reading order (part 2 is the held-out split)
PARTS=${PARTS:-1,3-2047}
MAX_READY_GB=${MAX_READY_GB:-10}
WORKERS=${WORKERS:-2}
SHARD_TOKENS=${SHARD_TOKENS:-50000000}
EOF
}

load_conf() {
    local overridden=()
    for v in STEPS BATCH LR WARMUP MODEL_FLAGS PARTS SHARD_TOKENS DATA_DIR STREAM_DIR; do
        if [ -n "${!v:-}" ]; then overridden+=("$v"); fi
    done
    if [ -f "$CONF" ]; then
        [ ${#overridden[@]} -eq 0 ] || log "note: using $CONF; ignoring ${overridden[*]} from the environment (edit run.conf to change)"
    else
        write_conf
        log "new run $RUN_NAME: settings in $CONF"
    fi
    # shellcheck disable=SC1090
    source "$CONF"
}

# ---------------------------------------------------------------- data setup
check_env() {
    "$PYTHON" -c "import jax, flax, optax, numpy" 2>/dev/null || die "jax/flax/optax/numpy missing: pip install -r $REPO/requirements.txt"
    if ! "$PYTHON" -c "import datasets, tokenizers" 2>/dev/null; then
        log "installing datasets + tokenizers (needed by the stream producer)"
        "$PYTHON" -m pip install -q datasets tokenizers || die "pip install datasets tokenizers failed"
    fi
    local info
    info=$("$PYTHON" -c "import jax; print(jax.device_count(), jax.devices()[0].platform)" 2>/dev/null | tail -1)
    log "devices: $info"
    local n=${info%% *}
    [[ "$n" =~ ^[0-9]+$ ]] || die "jax found no devices"
    [ $((BATCH % n)) -eq 0 ] || die "BATCH=$BATCH is not divisible by $n devices"
}

ensure_data() {
    if [ -f "$DATA_DIR/tokenizer.json" ] && [ -f "$DATA_DIR/val.npy" ] && [ -f "$DATA_DIR/meta.json" ]; then
        return
    fi
    if [ -f "$STATE_JSON" ] && [ ! -f "$DATA_DIR/tokenizer.json" ] && [ -z "${TOKENIZER:-}" ]; then
        die "$RUN_DIR has a checkpoint but $DATA_DIR/tokenizer.json is gone; a new tokenizer would not match it (set TOKENIZER=<its tokenizer.json>)"
    fi
    mkdir -p "$DATA_DIR"
    if [ -n "${TOKENIZER:-}" ] && [ ! -f "$DATA_DIR/tokenizer.json" ]; then
        cp "$TOKENIZER" "$DATA_DIR/tokenizer.json"
        log "reusing tokenizer $TOKENIZER"
    fi
    log "building tokenizer + held-out set in $DATA_DIR (first start only)"
    # train.npy here is a token sample only; training reads the stream
    (cd "$REPO" && "$PYTHON" -m experiments.prepare_ultrafineweb --out "$DATA_DIR" --train_tokens 1000000) \
        >> "$LOG_DIR/prepare.log" 2>&1 || die "prepare_ultrafineweb failed, see $LOG_DIR/prepare.log"
}

# ------------------------------------------------------------ stream producer
start_producer() {
    alive "$RUN_DIR/stream.pid" stream_ultrafineweb && return
    if grep -q STREAM_DONE "$LOG_DIR/stream.log" 2>/dev/null && [ -f "$STREAM_DIR/stream.json" ]; then
        return  # every part tokenised
    fi
    local from=()
    [ -f "$STATE_JSON" ] && from=(--from_checkpoint "$STATE_JSON")
    mkdir -p "$STREAM_DIR"
    log "starting stream producer -> $STREAM_DIR ${from[*]}"
    (cd "$REPO" && exec setsid nice -n 5 "$PYTHON" -m experiments.stream_ultrafineweb --out "$STREAM_DIR" \
        --tokenizer "$DATA_DIR/tokenizer.json" --parts "$PARTS" --shard_tokens "$SHARD_TOKENS" \
        --max_ready_gb "$MAX_READY_GB" --workers "$WORKERS" "${from[@]}" >> "$LOG_DIR/stream.log" 2>&1) &
    echo $! > "$RUN_DIR/stream.pid"
}

# ------------------------------------------------------------ live dashboard
DASHBOARD=${DASHBOARD:-1}
DASHBOARD_PORT=${DASHBOARD_PORT:-8765}
TUNNEL=${TUNNEL:-cloudflared}

start_dashboard() {
    [ "$DASHBOARD" = 1 ] || return 0
    if ! alive "$RUN_DIR/dashboard.pid" memory_pool_model.dashboard; then
        (cd "$REPO" && exec setsid nice -n 5 "$PYTHON" -m memory_pool_model.dashboard --run_dir "$RUN_DIR" \
            --port "$DASHBOARD_PORT" >> "$LOG_DIR/dashboard.log" 2>&1) &
        echo $! > "$RUN_DIR/dashboard.pid"
        for _ in $(seq 20); do [ -s "$RUN_DIR/dashboard.token" ] && break; sleep 0.5; done
        log "dashboard on port $DASHBOARD_PORT"
    fi
    start_tunnel
}

cloudflared_bin() {
    command -v cloudflared 2>/dev/null && return 0
    local bin=$BASE/bin/cloudflared arch
    if [ ! -x "$bin" ]; then
        case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64 | arm64) arch=arm64 ;; *) return 1 ;; esac
        mkdir -p "$BASE/bin"
        curl -fsSL -o "$bin.tmp" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch" \
            && chmod +x "$bin.tmp" && mv "$bin.tmp" "$bin" || return 1
    fi
    echo "$bin"
}

start_tunnel() {
    local token url bin
    token=$(cat "$RUN_DIR/dashboard.token" 2>/dev/null || true)
    if [ "$TUNNEL" != cloudflared ]; then
        echo "http://127.0.0.1:$DASHBOARD_PORT/?token=$token" > "$RUN_DIR/dashboard_url.txt"
        return 0
    fi
    alive "$RUN_DIR/tunnel.pid" cloudflared && return 0
    bin=$(cloudflared_bin) || { log "WARNING: no cloudflared; dashboard only at http://127.0.0.1:$DASHBOARD_PORT/?token=$token"; return 0; }
    : > "$LOG_DIR/tunnel.log"
    (exec setsid "$bin" tunnel --no-autoupdate --url "http://127.0.0.1:$DASHBOARD_PORT" >> "$LOG_DIR/tunnel.log" 2>&1) &
    echo $! > "$RUN_DIR/tunnel.pid"
    for _ in $(seq 60); do
        url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | head -n 1 || true)
        [ -n "$url" ] && break
        sleep 1
    done
    if [ -n "$url" ]; then
        (umask 077 && echo "$url/?token=$token" > "$RUN_DIR/dashboard_url.txt")
        log "dashboard: $url/?token=$token"
    else
        log "WARNING: tunnel gave no URL yet (see $LOG_DIR/tunnel.log)"
    fi
}

start_wandb() {
    [ -n "${WANDB_API_KEY:-}" ] || return 0
    alive "$RUN_DIR/wandb.pid" memory_pool_model.wandb_sync && return 0
    grep -q "W&B sync done" "$LOG_DIR/wandb.log" 2>/dev/null && [ "$(ckpt_step)" -ge "$STEPS" ] && return 0
    if ! "$PYTHON" -c "import wandb" 2>/dev/null; then
        "$PYTHON" -m pip install -q wandb || { log "WARNING: pip install wandb failed; no W&B mirror"; return 0; }
    fi
    (cd "$REPO" && exec setsid nice -n 5 "$PYTHON" -m memory_pool_model.wandb_sync --run_dir "$RUN_DIR" \
        --name "$RUN_NAME" >> "$LOG_DIR/wandb.log" 2>&1) &
    echo $! > "$RUN_DIR/wandb.pid"
    log "mirroring to Weights & Biases (project ${WANDB_PROJECT:-memory-pool-lm})"
}

start_monitors() {
    start_dashboard
    start_wandb
}

stop_monitors() {
    stop_pid "$RUN_DIR/wandb.pid" memory_pool_model.wandb_sync "W&B sync"
    stop_pid "$RUN_DIR/tunnel.pid" cloudflared tunnel
    stop_pid "$RUN_DIR/dashboard.pid" memory_pool_model.dashboard dashboard
}

# ------------------------------------------------------------------ training
train_once() {
    local vocab
    vocab=$("$PYTHON" -c "import json; print(json.load(open('$DATA_DIR/meta.json'))['vocab_size'])")
    # shellcheck disable=SC2086
    (cd "$REPO" && exec setsid "$PYTHON" -m memory_pool_model.train --task stream --stream_dir "$STREAM_DIR" \
        --eval_tokens "$DATA_DIR/val.npy" --vocab_size "$vocab" --eval_windows 256 \
        --steps "$STEPS" --batch_size "$BATCH" --lr "$LR" --warmup_steps "$WARMUP" \
        --log_every "$LOG_EVERY" --eval_every "$EVAL_EVERY" --checkpoint_every "$CKPT_EVERY" \
        --sample_every "${SAMPLE_EVERY:-0}" --sample_cpus "${SAMPLE_CPUS:-16}" --tokenizer "$DATA_DIR/tokenizer.json" \
        --data_parallel true $MODEL_FLAGS --save "$CKPT" --resume >> "$LOG_DIR/train.log" 2>&1) &
    local pid=$!
    echo $pid > "$RUN_DIR/train.pid"
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$POLL"
        start_producer  # restart it if it died
        start_monitors  # and the dashboard, tunnel, W&B mirror
    done
    local rc=0
    wait "$pid" || rc=$?
    rm -f "$RUN_DIR/train.pid"
    return $rc
}

ckpt_step() {
    [ -f "$STATE_JSON" ] && "$PYTHON" -c "import json; print(json.load(open('$STATE_JSON'))['step'])" 2>/dev/null || echo 0
}

run() {
    mkdir -p "$RUN_DIR"
    exec 9> "$RUN_DIR/.lock"
    flock -n 9 || die "$RUN_NAME is already running (see: $0 status)"
    load_conf
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$LOG_DIR/supervisor.log") 2>&1
    echo $$ > "$RUN_DIR/supervisor.pid"
    trap 'log "interrupted"; stop_pid "$RUN_DIR/train.pid" memory_pool_model.train training; stop_pid "$RUN_DIR/stream.pid" stream_ultrafineweb "stream producer"; stop_monitors; rm -f "$RUN_DIR/supervisor.pid"; exit 130' INT TERM

    check_env
    ensure_data
    local step
    step=$(ckpt_step)
    if [ "$step" -gt 0 ]; then log "resuming $RUN_NAME from step $step of $STEPS"; else log "starting $RUN_NAME: $STEPS steps"; fi
    start_producer
    start_monitors

    local restarts=0
    while true; do
        local before rc=0
        before=$(ckpt_step)
        log "training (checkpoint at step $before)"
        train_once || rc=$?
        if [ $rc -eq 0 ]; then
            log "training finished: $CKPT"
            [ "$DASHBOARD" = 1 ] && log "the dashboard stays up for the finished run ($0 url); $0 stop ends it"
            break
        fi
        local now
        now=$(ckpt_step)
        # a crash after progress doesn't count against the limit
        if [ "$now" -gt "$before" ]; then restarts=0; else restarts=$((restarts + 1)); fi
        log "training exited with $rc at checkpoint step $now (restart $restarts of $MAX_RESTARTS); last lines:"
        tail -n 5 "$LOG_DIR/train.log" | cut -c1-300
        [ $restarts -ge "$MAX_RESTARTS" ] && { stop_pid "$RUN_DIR/stream.pid" stream_ultrafineweb "stream producer"; die "too many failed restarts"; }
        sleep "$POLL"
        start_producer
    done
    stop_pid "$RUN_DIR/stream.pid" stream_ultrafineweb "stream producer"
    rm -f "$RUN_DIR/supervisor.pid"
}

MAX_RESTARTS=${MAX_RESTARTS:-5}
POLL=${POLL_SECONDS:-30}  # supervisor check interval

# -------------------------------------------------------------------- status
status() {
    [ -f "$CONF" ] || die "no run $RUN_NAME in $RUN_DIR"
    # shellcheck disable=SC1090
    source "$CONF"
    echo "run        $RUN_NAME ($RUN_DIR)"
    if alive "$RUN_DIR/supervisor.pid" train_stream; then echo "supervisor running (pid $(cat "$RUN_DIR/supervisor.pid"))"; else echo "supervisor not running"; fi
    if alive "$RUN_DIR/train.pid" memory_pool_model.train; then echo "training   running"; else echo "training   not running"; fi
    if alive "$RUN_DIR/stream.pid" stream_ultrafineweb; then echo "producer   running"; else echo "producer   not running"; fi
    echo "checkpoint step $(ckpt_step) of $STEPS"
    [ -f "$RUN_DIR/dashboard_url.txt" ] && echo "dashboard  $(cat "$RUN_DIR/dashboard_url.txt")$(alive "$RUN_DIR/dashboard.pid" memory_pool_model.dashboard || echo ' (not running)')"
    if [ -f "$CKPT.metrics.jsonl" ]; then
        (cd "$REPO" && "$PYTHON" -m memory_pool_model.metrics_report "$CKPT.metrics.jsonl" --brief 2>/dev/null) || true
    fi
    if [ -d "$STREAM_DIR" ]; then
        local n
        n=$(find "$STREAM_DIR" -maxdepth 1 -name 'p????-????.npy' | wc -l)
        echo "stream     $n shards ready, $(du -sh "$STREAM_DIR" 2>/dev/null | cut -f1) in $STREAM_DIR; $({ grep -c 'done,' "$LOG_DIR/stream.log" 2>/dev/null || true; }) parts finished this session"
        tail -n 1 "$LOG_DIR/stream.log" 2>/dev/null | sed 's/^/producer   /' || true
    fi
    echo "disk       $(df -h "$RUN_DIR" | awk 'NR==2 {print $4 " free on " $6}') (checkpoint), $(df -h "$STREAM_DIR" 2>/dev/null | awk 'NR==2 {print $4 " free on " $6}') (stream)"
}

stop() {
    [ -d "$RUN_DIR" ] || die "no run $RUN_NAME in $RUN_DIR"
    stop_pid "$RUN_DIR/supervisor.pid" train_stream supervisor
    stop_pid "$RUN_DIR/train.pid" memory_pool_model.train training
    stop_pid "$RUN_DIR/stream.pid" stream_ultrafineweb "stream producer"
    stop_monitors
    log "stopped; resume with: $0 (continues from checkpoint step $(ckpt_step))"
}

case "${1:-run}" in
    run | start | resume) run ;;
    status) status ;;
    url) cat "$RUN_DIR/dashboard_url.txt" 2>/dev/null || die "no dashboard link yet" ;;
    stop) stop ;;
    *) echo "usage: $0 [run|status|stop]" >&2; exit 2 ;;
esac
