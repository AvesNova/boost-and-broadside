# env.sh: the environment of every job on a box (sourced by runner.sh and bootstrap.sh).
# Every cache on /workspace (the volume), so a restart, and a later box of the same GPU
# model given the archive (cache.sh), compiles nothing again.
export WORKSPACE=${WORKSPACE:-/workspace}
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/cache/inductor
export TRITON_CACHE_DIR=$WORKSPACE/cache/triton
export UV_CACHE_DIR=$WORKSPACE/cache/uv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$WORKSPACE/bin:$PATH
# Inductor's compile workers and torch's CPU threads: the machine's CPUs shared by its jobs
# (one per GPU). The 8x box (192 CPUs): 24 each; a 1-GPU box: all of them. The box's CPU
# quota, not nproc, where a container is capped (cgroup cpu.max).
_cpus=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r _quota _period < /sys/fs/cgroup/cpu.max
  [ "$_quota" != max ] && _cpus=$(( _quota / _period ))
fi
_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$_gpus" -ge 1 ] || _gpus=1
_share=$(( _cpus / _gpus ))
[ "$_share" -ge 2 ] || _share=2
export TORCHINDUCTOR_COMPILE_THREADS=$_share
export OMP_NUM_THREADS=$_share
unset _cpus _quota _period _gpus _share
