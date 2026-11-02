# Plots for the writeup. Usage (from the repo root): gnuplot -e "dir='results'" scripts/plot.gp
if (!exists("dir")) dir = "results"
set datafile separator ","
set terminal pngcairo size 900,560 font "Sans,11" noenhanced
set grid lc rgb "#dddddd"
set key top left box opaque
P = "#1f77b4"; C = "#d62728"; P2 = "#2ca02c"; C2 = "#ff7f0e"
# Select rows whose first column equals `v` (skips the header).
sel(f, v) = sprintf("< awk -F, 'NR>1 && $1==\"%s\"' %s/%s", v, dir, f)
selp(f, col, v) = sprintf("< awk -F, 'NR>1 && $%d==\"%s\"' %s/%s", col, v, dir, f)

# ---------------------------------------------------------------- 1. fragmentation vs block size
set output dir."/fig_fragmentation.png"
set title "KV memory waste vs block size (simulator, 2 GB pool, 2000-request burst)"
set xlabel "block size (tokens)"; set ylabel "wasted slots / allocated slots (%)"
set logscale x 2; set xtics (8,16,32,64); set yrange [0:*]
plot sel("sim_blocksize.csv","paged") u 3:17 w lp lw 2 pt 7 lc rgb P t "paged: total waste", \
     ""                               u 3:18 w lp lw 2 pt 6 dt 2 lc rgb P t "paged: partial last block", \
     sel("sim_blocksize.csv","contiguous") u 3:17 w lp lw 2 pt 5 lc rgb C t "contiguous: total waste", \
     ""                               u 3:18 w lp lw 2 pt 4 dt 2 lc rgb C t "contiguous: partial last block"
unset logscale x; set xtics autofreq

# ---------------------------------------------------------------- 2. capacity vs pool size
set output dir."/fig_capacity.png"
set title "Peak concurrent requests vs KV pool size"
set xlabel "KV pool (GB)"; set ylabel "max running requests"
set logscale xy 2; set xtics (0.5,1,2,4,8); set yrange [*:*]
plot sel("sim_capacity.csv","paged") u ($4*16*22528/2**30):13 w lp lw 2 pt 7 lc rgb P t "paged (simulator)", \
     sel("sim_capacity.csv","contiguous") u ($4*16*22528/2**30):13 w lp lw 2 pt 5 lc rgb C t "contiguous (simulator)", \
     sel("bench_capacity.csv","paged") u 4:25 w p ps 2 pt 9 lc rgb P2 t "paged (GPU engine)", \
     sel("bench_capacity.csv","contiguous") u 4:25 w p ps 2 pt 11 lc rgb C2 t "contiguous (GPU engine)"
unset logscale; set xtics autofreq

# ---------------------------------------------------------------- 3. throughput vs pool size (GPU)
set output dir."/fig_throughput_pool.png"
set title "Offline throughput vs KV pool size (RTX 4080, TinyLlama-1.1B, 400-request burst)"
set xlabel "KV pool (GB)"; set ylabel "output tokens / s"; set yrange [0:*]
set logscale x 2; set xtics (0.5,1,2)
plot sel("bench_capacity.csv","paged") u 4:13 w lp lw 2 pt 7 lc rgb P t "paged", \
     sel("bench_capacity.csv","contiguous") u 4:13 w lp lw 2 pt 5 lc rgb C t "contiguous"
unset logscale; set xtics autofreq

# ---------------------------------------------------------------- 4. arrival-rate sweep (GPU)
set output dir."/fig_rate.png"
set terminal pngcairo size 1200,480 font "Sans,11" noenhanced
set multiplot layout 1,2 title "Online serving vs arrival rate (1 GB KV pool, Poisson arrivals)"
set title "throughput"; set xlabel "arrival rate (req/s)"; set ylabel "output tokens / s"
set yrange [0:*]
plot sel("bench_rate.csv","paged") u 5:13 w lp lw 2 pt 7 lc rgb P t "paged", \
     sel("bench_rate.csv","contiguous") u 5:13 w lp lw 2 pt 5 lc rgb C t "contiguous"
set title "latency"; set ylabel "seconds (log)"; set logscale y; set yrange [*:*]
plot sel("bench_rate.csv","paged") u 5:19 w lp lw 2 pt 7 lc rgb P t "paged TTFT p99", \
     sel("bench_rate.csv","contiguous") u 5:19 w lp lw 2 pt 5 lc rgb C t "contiguous TTFT p99", \
     sel("bench_rate.csv","paged") u 5:22 w lp lw 2 pt 6 dt 2 lc rgb P t "paged E2E p99", \
     sel("bench_rate.csv","contiguous") u 5:22 w lp lw 2 pt 4 dt 2 lc rgb C t "contiguous E2E p99"
unset multiplot; unset logscale
set terminal pngcairo size 900,560 font "Sans,11" noenhanced

# ---------------------------------------------------------------- 5. prefix caching (GPU)
set output dir."/fig_prefix.png"
set title "Prefix caching vs shared-prefix fraction (512-token system prompt, 12 req/s)"
set xlabel "fraction of requests sharing the system prompt"
set ylabel "mean TTFT (ms)"; set y2label "block hit rate (%)"; set y2tics; set ytics nomirror
set yrange [0:*]; set y2range [0:100]
plot sel("bench_prefix.csv","paged") u 6:($16*1000) w lp lw 2 pt 5 lc rgb C t "TTFT, prefix cache off", \
     sel("bench_prefix.csv","paged+prefix") u 6:($16*1000) w lp lw 2 pt 7 lc rgb P t "TTFT, prefix cache on", \
     sel("bench_prefix.csv","paged+prefix") u 6:($31*100) axes x1y2 w lp lw 2 dt 2 pt 6 lc rgb P2 t "hit rate (right axis)"
unset y2tics; unset y2label; set ytics mirror

set output dir."/fig_prefix_saved.png"
set title "Prefill work saved by prefix caching (simulator, 2 GB pool)"
set ylabel "prefill tokens computed (thousands)"; set yrange [0:*]
plot selp("sim_prefix.csv",2,"0") u 8:($24/1000) w lp lw 2 pt 5 lc rgb C t "prefix cache off", \
     selp("sim_prefix.csv",2,"1") u 8:($24/1000) w lp lw 2 pt 7 lc rgb P t "prefix cache on"

# ---------------------------------------------------------------- 6. attention kernel
set output dir."/fig_kernel.png"
set terminal pngcairo size 1200,480 font "Sans,11" noenhanced
set multiplot layout 1,2 title "Decode attention, one layer (32 q heads, 4 kv heads, head_dim 64, B=16)"
kern(b, v) = sprintf("< awk -F, 'NR>1 && $2==%d && $4==\"%s\"' %s/kernel_attn.csv", b, v, dir)
set logscale x 2; set xtics (128,256,512,1024,2048); set xlabel "context length"
set title "batch 32: time per call"; set ylabel "microseconds"; set yrange [0:*]
plot kern(32,"paged-v1") u 3:5 w lp lw 2 pt 7 lc rgb P t "paged v1", \
     kern(32,"contiguous-v1") u 3:5 w lp lw 2 pt 5 lc rgb C t "contiguous v1", \
     kern(32,"paged-v2") u 3:5 w lp lw 2 dt 2 pt 6 lc rgb P t "paged v2", \
     kern(32,"contiguous-v2") u 3:5 w lp lw 2 dt 2 pt 4 lc rgb C t "contiguous v2"
set title "achieved K/V bandwidth (% of 717 GB/s DRAM; >100% = L2-resident)"; set ylabel "% of peak"; set yrange [0:130]
plot kern(32,"paged-v1") u 3:7 w lp lw 2 pt 7 lc rgb P t "paged v1, batch 32", \
     kern(32,"paged-v2") u 3:7 w lp lw 2 dt 2 pt 6 lc rgb P t "paged v2, batch 32", \
     kern(128,"paged-v1") u 3:7 w lp lw 2 pt 7 lc rgb P2 t "paged v1, batch 128", \
     kern(128,"paged-v2") u 3:7 w lp lw 2 dt 2 pt 6 lc rgb P2 t "paged v2, batch 128", \
     kern(1,"paged-v2") u 3:7 w lp lw 2 dt 3 pt 2 lc rgb C2 t "paged v2, batch 1"
unset multiplot; unset logscale; set xtics autofreq
set terminal pngcairo size 900,560 font "Sans,11" noenhanced

# ---------------------------------------------------------------- 7. time series (GPU)
set output dir."/fig_trace.png"
set title "Running batch over time (1 GB pool, 300-request burst)"
set xlabel "time (s)"; set ylabel "running requests"; set yrange [0:*]
plot sprintf("< tail -n +2 %s/bench_trace_paged.csv", dir) u 1:7 w l lw 2 lc rgb P t "paged", \
     sprintf("< tail -n +2 %s/bench_trace_contiguous.csv", dir) u 1:7 w l lw 2 lc rgb C t "contiguous"

set output dir."/fig_sim_trace.png"
set title "Pool slots holding real tokens over time (simulator, 1 GB pool, 1000-request burst)"
set xlabel "engine step"; set ylabel "% of pool slots used by tokens"; set yrange [0:100]
plot sprintf("< tail -n +2 %s/sim_trace_paged.csv", dir) u 1:(100.0*$7/(2978*16)) w l lw 2 lc rgb P t "paged", \
     sprintf("< tail -n +2 %s/sim_trace_contiguous.csv", dir) u 1:(100.0*$7/(2978*16)) w l lw 2 lc rgb C t "contiguous"
