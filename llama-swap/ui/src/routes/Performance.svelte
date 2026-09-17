<script lang="ts">
  import { onMount } from "svelte";
  import { fetchPerformance } from "../stores/api";
  import { persistentStore } from "../stores/persistent";
  import type { SysStat, GpuStat } from "../lib/types";
  import PerformanceChart from "../components/PerformanceChart.svelte";
  import SegmentedControl from "../components/SegmentedControl.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { RefreshCw } from "@lucide/svelte";

  const COLORS = [
    "#3b82f6",
    "#ef4444",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#84cc16",
    "#f97316",
    "#14b8a6",
    "#a855f7",
    "#e11d48",
    "#0ea5e9",
    "#eab308",
    "#d946ef",
    "#22d3ee",
  ];

  const WINDOWS = [
    { label: "5分钟", ms: 5 * 60 * 1000 },
    { label: "15分钟", ms: 15 * 60 * 1000 },
    { label: "1小时", ms: 60 * 60 * 1000 },
  ] as const;

  const INTERVALS = [
    { label: "关闭", ms: 0 },
    { label: "5秒", ms: 5000 },
    { label: "10秒", ms: 10000 },
    { label: "30秒", ms: 30000 },
    { label: "60秒", ms: 60000 },
  ] as const;

  let selectedWindow = persistentStore("perf-window", 0);
  let selectedInterval = persistentStore("perf-refresh-interval", 0);
  let sysData = $state<SysStat[]>([]);
  let gpuData = $state<GpuStat[]>([]);
  let refreshing = $state(false);

  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let visible = $state(true);
  let mounted = $state(false);

  function cutoffTime(): number {
    return Date.now() - WINDOWS[$selectedWindow].ms;
  }

  function formatDelta(ts: string, refTime: number): string {
    const diffMs = new Date(ts).getTime() - refTime;
    const diffSec = Math.round(diffMs / 1000);
    const absSec = Math.abs(diffSec);
    const sign = diffSec <= 0 ? "-" : "+";
    if (absSec < 60) return `${sign}${absSec}s`;
    const min = Math.floor(absSec / 60);
    const sec = absSec % 60;
    if (sec === 0) return `${sign}${min}m`;
    return `${sign}${min}:${sec.toString().padStart(2, "0")}`;
  }

  const sysLabels = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    const refTime = new Date(stats[stats.length - 1].timestamp).getTime();
    return stats.map((s) => formatDelta(s.timestamp, refTime));
  });

  async function loadAll() {
    const resp = await fetchPerformance();
    if (resp) {
      sysData = resp.sys_stats ?? [];
      gpuData = resp.gpu_stats ?? [];
    }
  }

  async function loadIncremental() {
    const lastTs = sysData.length > 0 ? sysData[sysData.length - 1].timestamp : undefined;
    const resp = await fetchPerformance(lastTs);
    if (resp) {
      const newSys = resp.sys_stats ?? [];
      const newGpu = resp.gpu_stats ?? [];
      if (newSys.length > 0) {
        sysData = [...sysData, ...newSys];
      }
      if (newGpu.length > 0) {
        gpuData = [...gpuData, ...newGpu];
      }
    }
  }

  function startPolling() {
    stopPolling();
    const ms = INTERVALS[$selectedInterval].ms;
    if (ms <= 0) return;
    pollTimer = setInterval(() => {
      if (visible) {
        loadIncremental();
      }
    }, ms);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function handleVisibility() {
    visible = !document.hidden;
    if (visible && mounted) {
      loadAll().then(() => startPolling());
    } else {
      stopPolling();
    }
  }

  function handleIntervalChange(i: number) {
    $selectedInterval = i;
    if (visible && mounted) {
      startPolling();
    }
  }

  async function manualRefresh() {
    refreshing = true;
    await loadIncremental();
    refreshing = false;
  }

  $effect(() => {
    return () => {
      stopPolling();
    };
  });

  onMount(() => {
    mounted = true;
    document.addEventListener("visibilitychange", handleVisibility);
    loadAll().then(() => startPolling());

    return () => {
      mounted = false;
      stopPolling();
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  });

  // --- System charts (filtered by time window) ---

  const filteredSysStats = $derived(sysData.filter((s) => new Date(s.timestamp).getTime() >= cutoffTime()));

  const cpuDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    const coreCount = stats[0].cpu_util_per_core.length;
    const datasets = [];
    for (let i = 0; i < coreCount; i++) {
      datasets.push({
        label: `Core ${i}`,
        data: stats.map((s) => s.cpu_util_per_core[i]),
        borderColor: COLORS[i % COLORS.length],
      });
    }
    return datasets;
  });

  const memSwapDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    return [
      {
        label: "Memory Used %",
        data: stats.map((s) => (s.mem_used_mb / s.mem_total_mb) * 100),
        borderColor: "#3b82f6",
      },
      {
        label: "Swap Used %",
        data: stats.map((s) => (s.swap_total_mb > 0 ? (s.swap_used_mb / s.swap_total_mb) * 100 : 0)),
        borderColor: "#8b5cf6",
      },
    ];
  });

  const latestMemSwap = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return null;
    const s = stats[stats.length - 1];
    return {
      mem_total_mb: s.mem_total_mb,
      mem_used_mb: s.mem_used_mb,
      mem_used_pct: ((s.mem_used_mb / s.mem_total_mb) * 100).toFixed(1),
      swap_total_mb: s.swap_total_mb,
      swap_used_mb: s.swap_used_mb,
      swap_used_pct: s.swap_total_mb > 0 ? ((s.swap_used_mb / s.swap_total_mb) * 100).toFixed(1) : null,
    };
  });

  const loadDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    return [
      {
        label: "1 min",
        data: stats.map((s) => s.load_avg_1),
        borderColor: "#10b981",
      },
      {
        label: "5 min",
        data: stats.map((s) => s.load_avg_5),
        borderColor: "#f59e0b",
      },
      {
        label: "15 min",
        data: stats.map((s) => s.load_avg_15),
        borderColor: "#ef4444",
      },
    ];
  });

  const netBandwidthDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length < 2) return [];

    const ifaceNames = new Set<string>();
    for (const s of stats) {
      for (const n of s.net_io ?? []) {
        ifaceNames.add(n.name);
      }
    }

    const interfaces = [...ifaceNames].sort();
    if (interfaces.length === 0) return [];

    const datasets: { label: string; data: number[]; borderColor: string }[] = [];
    let colorIdx = 0;

    for (const iface of interfaces) {
      const recvData: number[] = [];
      const sentData: number[] = [];

      for (let i = 1; i < stats.length; i++) {
        const prev = stats[i - 1];
        const curr = stats[i];
        const prevIO = (prev.net_io ?? []).find((n) => n.name === iface);
        const currIO = (curr.net_io ?? []).find((n) => n.name === iface);

        if (!prevIO || !currIO) {
          recvData.push(0);
          sentData.push(0);
          continue;
        }

        const dtMs = new Date(curr.timestamp).getTime() - new Date(prev.timestamp).getTime();
        if (dtMs <= 0) {
          recvData.push(0);
          sentData.push(0);
          continue;
        }

        const dtSec = dtMs / 1000;
        recvData.push((((currIO.bytes_recv - prevIO.bytes_recv) / dtSec) * 8) / 1_000_000);
        sentData.push((((currIO.bytes_sent - prevIO.bytes_sent) / dtSec) * 8) / 1_000_000);
      }

      datasets.push({
        label: `${iface} in`,
        data: recvData,
        borderColor: COLORS[colorIdx % COLORS.length],
      });
      colorIdx++;
      datasets.push({
        label: `${iface} out`,
        data: sentData,
        borderColor: COLORS[colorIdx % COLORS.length],
      });
      colorIdx++;
    }

    return datasets;
  });

  const netBandwidthLabels = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length < 2) return [];
    const refTime = new Date(stats[stats.length - 1].timestamp).getTime();
    return stats.slice(1).map((s) => formatDelta(s.timestamp, refTime));
  });

  // --- GPU charts (filtered by time window) ---

  const filteredGpuStats = $derived(gpuData.filter((g) => new Date(g.timestamp).getTime() >= cutoffTime()));

  const hasGpuData = $derived(gpuData.length > 0);

  const gpuLabels = $derived.by(() => {
    const seen = new Set<string>();
    const labels: string[] = [];
    const stats = filteredGpuStats;
    if (stats.length === 0) return [];
    const refTime = new Date(stats[stats.length - 1].timestamp).getTime();
    for (const g of stats) {
      const label = formatDelta(g.timestamp, refTime);
      if (!seen.has(label)) {
        seen.add(label);
        labels.push(label);
      }
    }
    return labels;
  });

  function buildGpuDatasets(
    stats: GpuStat[],
    field: keyof Pick<GpuStat, "gpu_util_pct" | "mem_util_pct" | "temp_c" | "vram_temp_c" | "power_draw_w">,
  ) {
    if (stats.length === 0) return [];

    const byId = new Map<number, { name: string; values: number[] }>();
    for (const g of stats) {
      if (!byId.has(g.id)) {
        byId.set(g.id, { name: g.name, values: [] });
      }
      byId.get(g.id)!.values.push(g[field] as number);
    }

    const datasets = [];
    let colorIdx = 0;
    for (const [id, entry] of byId) {
      datasets.push({
        label: entry.name || `GPU ${id}`,
        data: entry.values,
        borderColor: COLORS[colorIdx % COLORS.length],
      });
      colorIdx++;
    }
    return datasets;
  }

  const gpuUtilDatasets = $derived(buildGpuDatasets(filteredGpuStats, "gpu_util_pct"));
  const gpuMemDatasets = $derived(buildGpuDatasets(filteredGpuStats, "mem_util_pct"));
  const gpuTempDatasets = $derived(buildGpuDatasets(filteredGpuStats, "temp_c"));
  const gpuVramTempDatasets = $derived(buildGpuDatasets(filteredGpuStats, "vram_temp_c"));
  const gpuPowerDatasets = $derived(buildGpuDatasets(filteredGpuStats, "power_draw_w"));
  const hasVramTemp = $derived(filteredGpuStats.some((g) => g.vram_temp_c > 0));
</script>

<div class="space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-xl font-semibold text-foreground">性能监控 (实验功能)</h2>
    <div class="flex items-center gap-4">
      <SegmentedControl items={WINDOWS} selected={$selectedWindow} onSelect={(i) => ($selectedWindow = i)} />
      <SegmentedControl
        items={INTERVALS}
        selected={$selectedInterval}
        onSelect={handleIntervalChange}
        label="自动刷新:"
      />
      <Button variant="outline" size="icon-sm" title="立即刷新" onclick={manualRefresh} disabled={refreshing}>
        <RefreshCw class={refreshing ? "animate-spin" : ""} />
      </Button>
    </div>
  </div>
  <p class="text-sm text-muted-foreground">
    此功能处于实验阶段。如需查看指南或反馈建议，请访问 <a
      class="underline hover:text-foreground"
      href="https://github.com/mostlygeek/llama-swap/discussions/771">社区讨论 #771</a
    >。
  </p>

  <!-- GPU Section -->
  <section class="space-y-4">
    <h3 class="text-lg font-medium text-foreground">显卡设备 (GPU)</h3>
    {#if !hasGpuData}
      <Card.Root class="py-0">
        <Card.Content class="p-4">
          <p class="text-muted-foreground">暂无 GPU 监控数据</p>
        </Card.Content>
      </Card.Root>
    {:else}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PerformanceChart
          title="GPU 核心使用率 (%)"
          labels={gpuLabels}
          datasets={gpuUtilDatasets}
          yMin={0}
          yMax={100}
          yLabel="%"
        />
        <PerformanceChart
          title="GPU 显存使用率 (%)"
          labels={gpuLabels}
          datasets={gpuMemDatasets}
          yMin={0}
          yMax={100}
          yLabel="%"
        />
        <PerformanceChart
          title="GPU 核心温度 (°C)"
          labels={gpuLabels}
          datasets={gpuTempDatasets}
          yMin={0}
          yLabel="°C"
        />
        {#if hasVramTemp}
          <PerformanceChart
            title="GPU 显存温度 (°C)"
            labels={gpuLabels}
            datasets={gpuVramTempDatasets}
            yMin={0}
            yLabel="°C"
          />
        {/if}
        <PerformanceChart
          title="GPU 运行功耗 (W)"
          labels={gpuLabels}
          datasets={gpuPowerDatasets}
          yMin={0}
          yLabel="W"
        />
      </div>
    {/if}
  </section>

  <!-- System Section -->
  <section class="space-y-4">
    <h3 class="text-lg font-medium text-foreground">系统资源 (System)</h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <PerformanceChart
        title="CPU 使用率 (%)"
        labels={sysLabels}
        datasets={cpuDatasets}
        yMin={0}
        yMax={100}
        yLabel="%"
        showLegend={false}
      />
      <div>
        <PerformanceChart
          title="物理内存与交换空间占用 (%)"
          labels={sysLabels}
          datasets={memSwapDatasets}
          yMin={0}
          yMax={100}
          yLabel="%"
        />
        {#if latestMemSwap}
          <div class="flex items-center justify-center gap-4 text-xs text-muted-foreground mt-1 px-4">
            <span
              >内存: <span class="text-foreground font-medium"
                >{latestMemSwap.mem_used_mb.toLocaleString()} / {latestMemSwap.mem_total_mb.toLocaleString()} MB ({latestMemSwap.mem_used_pct}%)</span
              ></span
            >
            {#if latestMemSwap.swap_used_pct !== null}
              <span
                >交换空间: <span class="text-foreground font-medium"
                  >{latestMemSwap.swap_used_mb.toLocaleString()} / {latestMemSwap.swap_total_mb.toLocaleString()} MB ({latestMemSwap.swap_used_pct}%)</span
                ></span
              >
            {/if}
          </div>
        {/if}
      </div>
      <PerformanceChart title="系统平均负载 (Load Average)" labels={sysLabels} datasets={loadDatasets} yMin={0} />
      {#if netBandwidthDatasets.length > 0}
        <PerformanceChart
          title="网络带宽 (Mbit/s)"
          labels={netBandwidthLabels}
          datasets={netBandwidthDatasets}
          yMin={0}
          yLabel="Mbit/s"
          showLegend={false}
        />
      {/if}
    </div>
  </section>
</div>
