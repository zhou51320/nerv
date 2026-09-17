<script lang="ts">
  import { onMount } from "svelte";
  import { getHardware } from "../stores/api";
  import type { HardwareAccelerator, HardwareSnapshot } from "../lib/types";
  import { formatCapacity } from "../lib/format";
  import { copyText } from "../lib/clipboard";
  import { Check, Copy } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Tabs, TabsContent, TabsList, TabsTrigger } from "$lib/components/ui/tabs/index.js";

  let hardware = $state<HardwareSnapshot | null>(null);
  let loading = $state(true);
  let error = $state("");
  let copied = $state(false);

  onMount(async () => {
    try {
      hardware = await getHardware();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Hardware detection unavailable";
    } finally {
      loading = false;
    }
  });

  function shown(value: string | number | null | undefined): string {
    return value === null || value === undefined || value === "" ? "Not detected" : String(value);
  }

  function titleCase(value: string): string {
    return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function osLabel(snapshot: HardwareSnapshot): string {
    return [snapshot.operating_system.name ?? titleCase(snapshot.operating_system.family), snapshot.operating_system.version]
      .filter(Boolean)
      .join(" ");
  }

  function acceleratorTitle(accelerator: HardwareAccelerator): string {
    return accelerator.model ?? `${titleCase(accelerator.kind)} ${accelerator.index + 1}`;
  }

  function environmentLabel(snapshot: HardwareSnapshot): string {
    return `${titleCase(snapshot.environment.kind)}${snapshot.environment.name ? ` (${snapshot.environment.name})` : ""}${snapshot.environment.version ? ` ${snapshot.environment.version}` : ""}`;
  }

  function driverLabel(accelerator: HardwareAccelerator): string {
    return accelerator.driver
      ? [accelerator.driver.name, accelerator.driver.version].filter(Boolean).join(" ") || "Not detected"
      : "Not detected";
  }

  function acceleratorSummary(accelerator: HardwareAccelerator): string[] {
    return [
      `Accelerator ${accelerator.index + 1}: ${acceleratorTitle(accelerator)}`,
      `  Type: ${titleCase(accelerator.kind)}`,
      `  Vendor: ${shown(accelerator.vendor)}`,
      `  Architecture: ${shown(accelerator.architecture)}`,
      `  Memory: ${accelerator.memory.capacity_bytes ? formatCapacity(accelerator.memory.capacity_bytes) : "Not detected"} (${titleCase(accelerator.memory.kind)})`,
      `  Driver: ${driverLabel(accelerator)}`,
      `  Power Limit: ${accelerator.power_limit_watts === null ? "Not detected" : `${accelerator.power_limit_watts} W`}`,
    ];
  }

  function hardwareSummary(snapshot: HardwareSnapshot): string {
    const acceleratorLines = snapshot.accelerators.length === 0
      ? ["No accelerators were detected or exposed to this process."]
      : snapshot.accelerators.flatMap((accelerator, index) => [
          ...(index > 0 ? [""] : []),
          ...acceleratorSummary(accelerator),
        ]);

    return [
      "Hardware Summary",
      "",
      "System",
      `  Operating System: ${osLabel(snapshot)}`,
      `  Kernel: ${shown(snapshot.operating_system.kernel)}`,
      `  Architecture: ${snapshot.architecture.name}`,
      `  Environment: ${environmentLabel(snapshot)}`,
      `  System Memory: ${formatCapacity(snapshot.memory.capacity_bytes)}`,
      "",
      "CPU",
      `  Model: ${shown(snapshot.cpu.model)}`,
      `  Vendor: ${shown(snapshot.cpu.vendor)}`,
      `  Sockets: ${shown(snapshot.cpu.socket_count)}`,
      `  Physical Cores: ${shown(snapshot.cpu.physical_core_count)}`,
      `  Logical Threads: ${shown(snapshot.cpu.logical_thread_count)}`,
      "",
      `Accelerators (${snapshot.accelerators.length})`,
      ...acceleratorLines,
    ].join("\n");
  }

  let summary = $derived(hardware ? hardwareSummary(hardware) : "");

  async function copySummary() {
    if (await copyText(summary)) {
      copied = true;
      window.setTimeout(() => (copied = false), 2000);
    }
  }
</script>

<div class="p-2">
  <div class="mt-4 mb-4">
    <h3 class="text-lg font-semibold">硬件信息</h3>
    <p class="text-sm text-muted-foreground">
      此功能处于实验阶段。如遇问题可在 <a
        class="underline hover:text-foreground"
        href="https://github.com/mostlygeek/llama-swap/issues/977">issue 977</a
      > 反馈。
    </p>
  </div>

  {#if loading}
    <div class="rounded-lg border p-6 text-sm text-muted-foreground">正在检测硬件环境…</div>
  {:else if error || !hardware}
    <div class="rounded-lg border border-destructive/50 p-6">
      <h4 class="font-semibold">硬件检测不可用</h4>
      <p class="mt-1 text-sm text-muted-foreground">{error || "启动时未获取到硬件快照信息。"}</p>
    </div>
  {:else}
    <Tabs value="overview">
      <TabsList variant="line">
        <TabsTrigger value="overview">图文概览</TabsTrigger>
        <TabsTrigger value="summary">纯文本报告</TabsTrigger>
      </TabsList>

      <TabsContent value="overview" class="mt-4">
        <div class="grid gap-4 lg:grid-cols-2">
          <section class="rounded-lg border p-4">
            <h4 class="mb-3 text-sm font-semibold text-muted-foreground">操作系统与平台</h4>
            <dl class="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-4 gap-y-2 text-sm">
              <dt class="text-muted-foreground">操作系统</dt><dd>{osLabel(hardware)}</dd>
              <dt class="text-muted-foreground">内核版本</dt><dd>{shown(hardware.operating_system.kernel)}</dd>
              <dt class="text-muted-foreground">系统架构</dt><dd>{hardware.architecture.name}</dd>
              <dt class="text-muted-foreground">运行环境</dt><dd>{environmentLabel(hardware)}</dd>
              <dt class="text-muted-foreground">系统总物理内存</dt><dd>{formatCapacity(hardware.memory.capacity_bytes)}</dd>
            </dl>
          </section>

          <section class="rounded-lg border p-4">
            <h4 class="mb-3 text-sm font-semibold text-muted-foreground">中央处理器 (CPU)</h4>
            <dl class="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-4 gap-y-2 text-sm">
              <dt class="text-muted-foreground">型号</dt><dd>{shown(hardware.cpu.model)}</dd>
              <dt class="text-muted-foreground">厂商</dt><dd>{shown(hardware.cpu.vendor)}</dd>
              <dt class="text-muted-foreground">物理插槽</dt><dd>{shown(hardware.cpu.socket_count)}</dd>
              <dt class="text-muted-foreground">物理核心数</dt><dd>{shown(hardware.cpu.physical_core_count)}</dd>
              <dt class="text-muted-foreground">逻辑线程数</dt><dd>{shown(hardware.cpu.logical_thread_count)}</dd>
            </dl>
          </section>
        </div>

        <section class="mt-4 rounded-lg border p-4">
          <div class="mb-3 flex items-baseline justify-between gap-4">
            <h4 class="text-sm font-semibold text-muted-foreground">硬件加速器 (GPU / NPU)</h4>
            <span class="text-xs text-muted-foreground">已检测到 {hardware.accelerators.length} 个设备</span>
          </div>
          {#if hardware.accelerators.length === 0}
            <p class="text-sm text-muted-foreground">未检测到或当前进程未获得硬件加速设备权限。</p>
          {:else}
            <div class="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
              {#each hardware.accelerators as accelerator (accelerator.index)}
                <article class="rounded-md border bg-muted/20 p-3">
                  <div class="mb-3">
                    <h5 class="font-medium">{acceleratorTitle(accelerator)}</h5>
                    <p class="text-xs text-muted-foreground">{shown(accelerator.vendor)} · {titleCase(accelerator.kind)}</p>
                  </div>
                  <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
                    <dt class="text-muted-foreground">计算架构</dt><dd>{shown(accelerator.architecture)}</dd>
                    <dt class="text-muted-foreground">显存容量</dt>
                    <dd>{accelerator.memory.capacity_bytes ? formatCapacity(accelerator.memory.capacity_bytes) : "未检测到"} ({titleCase(accelerator.memory.kind)})</dd>
                    <dt class="text-muted-foreground">驱动版本</dt><dd>{driverLabel(accelerator)}</dd>
                    <dt class="text-muted-foreground">功耗上限</dt>
                    <dd>{accelerator.power_limit_watts === null ? "未检测到" : `${accelerator.power_limit_watts} W`}</dd>
                  </dl>
                </article>
              {/each}
            </div>
          {/if}
        </section>
      </TabsContent>

      <TabsContent value="summary" class="mt-4">
        <section class="rounded-lg border p-4">
          <div class="mb-3 flex items-center justify-between gap-4">
            <p class="text-sm text-muted-foreground">纯文本格式，便于在问题报告或技术支持中直接复制粘贴。</p>
            <Button variant="outline" size="sm" onclick={copySummary} title="复制硬件信息">
              {#if copied}
                <Check /> 已复制
              {:else}
                <Copy /> 复制文本
              {/if}
            </Button>
          </div>

          <textarea
            class="min-h-112 w-full resize-y rounded-md border bg-muted/20 p-3 font-mono text-sm leading-5"
            aria-label="Hardware text summary"
            readonly
            value={summary}
          ></textarea>
        </section>
      </TabsContent>
    </Tabs>
  {/if}
</div>
