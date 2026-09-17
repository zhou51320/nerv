<script lang="ts">
  import type { ActivityStatsData } from "../lib/types";
  import { persistentStore } from "../stores/persistent";
  import TokenHistogram from "./TokenHistogram.svelte";
  import { ChevronDown, X } from "@lucide/svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Button } from "$lib/components/ui/button/index.js";

  interface Props {
    stats: ActivityStatsData | null;
  }

  let { stats }: Props = $props();

  const nf = new Intl.NumberFormat();
  const histogramCollapsed = persistentStore<boolean>("activity-histogram-collapsed", false);
</script>

<Card.Root class="relative p-3">
  <Button
    variant="ghost"
    size="icon-xs"
    class="text-muted-foreground absolute right-2 top-2 rounded-full"
    onclick={() => ($histogramCollapsed = !$histogramCollapsed)}
    title={$histogramCollapsed ? "展开统计图表" : "折叠统计图表"}
  >
    {#if $histogramCollapsed}
      <ChevronDown />
    {:else}
      <X />
    {/if}
  </Button>
  {#if !$histogramCollapsed}
    <div class="mb-3 flex flex-col gap-6 sm:flex-row">
      <div class="w-full min-w-0 sm:w-1/2">
        <div class="text-muted-foreground mb-1 text-sm font-medium">提示词处理速度 (Prompt Processing)</div>
        {#if stats?.prompt_histogram}
          <TokenHistogram
            data={stats.prompt_histogram}
            unit="tokens/秒"
            colorClass="text-amber-500 dark:text-amber-400"
          />
        {:else}
          <div class="text-muted-foreground py-6 text-center text-sm">暂无提示词处理速度数据</div>
        {/if}
      </div>
      <div class="w-full min-w-0 sm:w-1/2">
        <div class="text-muted-foreground mb-1 text-sm font-medium">文本生成速度 (Token Generation)</div>
        {#if stats?.gen_histogram}
          <TokenHistogram data={stats.gen_histogram} unit="tokens/秒" />
        {:else}
          <div class="text-muted-foreground py-6 text-center text-sm">暂无文本生成速度数据</div>
        {/if}
      </div>
    </div>
  {/if}
  <div class="grid grid-cols-4 gap-x-6 gap-y-1 text-sm">
    <div class="text-muted-foreground text-xs uppercase tracking-wider">请求次数</div>
    <div class="text-muted-foreground text-xs uppercase tracking-wider">缓存命中</div>
    <div class="text-muted-foreground text-xs uppercase tracking-wider">输入词元</div>
    <div class="text-muted-foreground text-xs uppercase tracking-wider">生成词元</div>
    <div class="text-sm">
      <span class="font-semibold">{nf.format(stats?.total_requests ?? 0)}</span> 次请求
    </div>
    <div class="text-sm">
      <span class="font-semibold">{nf.format(stats?.total_cache_tokens ?? 0)}</span> tokens
    </div>
    <div class="text-sm">
      <span class="font-semibold">{nf.format(stats?.total_input_tokens ?? 0)}</span> tokens
    </div>
    <div class="text-sm">
      <span class="font-semibold">{nf.format(stats?.total_output_tokens ?? 0)}</span> tokens
    </div>
  </div>
</Card.Root>
