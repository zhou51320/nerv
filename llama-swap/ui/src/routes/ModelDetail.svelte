<script lang="ts">
  import { params } from "svelte-spa-router";
  import { models } from "../stores/api";
  import { statusDotColor } from "../stores/modelLoad";
  import type { Model } from "../lib/types";
  import ModelLoadButton from "../components/ModelLoadButton.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Tabs, TabsList, TabsTrigger, TabsContent } from "$lib/components/ui/tabs/index.js";
  import { ExternalLink } from "@lucide/svelte";
  import ModelActivityTab from "../components/model/ModelActivityTab.svelte";
  import ModelLogsTab from "../components/model/ModelLogsTab.svelte";
  import ModelDetailsTab from "../components/model/ModelDetailsTab.svelte";
  import { modelServerPath } from "../lib/modelUtils";

  let modelId = $derived($params?.id ?? "");

  // Resolve the route param to a model record by ID, falling back to an
  // alias match so links to alias targets (e.g. selector targets) resolve.
  let model = $derived<Model | undefined>(
    $models.find((m) => m.id === modelId) ??
      $models.find((m) => m.aliases?.includes(modelId)),
  );
  let resolvedId = $derived(model?.id ?? modelId);
</script>

<div class="flex h-full flex-col gap-4 overflow-y-auto p-2">
  {#if !model}
    <Card.Root class="shrink-0 p-6">
      <p class="text-muted-foreground">未找到模型 “{modelId}”。</p>
      <a href="/" class="text-primary hover:underline">返回首页</a>
    </Card.Root>
  {:else}
    <Card.Root class="shrink-0 gap-0 overflow-hidden py-0">
      <Card.Header class="shrink-0 gap-2 border-b px-4 py-3">
        <div class="flex items-center gap-2">
          <span class={`size-2.5 shrink-0 rounded-full ${statusDotColor(model)}`}></span>
          <Card.Title class="text-lg">{model.name || model.id}</Card.Title>
          <span class="text-muted-foreground text-sm">({model.id})</span>
          <span class="text-muted-foreground text-xs font-medium">
            {model.state === "ready" ? "已就绪" : model.state === "starting" ? "启动中" : model.state === "stopping" ? "停止中" : "已停止"}
          </span>
          <div class="ml-auto flex items-center gap-2">
            {#if !model.peerID}
              <a
                href={modelServerPath(resolvedId)}
                target="_blank"
                rel="noopener noreferrer"
                class="text-muted-foreground hover:text-foreground"
                title="打开模型服务链接"
                aria-label="打开模型服务链接"
              >
                <ExternalLink class="size-4" />
              </a>
              <ModelLoadButton {model} size="sm" />
            {/if}
          </div>
        </div>
        {#if model.description}
          <p class="text-muted-foreground text-sm"><em>{model.description}</em></p>
        {/if}
        {#if model.aliases && model.aliases.length > 0}
          <p class="text-muted-foreground text-xs">别名: {model.aliases.join(", ")}</p>
        {/if}
      </Card.Header>
    </Card.Root>

    <Tabs value="activity" class="min-h-0 flex-1">
      <TabsList variant="line">
        <TabsTrigger value="activity">活动记录</TabsTrigger>
        <TabsTrigger value="logs">运行日志</TabsTrigger>
        <TabsTrigger value="details">配置详情</TabsTrigger>
      </TabsList>

      <!-- Activity -->
      <TabsContent value="activity">
        <ModelActivityTab modelId={resolvedId} />
      </TabsContent>

      <!-- Logs -->
      <TabsContent value="logs" class="min-h-0 flex-1">
        <ModelLogsTab modelId={resolvedId} />
      </TabsContent>

      <!-- Details -->
      <TabsContent value="details">
        <ModelDetailsTab model={model} />
      </TabsContent>
    </Tabs>
  {/if}
</div>
