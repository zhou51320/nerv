<script lang="ts">
  import { proxyLogs, upstreamLogs } from "../stores/api";
  import { screenWidth } from "../stores/theme";
  import { persistentStore } from "../stores/persistent";
  import LogPanel from "../components/LogPanel.svelte";
  import ResizablePanels from "../components/ResizablePanels.svelte";
  import { Tabs, TabsList, TabsTrigger, TabsContent } from "$lib/components/ui/tabs/index.js";

  type ViewMode = "proxy" | "upstream" | "panels";

  const viewModeStore = persistentStore<ViewMode>("logviewer-view-mode", "panels");

  let direction = $derived<"horizontal" | "vertical">(
    $screenWidth === "xs" || $screenWidth === "sm" ? "vertical" : "horizontal",
  );
</script>

<div class="flex flex-col h-full w-full gap-2">
  <Tabs
    value={$viewModeStore}
    onValueChange={(v) => v && viewModeStore.set(v as ViewMode)}
    class="flex flex-1 w-full flex-col gap-2 overflow-hidden"
  >
    <TabsList variant="line">
      <TabsTrigger value="panels">双侧分栏</TabsTrigger>
      <TabsTrigger value="proxy">代理日志</TabsTrigger>
      <TabsTrigger value="upstream">上游日志</TabsTrigger>
    </TabsList>

    <div class="flex-1 w-full overflow-hidden">
      <TabsContent value="panels" class="h-full">
        <ResizablePanels {direction} storageKey="logviewer-panel-group">
          {#snippet leftPanel()}
            <LogPanel id="proxy" title="代理服务日志" logData={$proxyLogs} />
          {/snippet}
          {#snippet rightPanel()}
            <LogPanel id="upstream" title="上游模型日志" logData={$upstreamLogs} />
          {/snippet}
        </ResizablePanels>
      </TabsContent>

      <TabsContent value="proxy" class="h-full">
        <LogPanel id="proxy" title="代理服务日志" logData={$proxyLogs} />
      </TabsContent>

      <TabsContent value="upstream" class="h-full">
        <LogPanel id="upstream" title="上游模型日志" logData={$upstreamLogs} />
      </TabsContent>
    </div>
  </Tabs>
</div>
