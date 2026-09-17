<script lang="ts">
  import { connectionState, themeName, themeMode, themes, type ThemeMode } from "../stores/theme";
  import { versionInfo } from "../stores/api";
  import { showCapabilityTags } from "../stores/modelDisplay";
  import { showGenerationStats } from "../stores/generationStats";
  import * as Select from "$lib/components/ui/select/index.js";
  import * as Switch from "$lib/components/ui/switch/index.js";
  import * as Label from "$lib/components/ui/label/index.js";

  const modes: { value: ThemeMode; label: string }[] = [
    { value: "light", label: "浅色模式" },
    { value: "dark", label: "深色模式" },
    { value: "system", label: "跟随系统" },
  ];

  let themeLabel = $derived(themes.find((t) => t.value === $themeName)?.label ?? "默认");
  let modeLabel = $derived(modes.find((m) => m.value === $themeMode)?.label ?? "跟随系统");
</script>

<div class="p-2">
  <div class="mt-4 mb-4">
    <h3 class="text-lg font-semibold">系统设置</h3>
  </div>

  <div class="rounded-lg border p-4 space-y-3 max-w-md mb-4">
    <h4 class="text-sm font-semibold text-muted-foreground">外观</h4>
    <div class="flex items-center justify-between gap-4">
      <span class="text-sm">配色主题</span>
      <Select.Root
        type="single"
        value={$themeName}
        onValueChange={(v) => v && themeName.set(v as typeof $themeName)}
      >
        <Select.Trigger class="w-40">{themeLabel}</Select.Trigger>
        <Select.Content>
          {#each themes as theme (theme.value)}
            <Select.Item value={theme.value}>{theme.label}</Select.Item>
          {/each}
        </Select.Content>
      </Select.Root>
    </div>
    <div class="flex items-center justify-between gap-4">
      <span class="text-sm">色彩模式</span>
      <Select.Root
        type="single"
        value={$themeMode}
        onValueChange={(v) => v && themeMode.set(v as ThemeMode)}
      >
        <Select.Trigger class="w-40">{modeLabel}</Select.Trigger>
        <Select.Content>
          {#each modes as mode (mode.value)}
            <Select.Item value={mode.value}>{mode.label}</Select.Item>
          {/each}
        </Select.Content>
      </Select.Root>
    </div>
  </div>

  <div class="rounded-lg border p-4 space-y-3 max-w-md mb-4">
    <h4 class="text-sm font-semibold text-muted-foreground">模型列表页</h4>
    <div class="flex items-start justify-between gap-4">
      <div>
        <Label.Root for="show-capability-tags" class="text-sm">显示模型能力标签</Label.Root>
        <p class="text-muted-foreground text-xs">
          在每个模型名称旁展示所有能力徽章（如视觉、工具调用、上下文窗口长度等）。
        </p>
      </div>
      <Switch.Root
        id="show-capability-tags"
        checked={$showCapabilityTags}
        onCheckedChange={(v) => showCapabilityTags.set(v)}
      />
    </div>
  </div>

  <div class="rounded-lg border p-4 space-y-3 max-w-md mb-4">
    <h4 class="text-sm font-semibold text-muted-foreground">对话交互</h4>
    <div class="flex items-start justify-between gap-4">
      <div>
        <Label.Root for="show-generation-stats" class="text-sm">显示生成速率统计</Label.Root>
        <p class="text-muted-foreground text-xs">
          在演练场对话消息下方显示 Token 数量、总耗时及生成速度（tokens/s）。
        </p>
      </div>
      <Switch.Root
        id="show-generation-stats"
        checked={$showGenerationStats}
        onCheckedChange={(v) => showGenerationStats.set(v)}
      />
    </div>
  </div>

  <div class="rounded-lg border p-4 space-y-2 max-w-md">
    <h4 class="text-sm font-semibold text-muted-foreground">构建版本信息</h4>
    <dl class="text-sm space-y-1">
      <div class="flex justify-between gap-4">
        <dt class="text-muted-foreground">事件流状态</dt>
        <dd class="font-medium">{$connectionState ?? "未连接"}</dd>
      </div>
      <div class="flex justify-between gap-4">
        <dt class="text-muted-foreground">版本号</dt>
        <dd class="font-medium">{$versionInfo?.version ?? "未知"}</dd>
      </div>
      <div class="flex justify-between gap-4">
        <dt class="text-muted-foreground">Git Commit</dt>
        <dd class="font-medium">{$versionInfo?.commit?.substring(0, 7) ?? "未知"}</dd>
      </div>
      <div class="flex justify-between gap-4">
        <dt class="text-muted-foreground">编译时间</dt>
        <dd class="font-medium">{$versionInfo?.build_date ?? "未知"}</dd>
      </div>
    </dl>
  </div>
</div>
