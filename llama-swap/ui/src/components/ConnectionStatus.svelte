<script lang="ts">
  import { connectionState } from "../stores/theme";

  let eventStatusColor = $derived.by(() => {
    switch ($connectionState) {
      case "connected":
        return "bg-emerald-500";
      case "connecting":
        return "bg-amber-500";
      case "disconnected":
      default:
        return "bg-red-500";
    }
  });

  let statusZh = $derived.by(() => {
    switch ($connectionState) {
      case "connected": return "已连接";
      case "connecting": return "正在连接...";
      case "disconnected": return "已断开";
      default: return "未连接";
    }
  });
  let tooltipText = $derived(`事件流状态: ${statusZh}`);
</script>

<div class="flex items-center" title={tooltipText}>
  <span class="inline-block w-3 h-3 rounded-full {eventStatusColor} mr-2"></span>
</div>
