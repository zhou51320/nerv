import { get } from "svelte/store";
import { persistentStore } from "./persistent";

export type PlaygroundTab = "chat" | "images" | "speech" | "audio" | "rerank" | "concurrency";

export const playgroundTabs: { id: PlaygroundTab; label: string }[] = [
  { id: "chat", label: "对话交互" },
  { id: "images", label: "图像生成" },
  { id: "speech", label: "语音合成" },
  { id: "audio", label: "语音转录" },
  { id: "rerank", label: "文本重排" },
  { id: "concurrency", label: "并发测试" },
];

export const selectedPlaygroundTab = persistentStore<PlaygroundTab>("playground-selected-tab", "chat");

/**
 * Drop a stored tab that no longer exists.
 *
 * "docs" was a tab here until Help became its own page. A browser that last
 * left the Playground on it would select a tab with nothing behind it: every
 * panel hidden, and a header reading "Playground /".
 */
export function resetUnknownPlaygroundTab(): void {
  if (!playgroundTabs.some((tab) => tab.id === get(selectedPlaygroundTab))) {
    selectedPlaygroundTab.set(playgroundTabs[0].id);
  }
}

resetUnknownPlaygroundTab();
