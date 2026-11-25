"use strict";
// Compare two file:// viewers through postMessage; no server or frame DOM access.
const embedded = window.parent !== window;
if (embedded) document.body.classList.add("comparisonEmbedded");
const baseLoadFrame = loadFrame;
function announceFrame() {
  if (embedded && manifest && currentCloud) window.parent.postMessage({
    type: "mask-comparison:loaded", sequence: manifest.frames[currentFrameIndex].sequence_index,
    sourceFrame: manifest.frames[currentFrameIndex].source_frame_id,
  }, "*");
}
loadFrame = async function(index) { await baseLoadFrame(index); announceFrame(); };
window.addEventListener("message", async (event) => {
  if (!embedded || event.source !== window.parent || !manifest || !glState) return;
  const message = event.data;
  if (message?.type === "mask-comparison:frame" && Number.isInteger(message.sequence)) {
    const index = manifest.frames.findIndex(f => f.sequence_index === message.sequence);
    if (index < 0) return;
    try {
      if (!currentCloud || currentFrameIndex !== index) await loadFrame(index);
      else announceFrame();
    } catch (error) { showError(error); }
  } else if (message?.type === "mask-comparison:mode" && ["rgb", "all"].includes(message.mode)) {
    setImageMode(message.mode);
  } else if (message?.type === "mask-comparison:reset") {
    $("resetView").click();
  }
});
const readyTimer = setInterval(() => {
  if (!currentCloud || !manifest) return;
  clearInterval(readyTimer);
  setImageMode("all");
  if (embedded) window.parent.postMessage({type: "mask-comparison:ready"}, "*");
}, 100);

if (embedded) {
  new ResizeObserver(() => window.parent.postMessage({
    type: "mask-comparison:height", height: Math.ceil(document.body.getBoundingClientRect().height)
  }, "*")).observe(document.body);
}
