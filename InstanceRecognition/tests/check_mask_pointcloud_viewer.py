#!/usr/bin/env python3
"""Actual file:// browser checks using the locally installed Chrome and websocket-client.

Run with system python3. No model or web server is started; a temporary isolated
Chrome profile is used and its process is terminated when checks finish.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket


class Browser:
    def __init__(self, socket_url: str):
        self.socket = websocket.create_connection(socket_url, timeout=45, suppress_origin=True)
        self.next_id = 0
        self.exceptions = []

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        request_id = self.next_id
        self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                self.exceptions.append(message["params"])
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def js(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait(self, expression: str, seconds: float = 40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.js(expression):
                return
            time.sleep(0.1)
        raise TimeoutError(f"Browser condition timed out: {expression}")

    def screenshot(self, path: Path):
        result = self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(result["data"]))

    def image_signature(self):
        # file:// RGB images taint the canvas; compare rendered pixels via CDP
        # without changing the browser's local-file security settings.
        clip = self.js("""
          (() => {
            const rect = document.getElementById('imageCanvas').getBoundingClientRect();
            return {x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height, scale: 1};
          })()
        """)
        result = self.call("Page.captureScreenshot", {"format": "png", "clip": clip})
        return hashlib.sha256(base64.b64decode(result["data"])).hexdigest()

    def mouse(self, event_type: str, x: float, y: float, *, button: str = "none", buttons: int = 0, delta_y: float = 0):
        params = {"type": event_type, "x": x, "y": y, "button": button, "buttons": buttons}
        if event_type == "mouseWheel":
            params["deltaX"] = 0
            params["deltaY"] = delta_y
        else:
            params["clickCount"] = 1
        return self.call("Input.dispatchMouseEvent", params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="mask-projection-chrome-"))
    results = {"page": str(args.page.resolve()), "profile": str(profile), "screenshots": [], "checks": {}}
    log = (args.output / "browser.log").open("wb")
    process = subprocess.Popen([
        "/usr/bin/google-chrome", "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
        "--disable-background-networking", "--disable-extensions", "--no-first-run",
        "--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--remote-debugging-port=0",
        f"--user-data-dir={profile}", "--window-size=1500,1050", "about:blank",
    ], stdout=log, stderr=log)
    browser = None
    try:
        port_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while not port_file.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Chrome exited before DevTools became available")
            time.sleep(0.1)
        port = int(port_file.read_text().splitlines()[0])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
            targets = json.load(response)
        target = next(item for item in targets if item["type"] == "page")
        browser = Browser(target["webSocketDebuggerUrl"])
        browser.call("Runtime.enable")
        browser.call("Page.enable")
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 1500, "height": 1050, "deviceScaleFactor": 1, "mobile": False})
        browser.call("Page.navigate", {"url": args.page.resolve().as_uri()})
        browser.wait("typeof currentCloud !== 'undefined' && currentCloud && document.getElementById('status').textContent.startsWith('已加载')")
        browser.js("""
          window.__contextMenuEvents = [];
          document.addEventListener('contextmenu', (event) => {
            window.__contextMenuEvents.push({
              target: event.target.id || event.target.tagName,
              defaultPrevented: event.defaultPrevented,
              x: event.clientX,
              y: event.clientY
            });
          });
        """)
        stats = browser.js("({frames:manifest.frames.length, options:document.getElementById('frameSelect').options.length, points:glState.drawCount, expected:manifest.frames[0].point_count, background:glState.bgCount, error:glState.gl.getError()})")
        assert stats["frames"] == stats["options"] == 500, stats
        assert stats["points"] == stats["expected"] and stats["background"] == 491666 and stats["error"] == 0, stats
        results["checks"]["initial"] = stats
        browser.wait("selectedInstanceIndices instanceof Set && !!document.getElementById('modeSelectedMasks')")
        browser.js("document.getElementById('modeAllMasks').click()")
        browser.wait("imageMode === 'all' && selectedInstanceIndices.size === 0")
        time.sleep(0.3)
        browser.screenshot(args.output / "frame_000000_all_masks.png")
        results["screenshots"].append("frame_000000_all_masks.png")

        browser.js("""
          window.__maskTest = {
            expectedSelectedPoints(indices) {
              const selected = new Set(indices);
              let count = 0;
              for (const bits of currentCloud.maskBits) {
                for (const index of selected) {
                  if (bits & (1 << index)) {
                    count += 1;
                    break;
                  }
                }
              }
              return count;
            },
            labels() {
              return Array.from(document.querySelectorAll('#maskLabels .maskLabel')).map((node) => ({
                index: Number(node.dataset.instanceIndex),
                text: node.textContent
              }));
            },
            selected() {
              return Array.from(selectedInstanceIndices).sort((a, b) => a - b);
            },
            clickInstance(index) {
              document.querySelector(`#instanceList button[data-instance-index="${index}"]`).click();
            },
            frameInfo() {
              return {
                index: currentFrameIndex,
                sourceFrameId: manifest.frames[currentFrameIndex].source_frame_id,
                imageMode,
                selected: this.selected(),
                drawCount: glState.drawCount,
                framePointCount: manifest.frames[currentFrameIndex].point_count,
                expectedSelected: this.expectedSelectedPoints(this.selected()),
                labels: this.labels(),
                activeButtons: Array.from(document.querySelectorAll('#instanceList button.active')).map((node) => Number(node.dataset.instanceIndex)),
                buttonTexts: Array.from(document.querySelectorAll('#instanceList button')).map((node) => node.textContent)
              };
            }
          };
        """)

        browser.js("document.getElementById('modeRgb').click()")
        browser.wait("imageMode === 'rgb' && selectedInstanceIndices.size === 0")
        rgb_signature = browser.image_signature()
        browser.js("__maskTest.clickInstance(0)")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.has(0) && glState.drawCount === __maskTest.expectedSelectedPoints([0])")
        time.sleep(0.3)
        selected_first = browser.js("__maskTest.frameInfo()")
        assert selected_first["selected"] == [0], selected_first
        assert selected_first["drawCount"] == selected_first["expectedSelected"], selected_first
        assert selected_first["labels"] and selected_first["labels"][0]["index"] == 0, selected_first
        assert str(browser.js("manifest.frames[0].instances[0].instance_id")) in selected_first["labels"][0]["text"], selected_first
        assert browser.js("manifest.frames[0].instances[0].class_name") in selected_first["labels"][0]["text"], selected_first
        assert browser.js("getComputedStyle(document.querySelector('#instanceList button.active .name')).color") == "rgb(238, 244, 255)"
        assert browser.js("document.querySelector('#instanceList button.active').getAttribute('aria-pressed')") == "true"
        browser.screenshot(args.output / "frame_000000_single_selected.png")
        results["screenshots"].append("frame_000000_single_selected.png")

        browser.js("__maskTest.clickInstance(1)")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.has(0) && selectedInstanceIndices.has(1) && glState.drawCount === __maskTest.expectedSelectedPoints([0, 1])")
        multi_selected = browser.js("__maskTest.frameInfo()")
        assert multi_selected["selected"] == [0, 1], multi_selected
        assert sorted(label["index"] for label in multi_selected["labels"]) == [0, 1], multi_selected
        assert multi_selected["drawCount"] == multi_selected["expectedSelected"], multi_selected

        browser.js("__maskTest.clickInstance(0)")
        browser.wait("imageMode === 'selected' && !selectedInstanceIndices.has(0) && selectedInstanceIndices.has(1) && glState.drawCount === __maskTest.expectedSelectedPoints([1])")
        cancelled_one = browser.js("__maskTest.frameInfo()")
        assert cancelled_one["selected"] == [1], cancelled_one
        assert [label["index"] for label in cancelled_one["labels"]] == [1], cancelled_one

        browser.js("__maskTest.clickInstance(1)")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.size === 0 && glState.drawCount === 0 && __maskTest.labels().length === 0")
        cancelled_last = {"frame": browser.js("__maskTest.frameInfo()"), "image": browser.image_signature()}
        assert cancelled_last["frame"]["selected"] == [], cancelled_last
        assert cancelled_last["frame"]["drawCount"] == 0, cancelled_last
        assert cancelled_last["frame"]["labels"] == [], cancelled_last
        assert browser.js("document.querySelectorAll('#instanceList button[aria-pressed=\"true\"]').length") == 0
        assert cancelled_last["image"] == rgb_signature, {"rgb": rgb_signature, "after": cancelled_last}

        browser.js("__maskTest.clickInstance(0); document.getElementById('modeRgb').click()")
        browser.wait("imageMode === 'rgb' && selectedInstanceIndices.has(0) && glState.drawCount === manifest.frames[currentFrameIndex].point_count")
        rgb_keeps_selection = {"frame": browser.js("__maskTest.frameInfo()"), "image": browser.image_signature()}
        assert rgb_keeps_selection["frame"]["selected"] == [0], rgb_keeps_selection
        assert rgb_keeps_selection["frame"]["drawCount"] == rgb_keeps_selection["frame"]["framePointCount"], rgb_keeps_selection
        assert rgb_keeps_selection["image"] == rgb_signature, {"rgb": rgb_signature, "after": rgb_keeps_selection}

        browser.js("document.getElementById('modeSelectedMasks').click()")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.has(0) && glState.drawCount === __maskTest.expectedSelectedPoints([0])")
        browser.js("document.getElementById('modeAllMasks').click()")
        browser.wait("imageMode === 'all' && selectedInstanceIndices.size === 0 && glState.drawCount === manifest.frames[currentFrameIndex].point_count")
        all_clears_selection = browser.js("__maskTest.frameInfo()")
        assert all_clears_selection["selected"] == [] and all_clears_selection["drawCount"] == all_clears_selection["framePointCount"], all_clears_selection

        browser.js("document.getElementById('frameSelect').value='90';document.getElementById('frameSelect').dispatchEvent(new Event('change'))")
        browser.wait("currentFrameIndex === 90 && document.getElementById('status').textContent.startsWith('已加载')")
        browser.js("__maskTest.clickInstance(0)")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.has(0) && glState.drawCount === __maskTest.expectedSelectedPoints([0])")
        guitar_selected = browser.js("__maskTest.frameInfo()")
        assert guitar_selected["sourceFrameId"] == 1006, guitar_selected
        assert "guitar" in guitar_selected["labels"][0]["text"], guitar_selected
        browser.screenshot(args.output / "frame_001006_guitar_selected.png")
        results["screenshots"].append("frame_001006_guitar_selected.png")
        browser.js("__maskTest.clickInstance(1)")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.has(0) && selectedInstanceIndices.has(1) && glState.drawCount === __maskTest.expectedSelectedPoints([0, 1])")
        browser.screenshot(args.output / "frame_001006_two_selected.png")
        results["screenshots"].append("frame_001006_two_selected.png")
        two_selected_1006 = browser.js("__maskTest.frameInfo()")
        assert two_selected_1006["selected"] == [0, 1], two_selected_1006
        assert sorted(label["index"] for label in two_selected_1006["labels"]) == [0, 1], two_selected_1006

        browser.js("document.getElementById('frameSelect').value='91';document.getElementById('frameSelect').dispatchEvent(new Event('change'))")
        browser.wait("currentFrameIndex === 91 && document.getElementById('status').textContent.startsWith('已加载')")
        switched_frame = browser.js("__maskTest.frameInfo()")
        assert switched_frame["imageMode"] == "selected", switched_frame
        assert switched_frame["selected"] == [] and switched_frame["drawCount"] == 0 and switched_frame["labels"] == [], switched_frame

        browser.js("document.getElementById('frameSelect').value='0';document.getElementById('frameSelect').dispatchEvent(new Event('change'))")
        browser.wait("currentFrameIndex === 0 && document.getElementById('status').textContent.startsWith('已加载') && selectedInstanceIndices.size === 0")
        browser.js("__maskTest.clickInstance(5); __maskTest.clickInstance(7)")
        browser.wait("imageMode === 'selected' && selectedInstanceIndices.has(5) && selectedInstanceIndices.has(7) && glState.drawCount === __maskTest.expectedSelectedPoints([5, 7])")
        same_name = browser.js("__maskTest.frameInfo()")
        assert same_name["selected"] == [5, 7], same_name
        assert sorted(label["index"] for label in same_name["labels"]) == [5, 7], same_name
        same_name_text = " ".join(label["text"] for label in same_name["labels"])
        assert "American shorthair" in same_name_text and "6" in same_name_text and "8" in same_name_text, same_name

        browser.js("""
          loadFrame(90);
          loadFrame(0);
        """)
        browser.wait("currentFrameIndex === 0 && document.getElementById('status').textContent.startsWith('已加载')")
        browser.js("__maskTest.clickInstance(5); __maskTest.clickInstance(5); __maskTest.clickInstance(7)")
        browser.wait("currentFrameIndex === 0 && imageMode === 'selected' && selectedInstanceIndices.size === 1 && selectedInstanceIndices.has(7)")
        time.sleep(0.5)
        rapid_toggle = browser.js("__maskTest.frameInfo()")
        assert rapid_toggle["index"] == 0 and rapid_toggle["selected"] == [7], rapid_toggle
        assert rapid_toggle["drawCount"] == rapid_toggle["expectedSelected"], rapid_toggle

        results["checks"]["mask_selection"] = {
            "single": selected_first,
            "multi": multi_selected,
            "cancelled_one": cancelled_one,
            "cancelled_last": cancelled_last,
            "rgb_keeps_selection": rgb_keeps_selection["frame"],
            "all_clears_selection": all_clears_selection,
            "guitar_frame_1006": guitar_selected,
            "two_selected_frame_1006": two_selected_1006,
            "frame_switch_clears_selection": switched_frame,
            "same_name_independent_ids": same_name,
            "rapid_toggle": rapid_toggle,
        }

        browser.js("document.getElementById('modeAllMasks').click()")
        browser.wait("imageMode === 'all' && selectedInstanceIndices.size === 0 && glState.drawCount === manifest.frames[currentFrameIndex].point_count")
        checked_frames = []
        for sequence in [1, 79, 150, 249, 350, 420, 499]:
            browser.js(f"document.getElementById('frameSelect').value='{sequence}';document.getElementById('frameSelect').dispatchEvent(new Event('change'))")
            browser.wait(f"currentFrameIndex === {sequence} && selectedInstanceIndices.size === 0 && document.getElementById('status').textContent.startsWith('已加载')")
            stats = browser.js("({index:currentFrameIndex,frame:manifest.frames[currentFrameIndex].source_frame_id,mode:imageMode,selected:__maskTest.selected(),points:glState.drawCount,expected:manifest.frames[currentFrameIndex].point_count,error:glState.gl.getError(),cache:frameCache.size,registry:registry.size})")
            assert stats["points"] == stats["expected"] and stats["error"] == 0, stats
            assert stats["mode"] == "all" and stats["selected"] == [], stats
            assert stats["cache"] <= 3 and stats["registry"] <= 4, stats
            checked_frames.append(stats)
        results["checks"]["frame_switches"] = checked_frames
        browser.screenshot(args.output / "frame_005577_all_masks.png")
        results["screenshots"].append("frame_005577_all_masks.png")
        browser.js("loadFrame(10);loadFrame(20);loadFrame(30)")
        browser.wait("currentFrameIndex === 30 && document.getElementById('status').textContent.startsWith('已加载')")
        results["checks"]["rapid_switch"] = browser.js("({index:currentFrameIndex,selected:__maskTest.selected(),points:glState.drawCount,expected:manifest.frames[30].point_count})")
        assert results["checks"]["rapid_switch"]["selected"] == [], results["checks"]["rapid_switch"]

        def canvas_point(offset_x=0, offset_y=0):
            rect = browser.js("""
              (() => {
                const rect = document.getElementById('cloudCanvas').getBoundingClientRect();
                return {left: rect.left, top: rect.top, width: rect.width, height: rect.height};
              })()
            """)
            return rect["left"] + rect["width"] / 2 + offset_x, rect["top"] + rect["height"] / 2 + offset_y

        def view_state():
            return browser.js("""
              (() => ({
                rotationX: glState.rotationX,
                rotationY: glState.rotationY,
                distance: glState.distance,
                panX: glState.panX,
                panY: glState.panY,
                matrix: Array.from(computeMatrix(glState.canvas.width / glState.canvas.height))
              }))()
            """)

        def canvas_hash():
            return browser.js("""
              (() => {
                renderCloud();
                const gl = glState.gl;
                const width = gl.drawingBufferWidth;
                const height = gl.drawingBufferHeight;
                const pixels = new Uint8Array(width * height * 4);
                gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                let hash = 2166136261;
                let nonBackground = 0;
                for (let i = 0; i < pixels.length; i += 4) {
                  hash ^= pixels[i] + (pixels[i + 1] << 8) + (pixels[i + 2] << 16) + (pixels[i + 3] << 24);
                  hash = Math.imul(hash, 16777619) >>> 0;
                  if (pixels[i] !== 5 || pixels[i + 1] !== 8 || pixels[i + 2] !== 13) nonBackground += 1;
                }
                return {width, height, hash, nonBackground};
              })()
            """)

        def assert_close(actual, expected, label, epsilon=1e-9):
            assert abs(actual - expected) <= epsilon, f"{label}: expected {expected}, got {actual}"

        def assert_state_unchanged(actual, expected, label):
            for key in ["rotationX", "rotationY", "distance", "panX", "panY"]:
                assert_close(actual[key], expected[key], f"{label}.{key}")

        browser.js("document.getElementById('resetView').click()")
        browser.wait("typeof glState.panX === 'number' && typeof glState.panY === 'number' && glState.panX === 0 && glState.panY === 0")
        before_pan = view_state()
        before_pan_pixels = canvas_hash()
        x, y = canvas_point()
        browser.mouse("mousePressed", x, y, button="right", buttons=2)
        browser.mouse("mouseMoved", x + 160, y + 90, button="right", buttons=2)
        browser.mouse("mouseReleased", x + 160, y + 90, button="right", buttons=0)
        time.sleep(0.2)
        after_pan = view_state()
        after_pan_pixels = canvas_hash()
        assert abs(after_pan["panX"]) > 1e-6 or abs(after_pan["panY"]) > 1e-6, after_pan
        assert_close(after_pan["rotationX"], before_pan["rotationX"], "right_drag.rotationX")
        assert_close(after_pan["rotationY"], before_pan["rotationY"], "right_drag.rotationY")
        assert after_pan["matrix"] != before_pan["matrix"], {"before": before_pan["matrix"], "after": after_pan["matrix"]}
        assert after_pan_pixels["hash"] != before_pan_pixels["hash"], {"before": before_pan_pixels, "after": after_pan_pixels}
        assert after_pan_pixels["nonBackground"] > 0, after_pan_pixels
        after_release = view_state()
        browser.mouse("mouseMoved", x + 260, y + 160, button="none", buttons=0)
        time.sleep(0.1)
        assert_state_unchanged(view_state(), after_release, "released_drag")

        browser.js("window.__contextMenuEvents = []")
        browser.mouse("mousePressed", x, y, button="right", buttons=2)
        browser.mouse("mouseReleased", x, y, button="right", buttons=0)
        browser.wait("window.__contextMenuEvents.length > 0", seconds=5)
        context_menu = browser.js("window.__contextMenuEvents[window.__contextMenuEvents.length - 1]")
        assert context_menu["target"] == "cloudCanvas" and context_menu["defaultPrevented"], context_menu

        browser.js("document.getElementById('resetView').click()")
        browser.wait("glState.panX === 0 && glState.panY === 0")
        before_rotate = view_state()
        x, y = canvas_point()
        browser.mouse("mousePressed", x, y, button="left", buttons=1)
        browser.mouse("mouseMoved", x + 120, y - 60, button="left", buttons=1)
        browser.mouse("mouseReleased", x + 120, y - 60, button="left", buttons=0)
        time.sleep(0.1)
        after_rotate = view_state()
        assert abs(after_rotate["rotationX"] - before_rotate["rotationX"]) > 1e-6 or abs(after_rotate["rotationY"] - before_rotate["rotationY"]) > 1e-6, after_rotate
        assert_close(after_rotate["panX"], before_rotate["panX"], "left_drag.panX")
        assert_close(after_rotate["panY"], before_rotate["panY"], "left_drag.panY")

        before_wheel = view_state()
        x, y = canvas_point()
        browser.mouse("mouseWheel", x, y, delta_y=-420)
        time.sleep(0.1)
        after_wheel = view_state()
        assert after_wheel["distance"] < before_wheel["distance"], {"before": before_wheel, "after": after_wheel}

        browser.js("document.getElementById('resetView').click()")
        browser.wait("glState.rotationX === -0.35 && glState.rotationY === 0.45 && glState.distance === 3.2 && glState.panX === 0 && glState.panY === 0")
        results["checks"]["controls_reset"] = view_state()

        x, y = canvas_point()
        browser.mouse("mousePressed", x, y, button="right", buttons=2)
        browser.mouse("mouseMoved", x + 80, y + 50, button="right", buttons=2)
        state_before_cancel = view_state()
        browser.js("""
          (() => {
            const canvas = document.getElementById('cloudCanvas');
            canvas.dispatchEvent(new PointerEvent('pointercancel', {
              bubbles: true,
              pointerId: 1,
              pointerType: 'mouse',
              button: 2,
              buttons: 2,
              clientX: 0,
              clientY: 0
            }));
          })()
        """)
        browser.mouse("mouseMoved", x + 180, y + 120, button="right", buttons=2)
        browser.mouse("mouseReleased", x + 180, y + 120, button="right", buttons=0)
        time.sleep(0.1)
        assert_state_unchanged(view_state(), state_before_cancel, "cancelled_drag")
        results["checks"]["controls"] = {
            "right_drag_pan": {"before": before_pan, "after": after_pan, "pixels_before": before_pan_pixels, "pixels_after": after_pan_pixels},
            "contextmenu": context_menu,
            "left_drag_rotate": {"before": before_rotate, "after": after_rotate},
            "wheel_zoom": {"before": before_wheel["distance"], "after": after_wheel["distance"]},
            "cancelled_drag": state_before_cancel,
        }

        browser.js("document.getElementById('colorMode').click();document.getElementById('showBackground').click();document.getElementById('resetView').click()")
        assert browser.js("glState.gl.getError()") == 0
        results["checks"]["desktop_no_horizontal_overflow"] = browser.js("document.documentElement.scrollWidth <= innerWidth")
        assert results["checks"]["desktop_no_horizontal_overflow"]
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 440, "height": 900, "deviceScaleFactor": 1, "mobile": True})
        time.sleep(0.4)
        results["checks"]["mobile_no_horizontal_overflow"] = browser.js("document.documentElement.scrollWidth <= innerWidth")
        assert results["checks"]["mobile_no_horizontal_overflow"]
        browser.screenshot(args.output / "mobile.png")
        results["screenshots"].append("mobile.png")
        browser.js("loadFrame(90)")
        browser.wait("currentFrameIndex === 90 && document.getElementById('status').textContent.startsWith('已加载')")
        browser.js("__maskTest.clickInstance(0); __maskTest.clickInstance(1); document.getElementById('imageCanvas').scrollIntoView({block: 'start'})")
        browser.wait("selectedInstanceIndices.size === 2 && __maskTest.labels().length === 2")
        time.sleep(0.2)
        results["checks"]["mobile_selected_labels_inside_image"] = browser.js("""
          (() => {
            const image = document.getElementById('imageCanvas').getBoundingClientRect();
            return Array.from(document.querySelectorAll('#maskLabels .maskLabel')).every(node => {
              const rect = node.getBoundingClientRect();
              return rect.left >= image.left && rect.top >= image.top && rect.right <= image.right && rect.bottom <= image.bottom;
            });
          })()
        """)
        assert results["checks"]["mobile_selected_labels_inside_image"]
        results["checks"]["mobile_button_content_inside_rows"] = browser.js("""
          Array.from(document.querySelectorAll('#instanceList button')).every(button => {
            const row = button.getBoundingClientRect();
            return Array.from(button.children).every(child => {
              const rect = child.getBoundingClientRect();
              return rect.top >= row.top && rect.bottom <= row.bottom && rect.left >= row.left && rect.right <= row.right;
            });
          })
        """)
        assert results["checks"]["mobile_button_content_inside_rows"]
        assert browser.js("document.documentElement.scrollWidth <= innerWidth")
        browser.screenshot(args.output / "mobile_two_selected.png")
        results["screenshots"].append("mobile_two_selected.png")
        results["runtime_exceptions"] = browser.exceptions
        assert not browser.exceptions, browser.exceptions
        results["passed"] = True
        print(json.dumps(results, indent=2, ensure_ascii=False))
    finally:
        if browser:
            browser.socket.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()
        (args.output / "browser_checks.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
