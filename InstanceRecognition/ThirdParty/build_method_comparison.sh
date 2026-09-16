#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/yoloe26-vs-yolo26-seg-comparison.png"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
FONT="DejaVu-Sans"

YOLOE_BUS="$SCRIPT_DIR/yoloe26_prompt_free/outputs/official_bus_s/annotated.jpg"
YOLO26_SEG_BUS="$SCRIPT_DIR/yolo26_seg/outputs/official_bus_s/annotated.jpg"
WEDETECT_BUS="$SCRIPT_DIR/WeDetect/wedetect_anything/outputs/official_bus.jpg"
YOLO26_DETECT_BUS="$SCRIPT_DIR/yolo26_detect/outputs/official_bus_s/annotated.jpg"
SAM3_BUS="$SCRIPT_DIR/yoloe26_sam3/outputs/sam3/official_bus_s/annotated.jpg"
SAM31_BUS="$SCRIPT_DIR/yoloe26_sam3/outputs/sam3.1/official_bus_s/annotated.jpg"

YOLOE_SCAN="$SCRIPT_DIR/yoloe26_prompt_free/outputs/scannet_scene0000_00_frame0162_s/annotated.jpg"
YOLO26_SEG_SCAN="$SCRIPT_DIR/yolo26_seg/outputs/scannet_scene0000_00_frame0162_s/annotated.jpg"
WEDETECT_SCAN="$SCRIPT_DIR/WeDetect/wedetect_anything/outputs/scannet_scene0000_00_frame0162.jpg"
YOLO26_DETECT_SCAN="$SCRIPT_DIR/yolo26_detect/outputs/scannet_scene0000_00_frame0162_s/annotated.jpg"
SAM3_SCAN="$SCRIPT_DIR/yoloe26_sam3/outputs/sam3/scannet_scene0000_00_frame0162_s/annotated.jpg"
SAM31_SCAN="$SCRIPT_DIR/yoloe26_sam3/outputs/sam3.1/scannet_scene0000_00_frame0162_s/annotated.jpg"

for image in \
  "$YOLOE_BUS" "$YOLO26_SEG_BUS" "$WEDETECT_BUS" "$YOLO26_DETECT_BUS" "$SAM3_BUS" "$SAM31_BUS" \
  "$YOLOE_SCAN" "$YOLO26_SEG_SCAN" "$WEDETECT_SCAN" "$YOLO26_DETECT_SCAN" "$SAM3_SCAN" "$SAM31_SCAN"; do
  test -f "$image"
done

convert "$YOLOE_BUS" -resize '470x627!' "$TMP_DIR/yoloe_bus.png"
convert "$YOLO26_SEG_BUS" -resize '470x627!' "$TMP_DIR/yolo26_seg_bus.png"
convert "$WEDETECT_BUS" -resize '470x627!' "$TMP_DIR/wedetect_bus.png"
convert "$YOLO26_DETECT_BUS" -resize '470x627!' "$TMP_DIR/yolo26_detect_bus.png"
convert "$SAM3_BUS" -resize '470x627!' "$TMP_DIR/sam3_bus.png"
convert "$SAM31_BUS" -resize '470x627!' "$TMP_DIR/sam31_bus.png"
convert "$YOLOE_SCAN" -resize '470x351!' "$TMP_DIR/yoloe_scan.png"
convert "$YOLO26_SEG_SCAN" -resize '470x351!' "$TMP_DIR/yolo26_seg_scan.png"
convert "$WEDETECT_SCAN" -resize '470x351!' "$TMP_DIR/wedetect_scan.png"
convert "$YOLO26_DETECT_SCAN" -resize '470x351!' "$TMP_DIR/yolo26_detect_scan.png"
convert "$SAM3_SCAN" -resize '470x351!' "$TMP_DIR/sam3_scan.png"
convert "$SAM31_SCAN" -resize '470x351!' "$TMP_DIR/sam31_scan.png"

convert -size 2938x1530 xc:'#f7f9fc' \
  -font "$FONT" \
  -fill '#111827' -gravity North -pointsize 45 -annotate +0+22 \
    'Image-only Object Recognition Comparison' \
  -fill '#6b7d99' -pointsize 20 -annotate +0+82 \
    'Same images · small variants · default published vocabularies · YOLOE masks are not passed to SAM' \
  -gravity NorthWest \
  -fill '#111827' -draw \
    'rectangle 24,116 494,220 rectangle 508,116 978,220 rectangle 992,116 1462,220 rectangle 1476,116 1946,220 rectangle 1960,116 2430,220 rectangle 2444,116 2914,220' \
  -gravity NorthWest -fill white -pointsize 25 \
  -annotate +117+132 'YOLOE-26 PF' \
  -annotate +656+132 'YOLO26-seg' \
  -annotate +1135+132 'WeDetect' \
  -annotate +1602+132 'YOLO26 Detect' \
  -annotate +2094+132 'YOLOE → SAM3' \
  -annotate +2556+132 'YOLOE → SAM3.1' \
  -fill '#a9b9ff' -pointsize 15 -annotate +107+180 'Open vocab · native masks' \
  -fill '#8ee3ee' -annotate +631+180 'COCO closed set · masks' \
  -fill '#f3d46a' -annotate +1116+180 'Open vocab · boxes only' \
  -fill '#f5ac72' -annotate +1584+180 'COCO closed set · boxes' \
  -fill '#a7f3d0' -annotate +2059+180 'YOLOE labels + SAM3 masks' \
  -fill '#c4b5fd' -annotate +2513+180 'YOLOE labels + SAM3.1 masks' \
  -gravity NorthWest -fill '#334155' -pointsize 22 -annotate +24+238 \
    'Official comparison image · bus.jpg' \
  "$TMP_DIR/yoloe_bus.png" -geometry +24+274 -composite \
  "$TMP_DIR/yolo26_seg_bus.png" -geometry +508+274 -composite \
  "$TMP_DIR/wedetect_bus.png" -geometry +992+274 -composite \
  "$TMP_DIR/yolo26_detect_bus.png" -geometry +1476+274 -composite \
  "$TMP_DIR/sam3_bus.png" -geometry +1960+274 -composite \
  "$TMP_DIR/sam31_bus.png" -geometry +2444+274 -composite \
  -fill '#eef1ff' -draw 'rectangle 24,901 494,951' \
  -fill '#e9fbfc' -draw 'rectangle 508,901 978,951' \
  -fill '#fff9e8' -draw 'rectangle 992,901 1462,951' \
  -fill '#fff1e7' -draw 'rectangle 1476,901 1946,951' \
  -fill '#eafaf2' -draw 'rectangle 1960,901 2430,951' \
  -fill '#f2edff' -draw 'rectangle 2444,901 2914,951' \
  -gravity NorthWest -fill '#3730a3' -pointsize 18 -annotate +159+916 '11 masks · 7 labels' \
  -fill '#176477' -annotate +651+916 '5 masks · 2 labels' \
  -fill '#97400e' -annotate +1104+916 '17 boxes · 8 labels' \
  -fill '#a23c12' -annotate +1601+916 '5 boxes · 2 labels' \
  -fill '#17795a' -annotate +2095+916 '11 masks · 7 labels' \
  -fill '#6d3fc0' -annotate +2569+916 '11 masks · 7 labels' \
  -gravity NorthWest -fill '#334155' -pointsize 22 -annotate +24+978 \
    'ScanNet · scene0000_00 · frame 0162' \
  "$TMP_DIR/yoloe_scan.png" -geometry +24+1012 -composite \
  "$TMP_DIR/yolo26_seg_scan.png" -geometry +508+1012 -composite \
  "$TMP_DIR/wedetect_scan.png" -geometry +992+1012 -composite \
  "$TMP_DIR/yolo26_detect_scan.png" -geometry +1476+1012 -composite \
  "$TMP_DIR/sam3_scan.png" -geometry +1960+1012 -composite \
  "$TMP_DIR/sam31_scan.png" -geometry +2444+1012 -composite \
  -fill '#eef1ff' -draw 'rectangle 24,1363 494,1413' \
  -fill '#e9fbfc' -draw 'rectangle 508,1363 978,1413' \
  -fill '#fff9e8' -draw 'rectangle 992,1363 1462,1413' \
  -fill '#fff1e7' -draw 'rectangle 1476,1363 1946,1413' \
  -fill '#eafaf2' -draw 'rectangle 1960,1363 2430,1413' \
  -fill '#f2edff' -draw 'rectangle 2444,1363 2914,1413' \
  -gravity NorthWest -fill '#3730a3' -pointsize 18 -annotate +152+1378 '14 masks · 12 labels' \
  -fill '#176477' -annotate +651+1378 '4 masks · 4 labels' \
  -fill '#97400e' -annotate +1110+1378 '6 boxes · 6 labels' \
  -fill '#a23c12' -annotate +1601+1378 '2 boxes · 2 labels' \
  -fill '#17795a' -annotate +2088+1378 '14 masks · 12 labels' \
  -fill '#6d3fc0' -annotate +2562+1378 '14 masks · 12 labels' \
  -gravity North -fill '#6b7d99' -pointsize 16 -annotate +0+1440 \
    'Counts use confidence/match threshold 0.25/0.40 respectively; qualitative examples, not benchmark scores.' \
  -pointsize 16 -annotate +0+1470 \
    'SAM3/SAM3.1 receive only YOLOE boxes (plus center-point fallback for six difficult SAM3.1 prompts), never YOLOE masks.' \
  "$OUT"

identify "$OUT"
