# 交接文件说明（给资产生成模块）

这份文档介绍场景理解模块交给下游的那份数据：它从哪里来、长什么样、读的时候需要注意什么。
不涉及资产生成该怎么做，也不规定下一步。

---

## 一、背景：这份数据在整条链路的什么位置

我们要把一间真实房间（ScanNet 的 `scene0000_00`，一套公寓）变成可编辑的三维场景。
链路分三段：

```
① 场景理解 ──→ ② 资产生成 ──→ ③ 场景组装
   认出房里有   为每个物体做出    把资产放回房间
   哪些东西     可加载的 3D 资产   该在的位置
```

**这份交接文件就是 ① 交给 ② 的接口。**

① 现在是一个逐窗口累积的 LLM 关联流程（harness）：每 5 帧一个窗口，模型判断每张 mask
属于哪个物体，逐渐建出一份物体清单。它还在迭代，结果还不稳定。

所以现在这份交接文件**暂时用 ScanNet 的实例标注产出**——标注是人工的，几何是激光重建的，
可以当作「理想情况下的 ①」来用。等 ① 稳定了，它会产出同样格式的目录，② 的代码不用改。

这也是为什么格式里有些字段现在是 `null`：它们在等 ① 补齐。

---

## 二、目录结构

一个场景一个目录：

```
artifacts/handoff/scene0000_00/gt/
├── manifest.json                 场景级信息：坐标系、单位、关键帧规则、统计
├── objects.json                  物体清单，一个数组，每个元素是一个物体
├── clouds/
│   └── obj_0032.ply              该物体的占据点云
└── keyframes/
    └── obj_0032/
        ├── 0269_rgb.jpg          原图，整帧
        ├── 0269_mask.png         该物体的二值 mask（0/255）
        └── 0269_boxed.jpg        原图 + 黄色框圈住该物体
```

当前这一份：67 个物体，约 230 MB（大头是关键帧图片，点云只有几 MB）。

**先用 `objects.json`，别先读 `manifest.json`。** 前者是清单，后者是说明。

---

## 三、物体记录长什么样

`objects.json` 里一个典型条目（省略了关键帧里的部分字段）：

```json
{
  "object_id": "obj_0032",
  "label": "couch",
  "label_quality": "specific",
  "description": null,
  "structural": false,
  "cloud": "clouds/obj_0032.ply",
  "point_count": 5099,
  "centroid": [3.50, 5.56, 0.58],
  "bounds": {"minimum": [...], "maximum": [...]},
  "frames_visible": 144,
  "key_frames": [ ... 最多 3 条 ... ]
}
```

| 字段 | 说明 |
|---|---|
| `object_id` | 物体标识，`obj_` + 四位数字 |
| `label` | 类别词。来自 ScanNet 的 NYU40 词表，比如 `couch`、`cabinet`、`refrigerator` |
| `label_quality` | 三值，见下方「标签」一节 |
| `description` | 自由描述。**现在全是 null** |
| `structural` | 是否为建筑结构（墙、地板、天花板、窗、门框） |
| `cloud` | 占据点云，相对路径，PLY 格式（xyz float32 + rgb uint8） |
| `point_count` | 点数，与 PLY 实际顶点数一致 |
| `centroid` / `bounds` | 米，场景坐标系 |
| `frames_visible` | 该物体在 500 帧采样里出现过几帧（不是关键帧数） |
| `key_frames` | 该物体的关键帧，1–3 条 |

点云是**实际扫描到的表面**，不是完整的物理实体。椅子背面如果没被拍到，点云里就没有。
`bounds` 因此是「扫描范围的包围盒」，不是物体真实尺寸。

---

## 四、关键帧

```json
{
  "seq": 269,
  "source_frame_id": 3006,
  "rgb":   "keyframes/obj_0032/0269_rgb.jpg",
  "mask":  "keyframes/obj_0032/0269_mask.png",
  "boxed": "keyframes/obj_0032/0269_boxed.jpg",
  "bbox_xyxy": [0, 110, 1296, 968],
  "visible_pixels": 799016,
  "visibility": 0.95,
  "touches_border": true,
  "role": "primary"
}
```

三份文件各有用途：`mask` 是像素级轮廓，`boxed` 是给人核对时的示意图，`rgb` 是原图。
原图和 mask 都是 1296×968，逐像素对齐。

`role` 为 `primary` 的那条是主视图；`secondary`、`third` 是不同视角。

**不是每个物体都有三帧。** 有的物体只能从一个方向看到（贴在墙上的镜子、嵌在墙里的窗），
这时会少给，不会用重复的帧凑数。所以关键帧数量本身就是信息。

---

## 五、四个容易踩的坑

**1. 坐标是 ScanNet 的原始扫描坐标系，不是轴对齐后的。**

单位米，+Z 向上。`manifest.json` 里 `coordinate_frame` 写的是 `scannet_scan`。
如果拿到别的来源的数据，先确认它在哪个坐标系——这份文件里的点和 AMB3R 重建
（harness 用的几何）是对齐的，和套用 `axisAlignment` 之后的 NYU 坐标系**差几米**。

**2. 每帧有两个编号。**

- `seq`：500 帧采样里的序号，0–499
- `source_frame_id`：ScanNet 原始帧号，0–5577，图片文件名用的是这个

两者不是等距映射（`source_frame_id ≈ seq × 5577/499`）。要回溯到原始数据集用后者，
要和 harness 的结果对照用前者。两个都给了。

**3. 标签分三档，不是二档。**

| `label_quality` | 数量 | 含义 |
|---|---|---|
| `specific` | 47 | 有具体类别，比如 `refrigerator` |
| `structural` | 15 | 建筑结构：九段墙、地板、天花板、两扇窗、两个门框 |
| `generic` | 5 | ScanNet 标注者也没认出来，词就是 `object` |

`generic` 那五个拿到的是没有信息量的词。`structural` 那些通常不是资产生成的对象，
但它们在场景组装里有用（墙和地板决定房间形状）。

**4. 点云质量取决于来源。**

GT 的点云是激光扫描 + 人工重建，密度高、表面连续。
将来 harness 产出的是 AMB3R 逐帧反投影，中位误差约 4 cm，有噪声和空洞。
同一份格式，几何质量不同——`manifest.json` 的 `producer` 字段会写明是哪一种。

---

## 六、`manifest.json` 里值得看一眼的

不用全读，但这几个字段会影响怎么理解数据：

| 字段 | 内容 |
|---|---|
| `producer.name` | 谁产出的。现在是 `scannet_ground_truth` |
| `producer.note` | 一句话说明这份几何是什么 |
| `coordinate_frame` / `units` / `up_axis` | 坐标系、单位、朝向 |
| `keyframe_policy` | 关键帧是怎么挑的：可见度下限、视角分离阈值、触边惩罚 |
| `label_quality_counts` | 三档标签各多少个 |
| `keyframe_view_counts` | 有几个物体拿到 3 帧 / 2 帧 / 1 帧 |

---

## 七、怎么看这份数据

有一个可视化页面，不用写代码就能逐物体检查：

```
site/handoff/scene0000_00/gt/index.html
```

左侧是物体列表（按编号排序，可过滤），中间是三个关键帧的三联图（原图 / mask / 框出目标），
右侧是 3D 点云，选中物体以蓝色高亮，背景是场景点云。

页面在检查的正是契约本身：框有没有框住目标、mask 全不全、三个视角是不是真的不同、标签够不够用。

---

## 八、校验

```bash
taskset -c 0 .venv/bin/python -m src.handoff.check artifacts/handoff/scene0000_00/gt
```

会检查：字段完整、`object_id` 唯一、点云文件存在且点数与声明一致、关键帧的三张图都在、
mask 白色像素数与 `visible_pixels` 相符、包围盒在图像范围内。

通过时输出 `OK`，否则逐条列出问题。

---

## 九、现在还缺什么

诚实地说清楚，免得踩坑：

- **`description` 全是 null。** GT 只有类别词，没有「什么颜色、什么材质、上面有没有东西」这类描述。
- **`mesh` 全是 null。** 格式里留了这个字段，因为 GT 其实有三角网格；但 harness 产出不了网格，
  所以契约里它是可选的。现在没填。
- **生成这份数据的代码**在 `src/handoff/`：
  - `schema.py` — 字段定义和校验规则，**改契约时先改这里**
  - `from_gt.py` — GT 的导出器
  - `check.py` — 校验器
  - `pagegen/build_handoff.py` — 可视化页面
