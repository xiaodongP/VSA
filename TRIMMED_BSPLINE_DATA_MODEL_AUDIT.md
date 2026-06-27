# Trimmed B-Spline 数据模型审计报告

> 范围：仅代码分析与报告，未修改任何代码。
> 所有结论均基于实际字段与调用链（含 `file:line`），而非类名推测。
> 审计日期：2026-06-27。

---

## 0. 执行摘要（先读这一段）

当前实现存在**两条并行的 trimmed B-spline 路径**，必须区分：

| 路径 | 入口 | 核心类型 | 是否持久化参数化资产 |
| --- | --- | --- | --- |
| 旧版单环路径 | `Fit selected region trimmed B-spline` ([main.cpp:4174](main.cpp#L4174)) | `TrimmedBSplineSurfacePatch`（[trimmed_bspline_surface.h:22](trimmed_bspline_surface.h#L22)） | 否（仅采样成 OBJ） |
| 主管线（论文复现） | `Fast standard trimmed pipeline` / `Run standard trimmed B-spline pipeline` ([main.cpp:4181](main.cpp#L4181)/[4184](main.cpp#L4184))、`Fast/Run all trimmed regions` ([main.cpp:4187](main.cpp#L4187)/[4190](main.cpp#L4190)) | `ReusableTrimmedBSplineSurface`（[reusable_trimmed_bspline_surface.h:21](reusable_trimmed_bspline_surface.h#L21)） | **写**了一份参数化 JSON，但**只写不读** |

**核心结论（一句话）**：主管线在运行期间确实在内存中构造了一个**完整的参数化 trimmed B-spline 资产对象**（`BSplineSurface3D` + UV trim loops），并把它**序列化**为 `trimmed_bspline_asset.json`（含 degree/knots/control_grid/UV trim loop，数据上属于真正的参数化资产）；但是——

1. 这个 JSON **是只写的**：代码库中没有任何加载/反序列化函数，没有任何地方从 JSON 重建 `BSplineSurface3D` 或 `ReusableTrimmedBSplineSurface`（见 §5、§7）。
2. 内存中的参数化对象在 `run_single_region_trimmed_bspline_pipeline` 返回后即被丢弃；返回值 `TrimmedBSplinePipelineResult` 只携带 `metrics` 与 `baseline_results`，而 [main.cpp](main.cpp) 只读取 `metrics` 后就让 `result` 离开作用域（见 §5）。
3. viewer / 批处理真正**重新加载、合并、显示、依赖**的对象，是采样后的**三角网格 OBJ**（`bspline_trimmed_surface.obj`）。
4. 拟合出来的 `fitted_curve`（2D B-spline trim 曲线）**从不参与裁剪/inside-test**——所有裁剪只认离散的 `uv_polyline`（见 §3、§6）。
5. `SharedSplineAssembly`（共享边界 / region adjacency）**没有接入主管线**，只存在于 `tests/` 与 `global_bspline_optimization.cpp`（见 §4）。

因此：**作为数据结构**它最接近 **B（base surface + 离散 UV trim loop）**；**作为系统真正长期复用的对象**它是 **C（临时参数化结果 + 持久 triangle mesh）**。强制单选的最终归类见 §10（选 **C**，并附 B 级 JSON 的关键限定）。

---

## 一、核心类型清单

### 1.1 `BSplineSurface3D` —— base B-spline 曲面（唯一权威的参数化曲面类型）

```text
类型名        BSplineSurface3D
定义          bspline.h:53-75（class）；实现 bspline.cpp:369(evaluate)/389(derivative)
字段
  int degree_u, degree_v                       U/V 次数
  std::vector<double> knots_u, knots_v         U/V 节点向量（open-uniform）
  std::vector<std::vector<Eigen::Vector3d>> control_grid   控制网（外层=U，内层=V）
字段语义      非有理张量积 B-spline 曲面。无 weights 字段（非 NURBS）。
              参数域由节点向量隐含给出（domain = [knots[degree], knots[end-degree-1]]）。
              控制网尺寸 = control_grid.size() × control_grid[0].size()，无独立字段存储。
所有权/生命周期 按值持有；可作为 ReusableTrimmedBSplineSurface.surface 的成员长期存活，
              但在主管线中仅在 pipeline 函数栈上临时存在（见 §5）。
创建者        fit_tensor_product_cubic_bspline_surface（tensor_product_bspline_fitter.cpp）
              → BSplineSurface3D 构造（见 main.cpp:983/1065、pipeline 中的 fit_d.surface）
消费者        evaluate/derivative；sample_bspline_surface；sample_reusable_trimmed_bspline_surface；
              export_bspline_surface_mesh_obj/control_net_obj；make_reusable_trimmed_bspline_surface
序列化        ✅ 序列化为 JSON（reusable_trimmed_bspline_surface.cpp:801 write_bspline_surface_json）
              ❌ 无反序列化（无 from_json / load）
```

`evaluate`/`derivative` 实现确认（[bspline.cpp:369](bspline.cpp#L369)、[bspline.cpp:389](bspline.cpp#L389)）：基于 `bspline_basis` / `bspline_basis_derivative` 的标准张量积求值。

### 1.2 `BSplineCurve3D` —— 3D B-spline 曲线

```text
定义          bspline.h:26-41
字段          int degree; vector<double> knots; vector<Vector3d> control_points
能力          evaluate(t) / derivative(t, order) / sample(count)
创建/消费     用于 guiding frame、shared boundary（SharedSplineBoundary.curve），
              以及 InitialBSplineSurfacePatch.boundary_curves
序列化        仅在 SharedSplineAssembly debug 路径里间接导出；asset JSON 不含 3D 曲线
```

### 1.3 `TrimCurve2D` —— 2D trim 曲线（B-spline 形式）

```text
定义          trimmed_bspline_surface.h:11-20
字段
  int degree
  vector<double> knots
  vector<Vector2d> control_points     2D 控制点
  vector<Vector2d> polyline           对应的离散采样折线
  bool valid; string reason
语义          由折线拟合出的 2D B-spline 曲线，**同时**保留原始折线。
关键事实      在资产/裁剪流程中**从未被用于求值或裁剪**（见 §3）。
创建者        fit_trim_curve_2d_from_polyline（trimmed_bspline_surface.cpp）
              在 make_reusable_trimmed_bspline_surface 中仅当 uv_polyline.size()>=4 时拟合
              （reusable_trimmed_bspline_surface.cpp:878-883）
消费者        仅被写进 JSON（export_reusable_trimmed_bspline_surface_json:948-962）
序列化        ✅ 写入 fitted_trim_curve 字段；❌ 无加载
```

### 1.4 `TrimmedBSplineSurfacePatch` —— 旧版单环 trimmed patch

```text
定义          trimmed_bspline_surface.h:22-30
字段
  BSplineSurface3D surface
  vector<Vector2d> outer_trim_polyline        外环离散折线
  TrimCurve2D      outer_trim_curve           外环拟合曲线
  bool valid; string reason
语义          单 outer loop，无 inner loop / 无 holes。
入口          Fit selected region trimmed B-spline（fit_and_show_trimmed_bspline_surface_for_region）
消费者        sample_trimmed_bspline_surface（按折线裁剪）
序列化        仅 debug OBJ；无 JSON 资产
```

### 1.5 `TrimLoop2D` —— pipeline 内部的 UV trim loop（过渡类型）

```text
定义          rectangular_domain_extension.h:12
（在管线中观测到的字段：is_perimeter, uv_polyline, vertex_ids）
语义          参数化阶段产出的 UV trim loop；经过 normalized_trim_loops 归一化到 [0,1]²
              （trimmed_bspline_pipeline.cpp:162-176），随后注入 make_reusable_trimmed_bspline_surface。
序列化        ❌ 不直接序列化；其内容被拷贝进 ReusableTrimLoop2D
```

### 1.6 `ReusableTrimLoop2D` —— 资产里的 trim loop（最终形态）

```text
定义          reusable_trimmed_bspline_surface.h:13-19
字段
  bool is_perimeter                          外环(true)/内环(false，代表孔)
  vector<int> source_vertex_ids              回指原始网格顶点 id（权威 3D 边界来源）
  vector<Vector2d> uv_polyline               ★权威 trim 表示（离散 UV 折线，闭合）
  vector<Vector3d> spatial_polyline          与 uv_polyline 一一对应的 3D 边界点
  TrimCurve2D fitted_curve                   装饰性拟合曲线（不参与裁剪）
语义          二维 UV 空间折线 + 同源 3D 点。所有裁剪/inside-test/CDT 约束均用 uv_polyline。
```

### 1.7 `ReusableTrimmedBSplineSurface` —— “资产”对象（序列化目标）

```text
定义          reusable_trimmed_bspline_surface.h:21-30
字段
  BSplineSurface3D surface                   base 曲面
  vector<ReusableTrimLoop2D> trim_loops      trim 环集合（structurally 支持多环/孔）
  Vector2d source_uv_min, source_uv_max      原始参数域（资产域固定 [0,1]²）
  int source_region_id                       来源区域 id
  string generator                           生成方式描述
  bool valid; string reason
创建者        make_reusable_trimmed_bspline_surface（reusable_trimmed_bspline_surface.cpp:838）
              在主管线 trimmed_bspline_pipeline.cpp:2679 构造
消费者        export_reusable_trimmed_bspline_surface_debug（写 JSON + 采样 OBJ）
              sample_reusable_trimmed_bspline_surface / sample_abc_boundary_controlled_trimmed_surface
序列化        ✅ trimmed_bspline_asset.json（write-only）；❌ 无 loader
生命周期      ★仅存在于 pipeline 栈帧；pipeline 返回后丢弃，main.cpp 不持有任何全局资产对象
```

### 1.8 `BoundaryRibbonSurface` —— ABC 预览用的边界 ribbon 曲面

```text
定义          reusable_trimmed_bspline_surface.h:32-40
字段          loop_id; BSplineSurface3D surface; boundary_uv; boundary_xyz; ribbon_width_uv; valid; reason
语义          为在边界处贴合权威 3D 边界而构造的 G0 blend ribbon（debug/preview）。
              属于 §0 表中“preview/debug derivative”，非 canonical asset。
```

### 1.9 `TrimmedBSplinePipelineResult` —— 主管线返回值

```text
定义          trimmed_bspline_pipeline.h:66-71
字段
  bool valid; string reason
  TrimmedBSplinePipelineMetrics metrics                  指标（被 main.cpp 使用）
  vector<TensorProductBSplineFitResult> baseline_results ★每个含一个 BSplineSurface3D surface
关键事实      baseline_results 携带了参数化曲面，但 main.cpp 只用 metrics，result 离开作用域即销毁。
```

### 1.10 `SharedSplineAssembly` / `SharedSplineBoundary` —— 共享边界（**未接入主管线**）

```text
定义          shared_spline_boundary.h:12-58
  SharedSplineBoundary { id, region_a, region_b, BSplineCurve3D curve, control_point_ids }
  GlobalControlPointPool / BSplinePatchTopology / SharedSplinePatch（含 4 条 shared_boundary_ids）
  SharedSplineAssembly { pool, patches, shared_boundaries, valid, reason }
能力          build_two_patch_shared_boundary_assembly / sample_shared_spline_assembly /
              check_shared_spline_adjacency（可检查 watertight、shared edge）
关键事实      ⚠️ 仅被 tests/test_shared_spline_boundary.cpp、tests/test_global_bspline_optimization.cpp
              以及 global_bspline_optimization.cpp 自身调用。
              trimmed_bspline_pipeline.cpp 与 main.cpp 均不引用。→ 主管线无 region adjacency。
```

---

## 二、base B-spline surface 的表示

**2.1 持久化字段清单（以 `BSplineSurface3D` 为准）**

| 项目 | 是否保存 | 位置 |
| --- | --- | --- |
| degree U / V | ✅ | `degree_u`, `degree_v`（[bspline.h:55-56](bspline.h#L55)）；JSON `degree_u/v`（[reusable_trimmed_bspline_surface.cpp:806-807](reusable_trimmed_bspline_surface.cpp#L806)） |
| knot vector U / V | ✅ | `knots_u`, `knots_v`（[bspline.h:57-58](bspline.h#L57)）；JSON `knots_u/v` |
| control points | ✅ | `control_grid`（[bspline.h:59](bspline.h#L59)）；JSON `control_grid` |
| **rational weights** | ❌ | 类型中无 weights 字段——**非有理 B-spline，非 NURBS** |
| parameter domain | ⚠️ 隐含 | 曲面本身不存域；域由节点向量隐含（`[knots[degree], knots[size-1-degree]]`）。资产层另存 `source_uv_min/max` + 固定资产域 `[0,1]²`（[reusable_trimmed_bspline_surface.cpp:911-918](reusable_trimmed_bspline_surface.cpp#L911)） |
| control-net 维度 | ⚠️ 隐含 | 无独立字段；= `control_grid.size()` × `control_grid[0].size()`；JSON 额外冗余写出 `control_count_u/v`（:820-823） |

**2.2 独立求值能力（pipeline 结束后能否独立调用）**

- `evaluate(u, v)` ✅ 已实现（[bspline.cpp:369](bspline.cpp#L369)）。
- `derivative(u, v, order_u, order_v)` ✅ 已实现（[bspline.cpp:389](bspline.cpp#L389)）。
  - `derivativeU(u,v)` ≡ `derivative(u,v,1,0)`；`derivativeV(u,v)` ≡ `derivative(u,v,0,1)`（无独立命名方法，但等价可得）。
- `normal(u,v)` ⚠️ 无独立方法，但可由 `derivative(u,v,1,0) × derivative(u,v,0,1)` 直接得到（代码中多处用采样后叉积计算法向，例如 `count_sampled_surface_fold_edges`）。

**关键限定**：上述能力**只对“仍持有 `BSplineSurface3D` 对象”的调用方成立**。在主管线中，该对象是 `fit_d.surface`（[trimmed_bspline_pipeline.cpp:2416](trimmed_bspline_pipeline.cpp#L2416) 等处的局部），它在 pipeline 函数内部被采样成网格、塞进 `ReusableTrimmedBSplineSurface` 写出 JSON，然后随函数返回而销毁。[main.cpp](main.cpp) 中**没有任何全局 `BSplineSurface3D` / `ReusableTrimmedBSplineSurface`**（grep 确认：`BSplineSurface3D` 仅作为 934/964/983/1065/1227/1612 等处的局部出现）。因此“pipeline 结束后能否独立求值”的答案是：

- **同一次运行内、持有对象时**：能（`sample_reusable_trimmed_bspline_surface` 就是基于 asset 重新 tessellate 的入口，[reusable_trimmed_bspline_surface.cpp:1004](reusable_trimmed_bspline_surface.cpp#L1004)）。
- **跨运行、从磁盘恢复后**：**不能**——没有 JSON loader，无法重建对象（见 §5、§7）。

**2.3 是临时存在还是采样后丢弃？**
对**主管线**而言：base surface 在 pipeline 栈上临时存在，最终被采样成 `bspline_trimmed_surface*.obj` 多份网格；其参数化形式**额外**写入 `trimmed_bspline_asset.json`，但该 JSON 没有反向通道。所以“采样成网格后丢弃”描述的是**内存语义**；磁盘上参数化数据仍在，只是无人读回。

---

## 三、trim loop 的实际表示

**实际保存为：二维 UV 空间的离散闭合折线（piecewise-linear polygon）。**

权威字段是 `ReusableTrimLoop2D::uv_polyline`（[reusable_trimmed_bspline_surface.h:16](reusable_trimmed_bspline_surface.h#L16)）。证据——所有裁剪相关逻辑都只读 `uv_polyline`：

- inside-test：`point_in_polygon_or_on_boundary` / `point_inside_asset_trim_region`（[reusable_trimmed_bspline_surface.cpp:82-122](reusable_trimmed_bspline_surface.cpp#L82)）——射线法对 `uv_polyline`。
- CDT 约束边：`build_cdt_trimmed_uv_mesh` 把 `uv_polyline` 的相邻顶点作为 constraint edges（:643-661）。
- 距离场：`distance_to_polyline_segments` / `distance_to_trim_loops`（:124-155）。
- 边界 ribbon、ABC 预览投影：全部基于 `uv_polyline` + `spatial_polyline`。

**`fitted_curve`（TrimCurve2D，真正的 2D B-spline 曲线）**：仅在 `make_reusable_trimmed_bspline_surface` 中拟合（:878-883），随后**只被写进 JSON**（:948-962），**从不参与任何求值/裁剪/CDT**。也就是说“trim 是 B-spline curve”只是 JSON 里的一份记录，运行期裁剪语义完全是分段线性的。

trim loop 的属性：

```text
位于二维 UV 空间还是三维       二维 UV（[0,1]² 资产域）。三维仅在 spatial_polyline 里作为对应点存在
是否保存 outer/inner 语义      ✅ is_perimeter（true=外环，false=内环/孔）
是否保存 loop orientation      ❌ 不显式存储朝向；inside-test 用射线法，不依赖方向
是否严格闭合                  ⚠️ 折线按闭合环处理（add_uv_point 去重 + (i+1)%n），
                              cleaned_loop 会去掉首尾重复点（reusable_trimmed_bspline_surface.cpp:223-235）
是否允许多个 loop / holes      结构上允许（vector<ReusableTrimLoop2D>，inside-test 会减去非 perimeter 环，
                              reusable_trimmed_bspline_surface.cpp:117-120）。但管线侧 rotation-angle 阶段
                              明确拒绝多连通区域（TRIMMED_BSPLINE_REPRODUCTION.md:52），实际主管线只产出单外环
是否能在 pipeline 结束后复用    ⚠️ 同 §2：对象内存里已被丢弃；JSON 里有数据但无 loader
```

如果是 B-spline curve（仅 `fitted_curve`）：degree/knots/control_points 均写入 JSON，但它**不是裁剪定义**，仅作记录。折线 `uv_polyline` 来自原始 region 边界顶点的 UV 映射（经 KKT/ARAP 参数化 → rectangular extension → normalized_trim_loops，来源链：`domain.authoritative_trim_loops`，[trimmed_bspline_pipeline.cpp:2676-2688](trimmed_bspline_pipeline.cpp#L2676)）。

---

## 四、三维边界如何定义

当前实现属于：

> **C. 仅为离散三维边界顶点**（`spatial_polyline = {x0, x1, …, xn}`），且这些点直接取自原始网格顶点。
> 同时**隐含可派生** B 形式：`C(t)=S(gamma(t))`——对 `uv_polyline` 求 `surface.evaluate` 即得到一条 3D 曲线（`write_surface_trim_loops_obj` 正是这么做，[reusable_trimmed_bspline_surface.cpp:388](reusable_trimmed_bspline_surface.cpp#L388)）。

但**不存在独立、权威、共享的 3D 曲线对象**：

- 资产里没有 `BSplineCurve3D` 类型的 3D 边界曲线字段（grep 确认 `ReusableTrimmedBSplineSurface` 不含 3D 曲线）。
- 权威 3D 边界 = `spatial_polyline`，源自 `source_vertex_ids → original_V.row(vid)`（[reusable_trimmed_bspline_surface.cpp:860-877](reusable_trimmed_bspline_surface.cpp#L860)）。
- `BSplineCurve3D` 形式的 3D 边界**只**出现在 `SharedSplineBoundary.curve`（[shared_spline_boundary.h:16](shared_spline_boundary.h#L16)），而该类型未接入主管线。

**相邻 region / 共享边界：**

- ❌ 主管线**没有 region adjacency**：每个 region 独立运行 `run_single_region_trimmed_bspline_pipeline`，各自从自己的 `BoundarySegmentationResult.loops`（[trimmed_region_input.h:62](trimmed_region_input.h#L62)）提取边界。
- ❌ 不存在 shared-edge ID；不存在让两个 region 引用同一条 3D 边界的机制（在主管线范围内）。
- `BoundarySegment` 有 `adjacent_region_id` 字段（[trimmed_region_input.h:38](trimmed_region_input.h#L38)），说明**输入侧感知到了邻接 region**，但该信息没有上提到资产层、没有用于跨 region 共享边界。
- 唯一实现了“两 patch 共享边界 + watertight 检查”的是 `SharedSplineAssembly`（[shared_spline_boundary.cpp:204](shared_spline_boundary.cpp#L204) `build_two_patch_shared_boundary_assembly`、`check_shared_spline_adjacency`），但它**只在 tests 和 global_bspline_optimization.cpp 中使用**，未进入 trimmed_bspline_pipeline / main.cpp。

> 因此“让两个 region 共享同一条三维边界”这一能力，在当前主管线中**不存在**；它仅作为独立的研究/测试模块存在。

---

## 五、最终输出对象归类

把主管线的最终输出对照题目选项：

| 选项 | 是否符合 | 说明 |
| --- | --- | --- |
| 1. 完整参数化 trimmed asset（base+UV loops+topology） | ❌ | 缺 topology/adjacency；trim 是折线非曲线语义 |
| 2. 简化 asset（base surface + UV polyline） | ✅ **数据层符合** | JSON 内容正是如此 |
| 3. 仅保存 base，trim 丢失 | ❌ | trim loops 完整写入 JSON |
| 4. 仅保存裁剪后三角网格 | ⚠️ | 这是**唯一被反向加载/显示**的对象，但不是唯一写出的对象 |
| 5. base 与 mesh 都存但无持久关联 | ⚠️ 部分 | 两者都存；JSON 与 OBJ 其实有派生关系，但 OBJ 不引用 JSON，运行期无关联 |
| 6. viewer debug object，离线不可复用 | ⚠️ | viewer 确实只 reload OBJ；但 JSON 客观存在于磁盘 |

**运行期实际依赖链**（这才是“长期保存并复用的对象”真相）：

1. pipeline 在内存构造 `ReusableTrimmedBSplineSurface` → 写 `trimmed_bspline_asset.json` + 多份采样 OBJ。
2. pipeline 返回 `TrimmedBSplinePipelineResult`（含 `baseline_results[].surface`）。
3. [main.cpp:1403-1436](main.cpp#L1403) `igl::readOBJ(output_dir + "/bspline_trimmed_surface.obj", splineSurfaceV, splineSurfaceF)` —— **从磁盘把三角网格读回**，存入全局 `splineSurfaceV/F`（[main.cpp:121](main.cpp#L121)）。
4. `result`（含参数化曲面）离开作用域销毁；`splineSurfaceV/F`（纯三角网格）留存在 viewer。
5. “all regions”批处理（[main.cpp:1807-1810](main.cpp#L1807)）同样只 `readOBJ` 每个区域的 `bspline_trimmed_surface.obj`，合并成一张大网格（`merged_vertices/faces`）。

→ **系统真正长期保存、重新加载、合并、显示的“trimmed B-spline surface”对象，是一张裁剪后的三角网格。** 参数化 JSON 是“写到磁盘但无人读回”的副产物。

---

## 六、Tessellation 数据流

主管线在 pipeline 末尾（[trimmed_bspline_pipeline.cpp:2702-2864](trimmed_bspline_pipeline.cpp#L2702)）会产出**多套** tessellation，最终挑一份作为 `bspline_trimmed_surface.obj`。逐步追踪（以资产派生的 `sample_reusable_trimmed_bspline_surface` 为基准，[reusable_trimmed_bspline_surface.cpp:1004-1075](reusable_trimmed_bspline_surface.cpp#L1004)）：

```text
步骤                         输入类型              输出类型             修改原对象?  保留对应关系?  丢失参数信息?
1 UV 网格生成                asset.trim_loops      nu×nv 的 full_uv     否           —             否
                             ([0,1]² 规则栅格)     (MatrixXd N×2)
2 inside/outside trim test   uv_polyline           保留的网格点 boolean 否           —             否
                             (射线法)
3 trim boundary 相交         —（规则栅格法用质     —                    否           —             —
                             心 inside-test 近似；
                             CDT 法才真正处理相交）
4 2D clipping                uv_polyline           kept triangles       否           —             否
5 CDT / triangulation        仅 CDT 路径用         Vector3i faces       否           —             否
                             CGAL CDT（约束边=
                             uv_polyline 相邻点）
6 UV vertices                compact_uv            MatrixXd UV(N×2)     否           ✅             否
7 3D evaluation              surface.evaluate      V(N×3)               否           ✅ UV 仍在     否
8 final V/F mesh             V,UV,F                写 OBJ               否           见下           —
9 viewer rendering           OBJ → readOBJ         splineSurfaceV/F     否           ❌ UV 被丢弃    ★是（见下）
10 OBJ export                V,F                   bspline_trimmed_surface.obj 否     ❌ 不写 UV     ★是
```

**最终三角网格顶点是否仍保存对应 `(u,v)`？**

- 在 **pipeline 内部函数的输出**里：✅ 保留。`sample_reusable_trimmed_bspline_surface(..., MatrixXd* UV)`、`sample_trimmed_surface_with_authoritative_boundary(..., UV*)`、`sample_bspline_on_region_topology_...(..., UV*)` 都把每个顶点的 UV 写回（[reusable_trimmed_bspline_surface.cpp:1062-1073](reusable_trimmed_bspline_surface.cpp#L1062)）。
- 在**最终写出的默认 viewer 网格** `bspline_trimmed_surface.obj` 里：❌ **丢失**。`write_mesh_obj`（[reusable_trimmed_bspline_surface.cpp:299](reusable_trimmed_bspline_surface.cpp#L299) / [trimmed_bspline_pipeline.cpp:80](trimmed_bspline_pipeline.cpp#L80)）只写 `v x y z` 与 `f`，**不写 `vt`**。UV 单独写到 `*_uv.obj`（如 `bspline_trimmed_surface_asset_uv.obj`），但 viewer 默认读的是无 UV 的那份。
- 在 **viewer 全局状态** `splineSurfaceV/F` 里：❌ 没有 UV 通道（`MatrixXd splineSurfaceV`，[main.cpp:121](main.cpp#L121)）。

另两条 tessellation 路径（均产 UV，但同样以无 UV 的 OBJ 作为最终 viewer mesh）：

- **CDT + 权威边界锁定**：`sample_trimmed_surface_with_authoritative_boundary`（[trimmed_bspline_pipeline.cpp:371](trimmed_bspline_pipeline.cpp#L371)）。把 `uv_polyline` 顶点锁到 `original_V` 的权威 3D 坐标，CGAL CDT 约束剖分，再做可选的 harmonic boundary residual 校正（:538-601）。
- **region topology 复用**：`sample_bspline_on_region_topology_with_authoritative_boundary`（[trimmed_bspline_pipeline.cpp:611](trimmed_bspline_pipeline.cpp#L611)）。直接用原 region 的 `local_faces` 拓扑，把每个顶点 UV 代入 `surface.evaluate`，可选 snap/harmonic 校正。

pipeline 的 `final_output_mode`（:2839-2849）会从 region-topology / CDT-locked / regular-grid 三者中择优，再经 `remove_degenerate_faces` + `orient_mesh_faces_consistently` 清洗，写入 `bspline_trimmed_surface.obj`（:2864）。

---

## 七、可复用能力判定

> “可复用”严格按“pipeline 结束、跨运行后仍能做”来判断（因为单次运行内持有对象时多数都能做）。

| 能力 | 判定 | 原因 / 代码位置 / 缺失数据 |
| --- | --- | --- |
| 修改控制点后重新 tessellate | **不支持（跨运行）** | 无 JSON loader 重建 `BSplineSurface3D`；`baseline_results` 已销毁。单次运行内若持有 asset 则可（`sample_reusable_trimmed_bspline_surface`）。缺失：反序列化器、全局资产句柄 |
| 修改 trim loop 后重新 tessellate | **不支持（跨运行）** | 同上；且 trim 是 `uv_polyline`，编辑后需重跑 inside-test/CDT。缺失：loader、trim 编辑 API |
| 提高采样分辨率后重新 tessellate | **不支持（跨运行）** | 同上。单次运行内 `sample_u/v` 是参数，可重采样。缺失：loader |
| 查询任意 UV 是否在 trim region 内 | **不支持（跨运行）** | `point_inside_asset_trim_region`（:103）逻辑存在，但需要 asset 对象。缺失：loader。单次运行内✅ |
| 沿 trim boundary 连续采样 | **部分支持** | 可沿 `uv_polyline` 离散采样；若想要真正连续需用 `fitted_curve`，但 fitted_curve 不参与任何求值。缺失：可求值的 2D trim curve 通道 |
| 计算边界曲率 | **不支持（跨运行）** | base `derivative` 可算曲率，但需对象；3D 边界是离散点。缺失：loader、3D 曲线对象 |
| 导出为 CAD/B-rep | **不支持** | 无 STEP/IGES/B-rep 写出；JSON 是项目私有格式（`ReusableTrimmedBSplineSurface.v1`）。缺失：B-rep 拓扑、CAD 导出器 |
| 让两个 region 共享同一条 3D 边界 | **不支持（主管线）** | `SharedSplineAssembly` 能力存在但未接入；主管线每 region 独立、各自一份 `spatial_polyline`。缺失：把 SharedSplineAssembly 接入 pipeline、邻接图、shared-edge id |

---

## 八、最小数据结构图（忠实反映当前代码）

```
TrimmedBSplinePipelineResult  (trimmed_bspline_pipeline.h:66)   ← pipeline 返回值；main.cpp 用完即弃
├── metrics: TrimmedBSplinePipelineMetrics
└── baseline_results: [TensorProductBSplineFitResult]           ← ★含 BSplineSurface3D，但无人长期持有
    └── surface: BSplineSurface3D
        ├── degree_u / degree_v
        ├── knots_u / knots_v
        └── control_grid[][]   (非有理，无 weights)

ReusableTrimmedBSplineSurface  (reusable_trimmed_bspline_surface.h:21)   ← 仅在 pipeline 栈帧存活
├── surface: BSplineSurface3D                                 → 写入 trimmed_bspline_asset.json (只写)
├── trim_loops: [ReusableTrimLoop2D]
│   ├── is_perimeter: bool
│   ├── source_vertex_ids: [int]        (回指 original_V)
│   ├── uv_polyline: [Vector2d]         ★权威 trim（离散 UV 折线，闭合）
│   ├── spatial_polyline: [Vector3d]    (权威 3D 边界离散点；option C)
│   └── fitted_curve: TrimCurve2D       (装饰性 2D B-spline；不参与裁剪，仅入 JSON)
├── source_uv_min / source_uv_max
└── source_region_id

        ↓ sample_reusable_trimmed_bspline_surface / CDT / region-topology（三选一，均产 UV）

最终输出（磁盘）：
├── trimmed_bspline_asset.json        参数化资产（只写，无 loader）
├── bspline_trimmed_surface.obj       ★最终 viewer mesh（无 UV；被 readOBJ 重新加载）
├── bspline_trimmed_surface_asset.obj 资产派生采样网格（无 UV）
├── bspline_full_surface.obj          未裁剪整面采样
├── bspline_trimmed_surface_*_uv.obj  UV 版本（与上面分离）
├── uv_trim_loops.obj / *_trim_loops.obj
└── abc_*_preview.obj / ribbon_*.obj  (preview/debug derivatives)

viewer 全局状态（main.cpp:121）：splineSurfaceV/F : MatrixXd/MatrixXi  ← 纯三角网格，无 UV、无参数化

SharedSplineAssembly (shared_spline_boundary.h:43)   ← 独立模块，未接入主管线
├── pool: GlobalControlPointPool
├── patches: [SharedSplinePatch] (4×shared_boundary_ids)
└── shared_boundaries: [SharedSplineBoundary {region_a, region_b, BSplineCurve3D curve}]
```

---

## 九、与理想 trimmed surface asset 对比

| 能力 | 当前实现 | 理想结构 |
| --- | --- | --- |
| base surface 可持久求值 | 🟡 数据完整写入 JSON，但无 loader，跨运行不可求值 | base surface 可随时 `evaluate/derivative` |
| trim loop 可编辑 | 🔴 折线静态、无编辑 API、无 loader | 可编辑的 UV trim loops（曲线/折线均可） |
| 3D boundary 可访问 | 🟡 仅离散 `spatial_polyline`；无 3D 曲线对象 | 权威 3D 边界曲线，可求值 |
| UV / 3D 对应 | 🟡 pipeline 内部函数保留；最终 viewer OBJ 丢失 | 每个顶点持久保留 `(u,v)` |
| 多 loop / holes | 🟡 数据结构支持，rotation-angle 阶段拒绝多连通 | 原生支持 outer/inner loops |
| region adjacency | 🔴 主管线无；`BoundarySegment.adjacent_region_id` 未上提 | 完整邻接图 |
| 共享边界 | 🔴 主管线无；仅 SharedSplineAssembly（test-only） | 相邻 patch 引用同一边界曲线 |
| 重新 tessellation | 🔴 跨运行不可；需重跑整条 pipeline | 改参数即可重 tessellate |
| B-rep 导出 | 🔴 无；JSON 为私有格式 | STEP/IGES/B-rep |

> 图例：🟢=支持　🟡=部分/有条件　🔴=不支持

---

## 十、结论

**当前实现是：**

> **C. 临时参数化结果 + 持久 triangle mesh**

补充限定（必须同时声明，否则误导）：

- 作为**磁盘上的数据结构**，`trimmed_bspline_asset.json` 同时满足 **B（base surface + 离散 UV trim loop）**：它确实保存了完整的 `BSplineSurface3D`（degree/knots/control_grid）与 UV trim loops（折线 + 装饰性 B-spline 曲线）。
- 但由于 **JSON 只写不读、内存资产在 pipeline 返回后丢弃、viewer 与批处理只 reload 三角网格**，实际被系统长期复用的对象是**裁剪后的三角网格**。因此整体归类落到 **C**。
- 它**不是 A**（缺 loader、缺 topology/adjacency、trim 不是曲线语义、非有理）；**不是 D**（参数化 JSON 客观存在，并非只生成网格）。

### 为支持“全局共享 B-spline boundary network”最少需要新增/重构的字段与接口（本次不实施）

1. **资产反序列化（最高优先级，解除 C 的根因）**
   - 新增 `ReusableTrimmedBSplineSurface load_reusable_trimmed_bspline_surface_json(const string&)`，重建 `BSplineSurface3D` + `ReusableTrimLoop2D`。
   - 让 `main.cpp` 持有 `std::vector<ReusableTrimmedBSplineSurface> g_region_assets`（按 region_id 索引）作为全局权威资产，viewer mesh 改为由资产 tessellate 派生。

2. **把 SharedSplineAssembly 接入主管线**
   - 在“all regions”批处理末尾，用每个 region 的 `BSplineSurface3D` + `BoundarySegment.adjacent_region_id` 构造 `SharedSplineAssembly`（替换当前仅 test 用的 `build_two_patch_shared_boundary_assembly`，扩展到 N-patch）。
   - 新增全局邻接表：`map<pair<region_a,region_b>, shared_boundary_id>`。

3. **共享边界数据模型（资产层新字段）**
   - 给 `ReusableTrimmedBSplineSurface` 增加 `vector<SharedBoundaryRef>`，其中 `SharedBoundaryRef { int boundary_id; int neighbor_region_id; int side; BSplineCurve3D curve; vector<int> control_point_ids_into_pool; }`。
   - 引入 `GlobalControlPointPool` 让相邻 patch 的边界控制点引用同一份 3D 点（实现“两个 region 引用同一条 3D 边界”）。

4. **trim 曲线语义升级（可选但推荐）**
   - 让 `fitted_curve`（或新增 `authoritative_trim_curve`）真正参与裁剪求值，提供沿边界连续采样与曲率；或显式声明 trim = polyline 并去掉误导性的 `fitted_curve`。

5. **持久 UV 通道**
   - 最终 `bspline_trimmed_surface.obj` 改写 `vt`（或在 `.obj` 旁固定写 `.mtd`/JSON），让 viewer mesh 顶点保留 `(u,v)`，恢复 UV/3D 对应。

6. **（可选）B-rep / CAD 导出**
   - 基于 base surface + UV trim loops + 邻接拓扑，新增 STEP/IGES writer，把资产升级到真正的 **A**。

> 以上 1–3 是“全局共享 B-spline boundary network”的最小必要集；4–6 为质量与可交换性的增强项。本次仅列项，未实施任何修改。
