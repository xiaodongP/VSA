# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

**默认使用中文进行交流。**

## 项目概述

基于 **Variational Shape Approximation (VSA)** 方法的 C++ 网格简化实现。给定一个三角网格，VSA 将面片分割为多个区域，对每个区域拟合平面代理，并生成简化后的三角网格。参考论文：[Cohen-Steiner, Alliez, Desbrun](https://hal.inria.fr/file/index/docid/70632/filename/RR-5371.pdf)。

## 构建与运行

需要 **libigl** 作为同级目录（`../libigl`）。CMake 会自动下载 Eigen、GLFW 和 GLAD。

```bash
cd build
cmake -G Ninja ..
cmake --build .
```

在 `build/` 目录下运行：
```bash
./variational_bin <模型名_无后缀> <代理数量> <阈值>
# 示例：
./variational_bin smooth_bunny 180 0.3
# 可选第4个参数 "f" 使用最远点初始化（替代随机初始化）
./variational_bin smooth_bunny 180 0.3 f
```

网格文件（OFF 格式）位于 `data/` 目录。模型名为去掉 `.off` 后缀的文件名。

## 架构

扁平化源码结构——所有 `.cpp` 和 `.h` 文件都在仓库根目录。构建使用 `file(GLOB SRCFILES *.cpp)`，因此添加到根目录的任何 `.cpp` 文件都会被自动编译。

### 非常规的 include 模式

头文件直接包含 `.cpp` 文件（例如 `#include "partitioning.cpp"`），并通过类似 `#ifndef HALFEDGE_DS_HEADER` 的保护宏防止重复包含。不要将这些拆分为传统的编译单元。

### 管线（数据流）

1. **加载网格** → 构建半边数据结构（`HalfedgeDS.cpp`、`HalfedgeBuilder.cpp`）
2. **分区**（`partitioning.cpp`）——面片邻接关系、区域初始化（随机或最远点种子 + 泛洪填充）
3. **迭代**：代理拟合（`proxies.cpp`）→ 区域重分配（按畸变误差贪心泛洪填充）
4. **锚点**（`anchors.cpp`）——找到 3 个及以上区域交汇处的边界顶点，根据阈值添加中间点
5. **三角化**（`triangulation.cpp`）——按最近锚点为顶点着色（Dijkstra），创建简化的面片列表
6. **重编号**（`renumbering.cpp`）——将锚点位置投影到代理平面上

### 关键全局状态

`distance.cpp` 定义了全局数组 `Face_area`、`Face_normal`、`Face_center`。`main.cpp` 保存了网格矩阵（`V`、`F`）、分区标签（`R`）、代理（`Proxies`）、半边结构（`he`）以及 `norme` 标志（0 = L2 距离，1 = L2,1 距离）。

### 交互控制

查看器基于 libigl 的 OpenGL+GLFW 查看器构建。键盘快捷键：`1`=隐藏网格，`2`=显示网格，`3`=单次迭代，`4`=绘制锚点，`5`=三角化，`6`=交替着色，`7`=绘制代理，`8`=10 次迭代，`9`=100 次迭代，`S`=运行至收敛，`L`=切换边显示。

## 代码规范

- 全局使用 `using namespace Eigen;` 和 `using namespace std;`
- 注释和变量名混合使用法语和英语
- `HalfedgeDS.cpp` 和 `HalfedgeBuilder.cpp` 作者为 Luca Castelli Aleardi（巴黎综合理工学院，INF574 课程）
- `treshold`（拼写错误）是控制锚点密度的阈值参数
