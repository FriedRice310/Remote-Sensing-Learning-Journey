# 遥感学习项目

这是一个遥感练习项目，旨在通过实践学习 GitHub🐙💻 和遥感技术🌐

**技术栈:** `Python`, `GDAL`, `NumPy`, `Matplotlib`

---

## 学习路径 🗺️

```mermaid
graph LR
A[数据基础] --> B[辐射定标]
B --> C[大气校正]
C --> D[指数计算，如NDVI]
D --> E[机器学习分类]
E --> F[深度学习应用]
```

---

## 目录 📖

1. [01-Foundation](#01-foundation)
   - [01-Raster-IO-with-GDAL](#01-raster-io-with-gdal)
   - [02-Raster-Processing-with-GDAL](#02-raster-processing-with-gdal)
   - [03-NDVI-Calculation](#03-ndvi-calculation)

---

## 01-Foundation

### 01-Raster-IO-with-GDAL

**目标：** 熟悉遥感栅格数据的基本结构，掌握使用GDAL库进行数据读取和简单的处理。

**数据源：** Landsat 9 L1TP数据，(path: 121/row: 40)，获取自 [USGS Earth Explorer](https://earthexplorer.usgs.gov/)

**结果与讨论：**

- 成功完成了真彩色图像的和成，但未进行大气校正，图像雾霾感较重
- 简单的进行了NDVI处理，但没有处理nan和除零的情况

| 真彩色合成 | 简单的NDVI处理 |
|:------------:|:------------:|
| ![真彩色合成实验](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/True%20Color%20Composite%20Image.png) | ![NDVI 处理实验](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/NDVI%20Image.png) |

---

### 02-Raster-Processing-with-GDAL

**目标：** 掌握遥感栅格数据的预处理，特别是大气校正

**结果与讨论：**

- 深刻理解了遥感数字数值（DN）与物理量（反射率、辐亮度）之间的区别

- 基本完成了大气校正，但是预期的黑暗像元法(DOS)未完成，最小辐亮度`Lmin`以0.0代替

- 计划通过 **中值滤波引导** 的方法，先对辐亮度图像进行中值滤波平滑噪声，找到最暗点的坐标，再于**原始图像**上该点周围取一个小窗口，计算其中位数作为`Lmin`，但在实际运行过程中发现最暗点预选部分运行较慢，且频繁追踪云的位置，需要进行改善和优化

**校正效果:**

- 图像整体整体更清晰，雾霾感降低（见对比图）

| 大气校正前 | 大气校正后 |
|:------------:|:------------:|
|![校正前](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/True%20Color%20Composite%20Image.png) |![校正后](01-Foundation/02-Atmospheric-Correction/rough_thumbnail/True%20Color%20Composite%20Image.png) |

---

### 03-NDVI-Calculation

**目标：** 使用大气校正后的数据计算NDVI，并处理了nan和分母为零的情况

**结果与讨论：**

- 由于图像经过了大气校正，结果更加准确（见对比图）

- 发现在部分水体中存在NDVI值较高的情况，通过对比真彩色图像和查找地图发现，该区域存在一个湿地保护区，推测是区内水体的大量水生植物反射近红外光导致NDVI值升高

| 大气校正前（表观反射率） | 大气校正后（地表反射率） |
|:------------:|:------------:|
|![校正前](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/NDVI%20Image.png) |![校正后](01-Foundation/03-NDVI-Calculation/rough_thumbnail/NDVI%20Image.png) |

---

## 关于我

[![GitHub](https://img.shields.io/badge/GitHub-FriedRice310-100000?logo=github)](https://github.com/FriedRice310)

⭐ 学艺不精，欢迎提出建议看法 ❤️

---
