# 遥感学习项目

这是一个遥感练习项目，旨在通过实践学习 GitHub 和遥感技术。

---

## 目录

1. [01-Foundation](#01-foundation)
   - [01-Raster-IO-with-GDAL](#01-raster-io-with-gdal)
   - [02-Raster-Processing-with-GDAL](#02-raster-processing-with-gdal)
   - [03-NDVI-Calculation](#03-ndvi-calculation)

---

## 01-Foundation

### 01-Raster-IO-with-GDAL

#### 简单处理

| 真彩色合成 | 简单的NDVI 处理 |
|------------|------------|
| ![真彩色合成实验](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/True%20Color%20Composite%20Image.png) | ![NDVI 处理实验](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/NDVI%20Image.png) |

---

### 02-Raster-Processing-with-GDAL

#### 大气校正前后对比

| 大气校正前 | 大气校正后 |
|------------|------------|
|![校正前](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/True%20Color%20Composite%20Image.png) |![校正后](01-Foundation/02-Atmospheric-Correction/rough_thumbnail/True%20Color%20Composite%20Image.png) |

---

### 03-NDVI-Calculation

使用大气校正后的数据，并处理了nan和分母为零的情况

#### NDVI结果展示并与大气校正前对比

| 大气校正前 | 大气校正后 |
|------------|------------|
|![校正前](01-Foundation/01-Raster-IO-with-GDAL/rough_thumbnail/NDVI%20Image.png) |![校正后](01-Foundation/03-NDVI-Calculation/rough_thumbnail/NDVI%20Image.png) |

---
