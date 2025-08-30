# Assignment_5_LiDAR_Processing
LiDAR Point Cloud Processing for Power Line Analysis - Assignment 5

**Student:** Studenka Lundahl  
**Course:** Industrial AI and eMaintenance - Part I: Theories & Concepts  
**Assignment 5:** Point cloud processing techniques  
**Date:** 2025

## 🎯 Project Overview

This project implements advanced LiDAR point cloud processing techniques for automated power line corridor analysis. The implementation demonstrates ground level detection, density-based clustering optimization, and catenary identification using geometric analysis.

## 🚀 Quick Start

```bash
python Code/share_SL_v5.py
```

**Requirements:**
- Python 3.7+
- NumPy
- Matplotlib  
- Scikit-learn
- Scipy

## 📊 Results Summary

### Task 1: Ground Level Detection
- **Dataset1:** 62.25 m (65.36% points retained)
- **Dataset2:** 62.27 m (75.49% points retained)
- **Method:** Histogram peak detection with buffer
- **Plots:** `Results/histogram_dataset*.png`

### Task 2: DBSCAN Clustering Optimization
- **Dataset1:** eps=0.462 (413 clusters, 2423 noise points)
- **Dataset2:** eps=0.543 (600 clusters, 3397 noise points)  
- **Method:** K-distance elbow method with 90th percentile
- **Note:** Many small clusters are generated, but catenary identification relies on XY span rather than cluster size
- **Plots:** `Results/elbow_dataset*.png`, `Results/clusters_dataset*.png`

### Task 3: Catenary Detection
- **Dataset1:** Cluster 3, XY span: 21.9m × 56.9m
- **Dataset2:** Cluster 2, XY span: 19.2m × 44.8m
- **Method:** Largest XY span analysis for linear structures
- **Plots:** `Results/catenary_dataset*.png`

## 📁 Repository Structure

```
Assignment_5_LiDAR_Processing/
│
├── Code/
│   ├── share_SL_v5.py          # Main implementation
│   ├── dataset1.npy            # Input dataset 1
│   └── dataset2.npy            # Input dataset 2
│
├── Results/
│   ├── histogram_dataset1.png   # Task 1: Ground detection
│   ├── histogram_dataset2.png
│   ├── ground_analysis_dataset1.png
│   ├── ground_analysis_dataset2.png
│   ├── elbow_dataset1.png       # Task 2: DBSCAN optimization
│   ├── elbow_dataset2.png
│   ├── clusters_dataset1.png
│   ├── clusters_dataset2.png
│   ├── catenary_dataset1.png    # Task 3: Catenary detection
│   ├── catenary_dataset2.png
│   ├── summary_dataset1.txt     # Numerical results
│   ├── summary_dataset2.txt
│   └── results_summary.md       # Combined analysis
│
├── Documentation/
│   └── Assignment_5_Report_Studenka_Lundahl.pdf  # Complete technical report
│
└── README.md                    # This file
```

## 🔬 Technical Implementation

### Task 1: Ground Level Detection (Grade 3)
- **Algorithm:** Histogram-based Z-coordinate analysis
- **Key Innovation:** Mode detection with adaptive buffer
- **Output:** Ground level threshold and filtered point cloud

### Task 2: DBSCAN Optimization (Grade 4)  
- **Algorithm:** K-distance elbow method for epsilon estimation
- **Key Innovation:** 90th percentile heuristic for robust parameter selection
- **Output:** Optimized clustering with automated parameter tuning

### Task 3: Catenary Identification (Grade 5)
- **Algorithm:** XY span-based geometric analysis  
- **Key Innovation:** Linear structure detection superior to point count methods
- **Output:** Catenary bounding box coordinates and visualization

## 📈 Key Performance Metrics

| Metric | Dataset 1 | Dataset 2 | Average |
|--------|-----------|-----------|---------|
| Ground Level (m) | 62.25 | 62.27 | 62.26 |
| Points Retained (%) | 65.36 | 75.49 | 70.43 |
| Optimal Eps | 0.462 | 0.543 | 0.503 |
| Clusters Found | 413 | 600 | 507 |
| Noise Ratio (%) | 5.14 | 5.32 | 5.23 |

## 🎓 Grade Achievement

- ✅ **Grade 3:** Ground level detection using histogram analysis
- ✅ **Grade 4:** DBSCAN clustering with optimal epsilon estimation  
- ✅ **Grade 5:** Catenary detection using XY span geometric analysis

## 📖 Documentation

**Complete Technical Report:** [View on GitHub](https://github.com/StudenkaLundahl/Assignment_5_LiDAR_Processing/blob/main/Documentation/Assignment_5_Report_Studenka_Lundahl.pdf)


The technical report includes:
- Detailed methodology explanations
- Algorithm implementation analysis  
- Comprehensive results discussion
- Performance evaluation metrics
- Future improvement recommendations

## 🛠️ Technical Features

**Advanced Implementations:**
- Automated parameter selection (no manual tuning required)
- Robust statistical methods (percentile-based approaches)
- Professional visualization with comprehensive annotations
- Complete documentation generation system
- Geometric analysis optimized for power line structures

**Code Quality:**
- Clean, well-documented Python implementation
- Modular design with reusable functions
- Comprehensive error handling and validation
- Professional-grade plotting and visualization

## 📚 References

Key research papers and resources used:
1. DBSCAN clustering algorithm fundamentals
2. Semantic 3D scene interpretation techniques  
3. Point-based classification for power line corridors
4. Optimal epsilon determination methods

*Complete references available in the technical report.*

## 📧 Contact

**Student:** [Studenka Lundahl]
**Email:** [stulun-5@student.ltu.se]  
**Course:** [Industrial AI and eMaintenance - Part I: Theories & Concepts]

---

*Project completed as part of [Industrial AI and eMaintenance - Part I: Theories & Concepts] - Point Cloud Processing Techniques*
