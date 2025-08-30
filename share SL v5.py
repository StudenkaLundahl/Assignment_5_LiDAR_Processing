'''
LiDAR Point Cloud Processing for Power Line Corridor Analysis

This project processes LiDAR point cloud data to identify key objects in power line corridors:
- Task 1 (Grade 3): Ground level detection and removal using histogram analysis
- Task 2 (Grade 4): DBSCAN clustering with optimal epsilon estimation via k-distance method
- Task 3 (Grade 5): Catenary identification using XY span analysis

Author: Studenka Lundahl
Course: Industrial AI and eMaintenance - Part I: Theories & Concepts
Date: 2025
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime


# ==========================================================
# Visualization Functions
# ==========================================================

def show_cloud(points, title="Point Cloud", dataset_name=None):
    """
    Display a 3D point cloud visualization.
    
    Args:
        points: numpy array of shape (n,3) containing [x,y,z] coordinates
        title: plot title
        dataset_name: if provided, saves the figure
    """
    ax = plt.axes(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.01)
    ax.set_title(title, pad=20)
    
    if dataset_name:
        filename = f"3d_view_{dataset_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"3D plot saved as: {filename}")
    
    plt.tight_layout()
    plt.show()


def show_scatter(x, y, title="Scatter Plot", dataset_name=None):
    """
    Display a 2D scatter plot.
    
    Args:
        x, y: coordinate arrays
        title: plot title
        dataset_name: optional name for saving
    """
    plt.scatter(x, y, s=2)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    
    if dataset_name:
        filename = f"scatter_{dataset_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved as: {filename}")
    
    plt.show()


# ==========================================================
# Task 1: Ground Level Detection
# ==========================================================

def get_ground_level(pcd, dataset_name=None, bins=100, buffer=1.0):
    """
    Estimate ground level using histogram peak detection.
    
    The method finds the most frequent Z-value (mode) and adds a buffer
    to ensure effective ground plane removal.
    
    Args:
        pcd: point cloud array (n×3)
        dataset_name: name for saving plots
        bins: number of histogram bins
        buffer: height buffer above detected mode
        
    Returns:
        float: estimated ground level
    """
    z_values = pcd[:, 2]
    hist, bin_edges = np.histogram(z_values, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Find peak (mode) and add buffer
    peak_idx = np.argmax(hist)
    ground_mode = float(bin_centers[peak_idx])
    final_ground_level = ground_mode + float(buffer)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(z_values, bins=bins, alpha=0.8, color="royalblue",
            edgecolor="black", label="Z-coordinate distribution")
    
    # Add ground level indicator
    ax.axvline(final_ground_level, color="red", linestyle="--", linewidth=2,
               label=f"Ground level: {final_ground_level:.2f} m")
    
    # Add annotation

    ax.set_title(f"Ground Level Detection - {dataset_name if dataset_name else 'Dataset'}")
    ax.set_xlabel("Z-coordinate (m)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", framealpha=0.95)
    
    # Save plot
    if dataset_name:
        filename = f"histogram_{dataset_name}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Histogram saved as: {filename}")
    plt.show()
    
    print(f"[Task 1] Detected ground mode: {ground_mode:.2f} m")
    print(f"[Task 1] Final ground level (mode + buffer={buffer:.2f}): {final_ground_level:.2f} m")
    
    return final_ground_level


def analyze_ground_removal(pcd, ground_level, dataset_name=None):
    """
    Analyze and visualize the effect of ground removal.
    
    Args:
        pcd: original point cloud
        ground_level: threshold for ground removal
        dataset_name: name for saving plots
        
    Returns:
        numpy array: points above ground level
    """
    pcd_above = pcd[pcd[:, 2] > ground_level]
    removed = pcd.shape[0] - pcd_above.shape[0]
    
    print(f"[Task 1] Original points: {pcd.shape[0]}")
    print(f"[Task 1] Points above ground ({ground_level:.2f} m): {pcd_above.shape[0]}")
    print(f"[Task 1] Removed (ground): {removed}")
    print(f"[Task 1] Kept: {(pcd_above.shape[0] / pcd.shape[0]) * 100:.2f}%")
    
    # Create analysis visualization
    plt.figure(figsize=(12, 5))
    plt.hist(pcd[:, 2], bins=50, alpha=0.6, label="All points", color="blue")
    plt.axvline(ground_level, color="r", linestyle="--", 
                label=f"Ground level: {ground_level:.2f} m")
    plt.hist(pcd_above[:, 2], bins=50, alpha=0.6, label="Above ground", color="green")
    plt.legend()
    plt.xlabel("Z-coordinate (m)")
    plt.ylabel("Frequency")
    plt.title(f"Ground Removal Analysis - {dataset_name if dataset_name else ''}")
    
    if dataset_name:
        filename = f"ground_analysis_{dataset_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Ground analysis saved as: {filename}")
    
    plt.show()
    return pcd_above


# ==========================================================
# Task 2: DBSCAN with Optimal Epsilon
# ==========================================================

def find_optimal_eps(pcd, min_samples=5, dataset_name=None):
    """
    Estimate optimal epsilon for DBSCAN using k-distance elbow method.
    
    The method computes k-nearest neighbor distances and uses the 90th percentile
    as a robust elbow detection heuristic.
    
    Args:
        pcd: point cloud data (n×3)
        min_samples: minimum samples parameter for DBSCAN
        dataset_name: name for saving plots
        
    Returns:
        float: optimal epsilon value
    """
    print(f"[Task 2] Finding optimal epsilon using k-distance method (k={min_samples})...")
    
    # Compute k-nearest neighbor distances
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(pcd)
    distances, _ = nbrs.kneighbors(pcd)
    
    # Sort k-th neighbor distances
    kth_distances = np.sort(distances[:, -1])  # Last column is k-th neighbor
    
    # Use 90th percentile as robust elbow proxy
    eps = float(np.percentile(kth_distances, 90))
    
    # Create k-distance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(kth_distances, lw=1.5, color='blue')
    ax.set_title(f"K-Distance Graph - {dataset_name if dataset_name else 'Dataset'}")
    ax.set_xlabel("Points (sorted by k-distance)")
    ax.set_ylabel("k-distance")
    
    # Add epsilon line and annotation
    ax.axhline(eps, color="red", linestyle="--", linewidth=2)
    
    # Position text annotation
    ymin, ymax = ax.get_ylim()
    pad = 0.015 * (ymax - ymin)
    ax.text(0.60 * len(kth_distances), eps + pad,
            f"eps ≈ {eps:.3f}",
            color="red", va="bottom",
            bbox=dict(facecolor="white", edgecolor="white", 
                      boxstyle="round,pad=0.2", alpha=0.9))
    
    # Add summary annotation
    ax.text(0.5, 0.98, f"Optimal eps ≈ {eps:.3f} (min_samples={min_samples})",
            transform=ax.transAxes, ha="center", color="red", va="top",
            bbox=dict(facecolor="white", edgecolor="white", 
                      boxstyle="round,pad=0.25", alpha=0.95),
            fontsize=11)
    
    if dataset_name:
        filename = f"elbow_{dataset_name}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Elbow plot saved as: {filename}")
    plt.show()
    
    print(f"[Task 2] Estimated optimal eps (90th percentile): {eps:.3f}")
    return eps


def apply_dbscan(pcd, eps, min_samples=5, dataset_name=None):
    """
    Apply DBSCAN clustering with given parameters.
    
    Args:
        pcd: point cloud data (n×3)
        eps: epsilon parameter
        min_samples: minimum samples parameter
        dataset_name: name for saving plots
        
    Returns:
        DBSCAN clustering object
    """
    print(f"[Task 2] Applying DBSCAN with eps={eps:.3f}, min_samples={min_samples}")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pcd)
    labels = clustering.labels_
    
    # Count clusters and noise
    noise_mask = labels == -1
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(unique_clusters)
    n_noise = int(np.sum(noise_mask))
    
    print(f"[Task 2] Results: {n_clusters} clusters, {n_noise} noise points")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot noise points
    if np.any(noise_mask):
        ax.scatter(pcd[noise_mask, 0], pcd[noise_mask, 1],
                   s=2, c="black", alpha=0.5, label="Noise")
    
    # Plot clusters with different colors
    if n_clusters > 0:
        cmap = plt.get_cmap("tab20")
        for i, cluster_id in enumerate(unique_clusters):
            mask = labels == cluster_id
            ax.scatter(pcd[mask, 0], pcd[mask, 1], 
                      s=2, color=cmap(i % 20), alpha=0.95)
        
        # Create legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', linestyle='',
                          markersize=6, markerfacecolor='tab:blue',
                          markeredgecolor='none', label='Clusters')]
        if np.any(noise_mask):
            handles.append(Line2D([0], [0], marker='o', linestyle='',
                                  markersize=6, markerfacecolor='black',
                                  markeredgecolor='none', label='Noise'))
        
        ax.legend(handles=handles, loc="upper right", 
                 bbox_to_anchor=(0.98, 0.98), framealpha=0.95, fontsize=9)
    
    ax.set_title(f"DBSCAN Clustering - {dataset_name or ''} "
                f"(clusters={n_clusters}, noise={n_noise})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    
    if dataset_name:
        filename = f"clusters_{dataset_name}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Cluster plot saved as: {filename}")
    
    plt.show()
    return clustering


# ==========================================================
# Task 3: Catenary Detection by XY Span
# ==========================================================

def find_catenary_by_span(pcd, clustering, dataset_name=None, span_metric="max"):
    """
    Identify the catenary cluster by largest XY span.
    
    This method is more appropriate than population count for power lines,
    as catenaries are characterized by their long linear span rather than
    point density.
    
    Args:
        pcd: point cloud data (n×3)
        clustering: DBSCAN clustering result
        dataset_name: name for saving plots
        span_metric: "max" for max(Δx, Δy) or "area" for Δx * Δy
        
    Returns:
        tuple: (min_x, min_y, max_x, max_y) of catenary bounding box
    """
    labels = clustering.labels_
    noise_mask = labels == -1
    cluster_mask = labels >= 0
    
    if not np.any(cluster_mask):
        print("[Task 3] No valid clusters found (all noise).")
        return None
    
    # Find cluster with largest XY span
    best_label = None
    best_score = -np.inf
    best_bbox = None
    
    print(f"[Task 3] Analyzing clusters using '{span_metric}' span metric...")
    
    for cluster_id in np.unique(labels[cluster_mask]):
        pts = pcd[labels == cluster_id]
        min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
        max_x, max_y = np.max(pts[:, 0]), np.max(pts[:, 1])
        dx, dy = max_x - min_x, max_y - min_y
        
        if span_metric == "area":
            score = dx * dy
        else:  # "max" - recommended for linear structures like cables
            score = max(dx, dy)
        
        if score > best_score:
            best_score = score
            best_label = cluster_id
            best_bbox = (float(min_x), float(min_y), float(max_x), float(max_y))
    
    # Extract coordinates for reporting
    cx0, cy0, cx1, cy1 = best_bbox
    print(f"[Task 3] Selected catenary cluster: {best_label}")
    print(f"[Task 3] Catenary XY bounding box:")
    print(f"         min(x)={cx0:.3f}, min(y)={cy0:.3f}")
    print(f"         max(x)={cx1:.3f}, max(y)={cy1:.3f}")
    
    # Create visualization
    catenary_mask = labels == best_label
    other_clusters_mask = np.logical_and(cluster_mask, ~catenary_mask)
    
    plt.figure(figsize=(10, 10))
    
    # Plot different point types
    if np.any(noise_mask):
        plt.scatter(pcd[noise_mask, 0], pcd[noise_mask, 1], 
                   s=2, c="black", alpha=0.4, label="Noise")
    if np.any(other_clusters_mask):
        plt.scatter(pcd[other_clusters_mask, 0], pcd[other_clusters_mask, 1], 
                   s=2, c="#7ec8e3", alpha=0.95, label="Other clusters")
    
    plt.scatter(pcd[catenary_mask, 0], pcd[catenary_mask, 1], 
               s=3, c="red", alpha=0.95, label="Catenary (largest XY span)")
    
    plt.title(f"Catenary Detection by XY Span - {dataset_name if dataset_name else ''}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc="best", framealpha=0.95)
    
    if dataset_name:
        filename = f"catenary_{dataset_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Catenary cluster plot saved as: {filename}")
    plt.show()
    
    return best_bbox


# ==========================================================
# Documentation and Summary Functions
# ==========================================================

def write_per_dataset_summary(name, ground_level, eps, bbox, n_clusters, n_noise, kept_pct):
    """Write a summary file for each dataset."""
    lines = [
        f"Dataset: {name}",
        f"Ground level (final): {ground_level:.3f} m",
        f"DBSCAN eps (estimated): {eps:.3f}",
        f"Clusters (excl. noise): {n_clusters}",
        f"Noise points: {n_noise}",
        f"Kept after ground removal: {kept_pct:.2f}%",
    ]
    if bbox is not None:
        lines.append(
            "Catenary bbox (XY): "
            f"min(x)={bbox[0]:.3f}, min(y)={bbox[1]:.3f}, "
            f"max(x)={bbox[2]:.3f}, max(y)={bbox[3]:.3f}"
        )
    else:
        lines.append("Catenary bbox (XY): N/A (no clusters)")
    
    text = "\n".join(lines) + "\n"
    fname = f"summary_{name}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Summary] Dataset summary saved: {fname}")
    return text


def write_combined_markdown(summaries_blocks):
    """Generate comprehensive README content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [
        "# LiDAR Processing Results Summary",
        f"_Generated: {now}_",
        "",
        "## Task 1 — Ground Level Detection",
        "- Ground level detected using Z-histogram peak (mode) plus safety buffer",
        "- Method ensures robust ground plane removal for overhead structure analysis",
        "- See: `histogram_datasetX.png` and `ground_analysis_datasetX.png`",
        "",
        "## Task 2 — DBSCAN Optimal Epsilon",
        "- Epsilon estimated using k-distance elbow method with 90th percentile heuristic",
        "- Approach provides robust clustering without manual parameter tuning", 
        "- See: `elbow_datasetX.png` and `clusters_datasetX.png`",
        "",
        "## Task 3 — Catenary Detection",
        "- Catenary identified by largest XY span (more appropriate than point count for linear structures)",
        "- Method correctly identifies power line cables based on their geometric properties",
        "- See: `catenary_datasetX.png`",
        "",
        "---",
        ""
    ]
    body = "\n\n".join(summaries_blocks)
    md = "\n".join(header) + body + "\n"
    
    with open("results_summary.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("[Summary] Combined markdown saved: results_summary.md")
    return md


def make_readme_block(name, ground_level, eps, bbox, n_clusters, n_noise, kept_pct):
    """Format individual dataset results for README."""
    bbox_str = (f"min(x)={bbox[0]:.3f}, min(y)={bbox[1]:.3f}, "
                f"max(x)={bbox[2]:.3f}, max(y)={bbox[3]:.3f}"
                if bbox is not None else "N/A (no clusters)")
    
    block = [
        f"### {name}",
        f"- **Ground level:** `{ground_level:.3f} m`",
        f"- **Optimal eps:** `{eps:.3f}`",
        f"- **Clusters (excl. noise):** `{n_clusters}` · **Noise points:** `{n_noise}`",
        f"- **Points kept after ground removal:** `{kept_pct:.2f}%`",
        f"- **Catenary XY bounding box:** `{bbox_str}`",
        "",
        "**Generated files:**",
        f"- `histogram_{name}.png`, `ground_analysis_{name}.png`",
        f"- `elbow_{name}.png`, `clusters_{name}.png`",
        f"- `catenary_{name}.png`",
        "",
        "---",
        ""
    ]
    return "\n".join(block)


# ==========================================================
# Main Processing Pipeline
# ==========================================================

if __name__ == "__main__":
    print("="*70)
    print("LIDAR POINT CLOUD PROCESSING FOR POWER LINE CORRIDOR ANALYSIS")
    print("="*70)
    
    summaries_blocks = []
    
    # Process both datasets
    for dataset_file in ["dataset1.npy", "dataset2.npy"]:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        print("\n" + "=" * 60)
        print(f"Processing {dataset_name}")
        print("=" * 60)
        
        # Load dataset
        pcd = np.load(dataset_file)
        print(f"Loaded {dataset_name}: {pcd.shape[0]} points")
        
        # TASK 1: Ground level detection and removal
        print(f"\n--- Task 1: Ground Level Detection ---")
        ground_level = get_ground_level(pcd, dataset_name, bins=100, buffer=1.0)
        pcd_above_ground = analyze_ground_removal(pcd, ground_level, dataset_name)
        kept_percentage = (pcd_above_ground.shape[0] / pcd.shape[0]) * 100.0
        
        # TASK 2: DBSCAN clustering with optimal epsilon
        print(f"\n--- Task 2: DBSCAN Clustering ---")
        min_samples = 5
        optimal_eps = find_optimal_eps(pcd_above_ground, min_samples=min_samples, 
                                     dataset_name=dataset_name)
        clustering = apply_dbscan(pcd_above_ground, optimal_eps, 
                                min_samples=min_samples, dataset_name=dataset_name)
        
        # Count final results
        labels = clustering.labels_
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_noise = int(np.sum(labels == -1))
        
        # TASK 3: Catenary detection
        print(f"\n--- Task 3: Catenary Detection ---")
        catenary_bbox = find_catenary_by_span(pcd_above_ground, clustering, 
                                            dataset_name=dataset_name, span_metric="max")
        
        # Generate documentation
        print(f"\n--- Generating Documentation ---")
        write_per_dataset_summary(
            name=dataset_name,
            ground_level=ground_level,
            eps=optimal_eps,
            bbox=catenary_bbox,
            n_clusters=n_clusters,
            n_noise=n_noise,
            kept_pct=kept_percentage
        )
        
        readme_block = make_readme_block(
            name=dataset_name,
            ground_level=ground_level,
            eps=optimal_eps,
            bbox=catenary_bbox,
            n_clusters=n_clusters,
            n_noise=n_noise,
            kept_pct=kept_percentage
        )
        summaries_blocks.append(readme_block)
    
    # Generate final documentation
    print("\n" + "=" * 60)
    print("GENERATING FINAL DOCUMENTATION")
    print("=" * 60)
    write_combined_markdown(summaries_blocks)
    
    print(f"\n🎉 Processing completed successfully!")
    print(f"📁 All results, plots, and documentation have been generated.")
    print(f"📋 Check 'results_summary.md' for the complete analysis report.")
