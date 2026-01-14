# ==============================================================================
# Integrated CITE-seq Pipeline (Dynamic QC Version + Feature Filtering)
# Order: Load -> Metadata Merge/Rename -> Dynamic QC Filter (Cells)
#      -> Low-Expression Feature Filter (Genes/Proteins)
#      -> CLR Norm -> Export
# ==============================================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(anndata)
  library(dplyr)
})

# =========================== 1. 用户配置区域 (USER CONFIG) ===========================

# --- 输入路径 ---
PATH_RNA_INPUT  <- "/mnt/scratch/zhan2210/data/PBMC/train_cite_inputs_raw.h5ad"
PATH_PROT_INPUT <- "/mnt/scratch/zhan2210/data/PBMC/train_cite_targets_raw.h5ad"

# --- 输出路径 ---
DIR_OUTPUT      <- "/mnt/scratch/zhan2210/data/PBMC/"
PREFIX_OUTPUT   <- "PBMC_clean"

# --- 1. Metadata 处理开关 ---
DO_PROCESS_METADATA <- FALSE
PATH_MAPPING_CSV    <- "/mnt/scratch/zhan2210/data/SLN-celltype.csv"
COL_ORIGINAL        <- "cell_type"
COL_FINAL           <- "celltypes"
REMOVE_KEYWORD      <- "Removed"

# --- 2. 质控 (QC) - RNA 阈值 (Cell-level) ---
QC_MIN_GENES    <- 200
QC_MAX_GENES    <- 6000
QC_MAX_MITO     <- 15
MITO_PATTERN    <- "^mt-|^MT-"

# --- 3. 质控 (QC) - 动态 ADT 阈值参数 (Cell-level) ---
QC_ADT_PROB_LOW  <- 0.01
QC_ADT_PROB_HIGH <- 0.99
QC_ADT_MIN_FLOOR <- 200
QC_ADT_MAX_CEIL  <- 20000

# --- 4. Feature-level 过滤 (Genes/Proteins) ---
QC_DO_FILTER_FEATURES <- TRUE

# RNA genes（保留你原来的绝对阈值）
QC_FEAT_MIN_CELLS_GENE  <- 20
QC_FEAT_MIN_TOTAL_GENE  <- 100

# ADT proteins（原来的绝对阈值保留，但下面会“改成分位数策略”）
QC_FEAT_MIN_CELLS_PROT  <- 50
QC_FEAT_MIN_TOTAL_PROT  <- 100

# --- ADT feature filtering 新增参数（按百分比/分位数更适合 130 左右蛋白面板）---
QC_ADT_FEAT_LOW_Q_TOTAL <- 0.05   # 删掉 total counts 最低的 10% 蛋白
QC_ADT_FEAT_LOW_Q_RATE  <- 0.05   # 删掉 detect rate 最低的 10% 蛋白
QC_ADT_DETECT_COUNT     <- 1      # counts > 1 才算“检测到”，比 >0 更能排除背景
QC_ADT_FEAT_TOTAL_FLOOR <- 50     # total counts 的最低地板，避免分位数太低
QC_ADT_FEAT_RATE_FLOOR  <- 0.005  # detect rate 最低地板（>=0.5% 细胞）

# ==============================================================================

if(!dir.exists(DIR_OUTPUT)) dir.create(DIR_OUTPUT, recursive = TRUE)

# ------------------------------------------------------------------------------
# [Step 1] 加载数据与构建对象
# ------------------------------------------------------------------------------
cat("\n[1/6] Loading Data...\n")
ad_rna  <- read_h5ad(PATH_RNA_INPUT)
ad_prot <- read_h5ad(PATH_PROT_INPUT)

# 检查对齐
if (!all(rownames(ad_rna$obs) == rownames(ad_prot$obs))) {
  stop("Error: Cell IDs mismatch between RNA and Protein inputs!")
}

# 转换为 Seurat 对象
counts_rna <- Matrix::t(ad_rna$X)
rownames(counts_rna) <- rownames(ad_rna$var)
colnames(counts_rna) <- rownames(ad_rna$obs)

counts_prot <- Matrix::t(ad_prot$X)
rownames(counts_prot) <- rownames(ad_prot$var)
colnames(counts_prot) <- rownames(ad_prot$obs)

# 创建对象
sobj <- CreateSeuratObject(counts = counts_rna, assay = "RNA", meta.data = ad_rna$obs)
sobj[["ADT"]] <- CreateAssayObject(counts = counts_prot)

cat(sprintf("  -> Initial loaded cells: %d\n", ncol(sobj)))
cat(sprintf("  -> Initial genes: %d | proteins: %d\n",
            nrow(sobj[["RNA"]]), nrow(sobj[["ADT"]])))

# ------------------------------------------------------------------------------
# [Step 2] Metadata 合并与清洗
# ------------------------------------------------------------------------------
if (DO_PROCESS_METADATA) {
  cat("\n[2/6] Processing Metadata (Merge/Rename)...\n")
  
  if (!file.exists(PATH_MAPPING_CSV)) stop("Mapping CSV file not found!")
  map_df <- read.csv(PATH_MAPPING_CSV, stringsAsFactors = FALSE)
  mapping_vec <- setNames(map_df[[2]], map_df[[1]])
  
  if (COL_ORIGINAL %in% colnames(sobj@meta.data)) {
    current_labels <- as.character(sobj@meta.data[[COL_ORIGINAL]])
    new_labels <- mapping_vec[current_labels]
    names(new_labels) <- colnames(sobj)
    
    na_idx <- is.na(new_labels)
    if (any(na_idx)) new_labels[na_idx] <- current_labels[na_idx]
    
    sobj <- AddMetaData(sobj, metadata = new_labels, col.name = COL_FINAL)
    
    cells_to_keep <- sobj@meta.data[[COL_FINAL]] != REMOVE_KEYWORD
    if (sum(!cells_to_keep) > 0) {
      cat(sprintf("  -> Removing %d cells labeled as '%s'...\n", sum(!cells_to_keep), REMOVE_KEYWORD))
      sobj <- sobj[, cells_to_keep]
    }
  } else {
    warning(paste("Column", COL_ORIGINAL, "not found in metadata! Skipping mapping."))
  }
} else {
  cat("\n[2/6] Skipping Metadata processing.\n")
  if (COL_ORIGINAL %in% colnames(sobj@meta.data)) {
    sobj[[COL_FINAL]] <- sobj[[COL_ORIGINAL]]
  }
}

# ------------------------------------------------------------------------------
# [Step 3] 动态质控过滤 (Cell-level Dynamic QC)
# ------------------------------------------------------------------------------
cat("\n[3/6] Performing Dynamic Quality Control (RNA & Protein, Cell-level)...\n")

# 1) 计算线粒体比例
sobj[["percent.mt"]] <- PercentageFeatureSet(sobj, pattern = MITO_PATTERN)

# 2) 动态计算 ADT 阈值
adt_counts <- sobj$nCount_ADT
dist_lower <- quantile(adt_counts, probs = QC_ADT_PROB_LOW)
dist_upper <- quantile(adt_counts, probs = QC_ADT_PROB_HIGH)

final_min_adt <- max(dist_lower, QC_ADT_MIN_FLOOR)
final_max_adt <- min(dist_upper, QC_ADT_MAX_CEIL)

cat(sprintf("  -> Dynamic ADT Thresholds: Min > %.1f, Max < %.1f (based on %.1f%%-%.1f%% dist)\n",
            final_min_adt, final_max_adt, QC_ADT_PROB_LOW*100, QC_ADT_PROB_HIGH*100))

# ==============================================================================
# [Debug Tool] 过滤“验尸报告” (Filter Audit) - BEFORE subset
# ==============================================================================
cat("\n--- Quality Control Death Audit (Cell-level) ---\n")

bad_rna_low  <- sobj$nFeature_RNA <= QC_MIN_GENES
bad_rna_high <- sobj$nFeature_RNA >= QC_MAX_GENES
bad_mito     <- sobj$percent.mt   >= QC_MAX_MITO
bad_adt_low  <- sobj$nCount_ADT   <= final_min_adt
bad_adt_high <- sobj$nCount_ADT   >= final_max_adt

cat(sprintf("Total Cells Before Filter: %d\n", ncol(sobj)))
cat("----------------------------------------------------\n")
cat(sprintf("1. Removed by Low Genes (<%d):       %d cells\n", QC_MIN_GENES, sum(bad_rna_low)))
cat(sprintf("2. Removed by High Genes (>%d):      %d cells\n", QC_MAX_GENES, sum(bad_rna_high)))
cat(sprintf("3. Removed by High Mito (>%d%%):       %d cells\n", QC_MAX_MITO, sum(bad_mito)))
cat(sprintf("4. Removed by Low Protein (<%.1f):    %d cells\n", final_min_adt, sum(bad_adt_low)))
cat(sprintf("5. Removed by High Protein (>%.1f):   %d cells\n", final_max_adt, sum(bad_adt_high)))
cat("----------------------------------------------------\n")

all_bad <- bad_rna_low | bad_rna_high | bad_mito | bad_adt_low | bad_adt_high
cat(sprintf("Total Removed (Union of all):       %d cells\n", sum(all_bad)))
cat(sprintf("Remaining Healthy Cells:            %d cells\n", ncol(sobj) - sum(all_bad)))

# 3) 执行 cell 过滤
n_before <- ncol(sobj)
sobj <- subset(
  sobj,
  subset =
    nFeature_RNA > QC_MIN_GENES &
    nFeature_RNA < QC_MAX_GENES &
    percent.mt   < QC_MAX_MITO  &
    nCount_ADT   > final_min_adt &
    nCount_ADT   < final_max_adt
)
n_after <- ncol(sobj)
cat(sprintf("  -> Cell QC Complete. Removed %d cells. Remaining: %d\n", n_before - n_after, n_after))

# ------------------------------------------------------------------------------
# [Step 4] Feature-level 过滤 (Genes/Proteins Low Expression Filter)
# ------------------------------------------------------------------------------
cat("\n[4/6] Performing Feature Filtering (Low-expression genes/proteins)...\n")

if (QC_DO_FILTER_FEATURES) {
  
  # ---- RNA genes（保持你原来的阈值） ----
  rna_counts_mat <- GetAssayData(sobj, assay = "RNA", layer = "counts")
  gene_detect_cells <- Matrix::rowSums(rna_counts_mat > 0)
  gene_total_counts <- Matrix::rowSums(rna_counts_mat)
  
  keep_genes <- (gene_detect_cells >= QC_FEAT_MIN_CELLS_GENE) &
    (gene_total_counts >= QC_FEAT_MIN_TOTAL_GENE)
  
  cat(sprintf("  -> RNA genes before: %d | keep: %d | removed: %d\n",
              nrow(rna_counts_mat), sum(keep_genes), sum(!keep_genes)))
  
  sobj[["RNA"]] <- subset(sobj[["RNA"]], features = names(keep_genes)[keep_genes])
  
  # ---- ADT proteins（改成：分位数 + 更严格 detect） ----
  adt_counts_mat <- GetAssayData(sobj, assay = "ADT", layer = "counts")
  
  prot_total_counts <- Matrix::rowSums(adt_counts_mat)
  prot_detect_cells <- Matrix::rowSums(adt_counts_mat > QC_ADT_DETECT_COUNT)
  prot_detect_rate  <- prot_detect_cells / ncol(sobj)
  
  # 低分位阈值（按百分比删）
  q_total <- quantile(prot_total_counts, probs = QC_ADT_FEAT_LOW_Q_TOTAL)
  q_rate  <- quantile(prot_detect_rate,  probs = QC_ADT_FEAT_LOW_Q_RATE)
  
  min_total <- max(as.numeric(q_total), QC_ADT_FEAT_TOTAL_FLOOR)
  min_rate  <- max(as.numeric(q_rate),  QC_ADT_FEAT_RATE_FLOOR)
  
  cat(sprintf("  -> ADT feature filter thresholds: total>=%.1f (q=%.2f, floor=%d), detect_rate>=%.4f (q=%.2f, floor=%.4f, counts>%d)\n",
              min_total, QC_ADT_FEAT_LOW_Q_TOTAL, QC_ADT_FEAT_TOTAL_FLOOR,
              min_rate,  QC_ADT_FEAT_LOW_Q_RATE,  QC_ADT_FEAT_RATE_FLOOR, QC_ADT_DETECT_COUNT))
  
  keep_prots <- (prot_total_counts >= min_total) & (prot_detect_rate >= min_rate)
  
  cat(sprintf("  -> ADT proteins before: %d | keep: %d | removed: %d\n",
              nrow(adt_counts_mat), sum(keep_prots), sum(!keep_prots)))
  
  sobj[["ADT"]] <- subset(sobj[["ADT"]], features = names(keep_prots)[keep_prots])
  
  DefaultAssay(sobj) <- "RNA"
  
  cat(sprintf("  -> After feature filtering: genes=%d | proteins=%d | cells=%d\n",
              nrow(sobj[["RNA"]]), nrow(sobj[["ADT"]]), ncol(sobj)))
} else {
  cat("  -> Skipping feature filtering.\n")
}

# ------------------------------------------------------------------------------
# [Step 5] CLR 标准化 (Protein Normalization)
# ------------------------------------------------------------------------------
cat("\n[5/6] Performing CLR Normalization (ADT)...\n")
sobj <- NormalizeData(sobj, assay = "ADT", normalization.method = "CLR", margin = 2)

# ------------------------------------------------------------------------------
# [Step 6] 导出结果 (Export Split H5ADs)
# ------------------------------------------------------------------------------
cat("\n[6/6] Exporting results...\n")

# 1) 提取 RNA 矩阵 (Cells x Genes)
rna_out <- Matrix::t(GetAssayData(sobj, assay = "RNA", layer = "counts"))

# 2) 提取 Protein 矩阵 (Cells x Proteins) - CLR normalized data layer
prot_out <- Matrix::t(GetAssayData(sobj, assay = "ADT", layer = "data"))

# 3) Metadata
final_obs <- sobj@meta.data

# 4) Gene metadata：按筛选后的 gene 顺序对齐
final_genes <- colnames(rna_out)
final_var <- ad_rna$var[final_genes, , drop = FALSE]

# 5) Protein metadata：按筛选后的 protein 顺序对齐
final_prot_names <- colnames(prot_out)
final_var_prot <- ad_prot$var[final_prot_names, , drop = FALSE]

# --- 保存 RNA Input ---
path_rna_out <- file.path(DIR_OUTPUT, paste0(PREFIX_OUTPUT, "_input_rna.h5ad"))
ad_rna_final <- AnnData(X = rna_out, obs = final_obs, var = final_var)
write_h5ad(ad_rna_final, path_rna_out)
cat(sprintf("  -> Saved cleaned RNA: %s\n", path_rna_out))

# --- 保存 Protein Target ---
path_prot_out <- file.path(DIR_OUTPUT, paste0(PREFIX_OUTPUT, "_target_protein_clr.h5ad"))
ad_prot_final <- AnnData(X = prot_out, obs = final_obs, var = final_var_prot)
write_h5ad(ad_prot_final, path_prot_out)
cat(sprintf("  -> Saved cleaned Protein (CLR): %s\n", path_prot_out))

cat("\nAll Done!\n")