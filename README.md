# VolTree — Mã nguồn (để review + chạy)

Mã nguồn cho bài báo **VolTree: Earnings-Aware Gradient Boosting with Leaf-Graph Smoothing for
Stock Volatility Forecasting** (`01_paper_VolTree.pdf`).

Thư mục này giữ **đúng cấu trúc đường dẫn** như repo gốc để các `import` (bootstrap `sys.path` theo
đường dẫn tương đối) hoạt động khi chạy.

## Repo đầy đủ (public) — để chạy end-to-end
Toàn bộ codebase + pipeline dữ liệu ở đây (mã nguồn này là bản trích cho tiện review):

    (anonymized repository — see paper code footnote)

Chạy full (mọi horizon, cả 2 thị trường, sinh lại số trong paper) cần **dữ liệu đã xử lý**
(`data/processed/`, `results/gamma_gbm/*.parquet`) — dung lượng lớn, **không kèm** trong bản trích này
(gitignore trong repo). Clone repo để có pipeline dữ liệu đầy đủ.

## Bản đồ mã nguồn → bài báo

| Thành phần trong paper | File |
|---|---|
| **VolTree lõi** — walk-forward XGB (gamma) vs XGB+leaf-graph, DM, spike-robust, α fit (Table 2, §4.2/4.3) | `baselines/2026-09-18_leaf_graph_paper/code/run_leaf_graph_paper.py` |
| Leaf-cooccurrence graph (kNN trên leaf-Hamming, làm mượt ŷ) (§3.5) | `baselines/2026-09-18_leaf_graph_paper/code/leaf_graph_lib.py` |
| Hằng số cấu hình (walk-forward, embargo, MIN_ROWS, α-grid, spike windows) (§3.7) | `baselines/2026-09-18_leaf_graph_paper/code/leaf_graph_paper_config.py` |
| **Lịch earnings leak-free** (predicted-at-origin cadence; §3.4, Fig 2, Limitations) | `baselines/2026-09-19_expected_schedule/code/expected_schedule.py` |
| DM earnings audit (gain 2.7–3.2%, p<1e-3; §5.1) | `baselines/2026-09-19_expected_schedule/code/earnings_dm.py` |
| Panel đặc trưng + OWN-8 + earnings features + XGB champion floor (§3.2) | `scripts/eda/full_matrix.py` |
| Đồ thị tương quan train-only (GNNHAR + fold boundaries `FOLDS`, `TRAIN_START`) (§3.4, §3.7) | `scripts/eda/vn_gbm_graph_stage1.py` |
| Event study earnings (Fig 2) | `scripts/eda/earnings_event_study.py` |
| Chẩn đoán cadence (median error) | `scripts/eda/earnings_pit_cadence.py` |
| **Diebold–Mariano** (date-clustered) | `submission/soict_lstm_gat/metrics.py` |
| Kiểm tra over/under-fit (evidence gate) | `scripts/quality_gate/overfit_check.py` |
| OWN-8 feature list (single source) | `baselines/2026-09-13_paper_models/code/config.py` |
| Thống kê phụ trợ | `baselines/2026-08-21_har_anchored_residual/code/stats.py` |

Baseline **GARCH / HAR / GNNHAR** (Table 1) nằm ở repo: `baselines/2026-09-14_gnnhar/`,
`baselines/2026-09-13_paper_models/code/full_compare.py`, `results/gamma_gbm/garch_*.json`.

## Review nhanh (không cần dữ liệu)
Đọc theo thứ tự: `run_leaf_graph_paper.py` (vòng walk-forward + DM + verdict) → `leaf_graph_lib.py`
(dựng đồ thị leaf + làm mượt) → `expected_schedule.py` (chống rò rỉ: `predict_schedule` dùng
`median(gaps[:i-1])`, chỉ lộ ngày dự đoán từ index ≥ `MIN_HISTORY`).

## Chạy test — 49 test PASS ngay (đã kèm dữ liệu mẫu nhỏ)
Cần Python 3.10+ và `pip install numpy pandas scikit-learn xgboost pyarrow pytest`.

Chạy **RIÊNG từng thư mục test** (chạy gộp 2 baseline cùng lúc sẽ lỗi `conftest` trùng tên — hạn chế
của pytest, không phải lỗi code):

    # từ thư mục source_code/ này
    python -m pytest baselines/2026-09-19_expected_schedule/test -q     # 11 passed  (leak-free schedule + earnings DM)
    python -m pytest baselines/2026-09-18_leaf_graph_paper/test  -q     # 33 passed  (leaf-graph walk-forward)
    python -m pytest scripts/eda/test_earnings_event_study.py    -q     #  5 passed  (event study, Fig 2)

Phần lớn test monkeypatch data loader (dữ liệu giả) để kiểm chứng kiến trúc + **tính không-rò-rỉ**;
một số test đọc dữ liệu mẫu THẬT đã kèm sẵn trong bản trích này:
`data/raw/vn_earnings/hose_disclosures.csv`, `results/gamma_gbm/hose_earnings_combined.parquet`,
`results/gamma_gbm/expected_schedule/earnings_dm_sp500.json`.

## Chạy thật (sinh lại số Table 2) — cần repo đầy đủ
Vòng walk-forward thật đọc **giá OHLCV đã xử lý** (`data/processed/`) — **không kèm** ở đây (lớn,
gitignore). Clone repo public rồi:

    python baselines/2026-09-18_leaf_graph_paper/code/run_leaf_graph_paper.py --market hose  --featureset full
    python baselines/2026-09-18_leaf_graph_paper/code/run_leaf_graph_paper.py --market sp500 --featureset full

Kết quả ghi ra `results/gamma_gbm/leaf_graph_paper_<market>_<full|noearn>_h<h>.json` (1 file/horizon,
kèm DM p-value, gain, spike-robustness, α theo fold) — chính là số trong Table 2 của paper.
