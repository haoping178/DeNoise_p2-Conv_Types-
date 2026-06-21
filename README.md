# DeNoise_p2-Conv_Types-

## 1️.專案簡介

本專案採用卷積神經網路（CNN）與融合學習技術，專門針對背景噪音去除問題進行最佳化設計。

### 技術核心

- **主要演算法**：SC-CHM（聲道-通道諧波幅度）融合模型
- **工作原理**：透過多層級特徵提取（聲道級 + 頻譜級），同時學習短時和長時依賴關係
- **應用場景**：語音增強、通話品質改善、音訊清潔、語音識別前置處理
- **目標指標**：降低 SNR（信噪比）、改善 PESQ 和 STOI 評分


## 2.主要功能


| 功能模組           | 詳細說明                                           |
| ------------------ | -------------------------------------------------- |
| **多架構支援**   | Conv-TasNet、U-Net、Encoder-Decoder 等多種網絡設計 |
| **靈活資料管理** | 支援批量處理、動態批大小調整、多工作進程加載       |
| **完整訓練框架** | 自動混合精度（AMP）、分散式訓練、檢查點恢復        |
| **模型管理** | 自動保存最佳模型、早期停止、學習率動態調整         |
| **音訊前置處理** | MFCC 提取、譜圖增強、數據增強技術                  |
| **完整評估工具** | 即時監控、可視化圖表、多指標評估                   |

---

## 3️.詳細專案結構

```
denoiser-main/
│
├──  核心模型與架構
│   ├── model.py                    # 基礎模型框架
│   │   ├── ChannelWiseLayerNorm   # 通道級層正規化
│   │   ├── ResidualBlock           # 殘差連接模組
│   │   └── param()                 # 參數統計工具
│   │
│   └── model_SC_CHM_Fusion.py      # SC-CHM 融合模型 
│       ├── SpeechChannelEncoder    # 聲道編碼器
│       ├── ChannelHarmonicsDecoder # 諧波解碼器
│       └── FusionLayer             # 特徵融合層
│
├──  訓練與優化
│   ├── trainer.py                  # 訓練迴圈核心
│   │   ├── train_epoch()           # 單個 epoch 訓練
│   │   ├── validate()              # 驗證集評估
│   │   ├── save_checkpoint()       # 檢查點保存
│   │   └── lr_scheduler            # 學習率排程
│   │
│   ├── trainnew_blue.py           # 新版訓練指令碼
│   │   ├── 命令列參數解析
│   │   ├── 模型初始化
│   │   └── 訓練流程協調
│   │
│   └── train_blue.sh               # Shell 訓練啟動腳本
│       └── 環境配置與執行
│
├──  資料與配置
│   ├── dataset.py                  # 資料集加載
│   │   ├── AudioDataset            # 自定義資料集類
│   │   ├── load_audio()            # 音訊讀取
│   │   ├── __getitem__()           # 批量取樣
│   │   └── augmentation()          # 資料增強
│   │
│   ├── audio.py                    # 音訊 I/O 處理
│   │   ├── write_wav()             # 儲存 WAV 檔案
│   │   ├── read_wav()              # 讀取 WAV 檔案
│   │   ├── normalize()             # 正規化
│   │   ├── apply_stft()            # 時頻轉換
│   │   └── MAX_INT16               # 音訊量度常數
│   │
│   ├── conf.py                     # 全局配置檔案
│   │   ├── trainer_conf{}          # 訓練超參數
│   │   ├── nnet_conf{}             # 模型超參數
│   │   ├── train_data              # 訓練資料路徑
│   │   ├── dev_data                # 驗證資料路徑
│   │   └── chunk_size              # 音訊分塊大小
│   │
│   └── gen_scp.py                  # 資料列表生成器
│       ├── 掃描資料目錄
│       ├── 生成 .scp 檔案
│       └── 清單驗證
│
├──  工具與日誌
│   └── utils.py                    # 通用工具函數
│       ├── get_logger()            # 日誌系統
│       ├── count_parameters()      # 模型參數計數
│       ├── set_seed()              # 隨機種子設定
│       ├── load_obj()              # 物件 CUDA 轉移
│       └── metrics_compute()       # 指標計算
│
├──  模型檢查點目錄
│   └── ckpt/
│       ├── data.json               # 資料配置快照
│       ├── mdl.json                # 模型狀態快照
│       ├── trainer.json            # 訓練器狀態快照
│       ├── best_model.pth          # 最佳模型權重
│       └── last_checkpoint.pth     # 最新檢查點
│
└──  documentation
    └── README.md                   # 本檔案
```

---

## 4️.詳細環境要求

### 最低要求

```
Python 3.6+              (建議 3.8 或以上)
PyTorch 1.5.0+           (建議最新穩定版本)
CUDA 10.1+               (如需 GPU 加速)
cuDNN 7.6+               (配合 CUDA 使用)
```

### 核心套件版本

```
torch>=1.5.0             # 深度學習框架
torchvision>=0.6.0       # 視覺工具（選用）
numpy>=1.19.0            # 數值計算
scipy>=1.5.0             # 科學計算
matplotlib>=3.3.0        # 圖表繪製（選用）
librosa>=0.8.0           # 音訊分析（選用）
tensorboard>=2.3         # 訓練監控（選用）
```

### 硬體建議


| 配置       | 推薦規格                            |
| ---------- | ----------------------------------- |
| **CPU**    | Intel i7/Ryzen 7 或同級以上         |
| **GPU**    | NVIDIA RTX 2080 或更高（8GB+ VRAM） |
| **記憶體** | 16GB+ RAM                           |
| **儲存**   | 256GB+ SSD                          |

---

## 5️.詳細安裝步驟

### 步驟 1: 環境準備

```bash
# 檢查 Python 版本
python --version

# 檢查是否有 CUDA（GPU 使用者）
nvidia-smi
```

### 步驟 2: 克隆與初始化

```bash
# 克隆專案
git clone <your-repo-url>
cd denoiser-main

# 初始化 git 子模組（如有）
git submodule update --init --recursive
```

### 步驟 3: 建立虛擬環境

```bash
# 使用 venv（推薦）
python -m venv venv

# 啟動虛擬環境
# Linux/macOS
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat
```

### 步驟 4: 安裝依賴

#### 4.1 安裝 PyTorch（根據系統選擇）

```bash
# CPU 版本
pip install torch torchvision torchaudio

# GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 版本 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4.2 安裝其他依賴

```bash
# 安裝核心套件
pip install numpy scipy matplotlib

# 安裝可選工具
pip install librosa tensorboard tqdm

# 或一次安裝所有（建立 requirements.txt）
pip install -r requirements.txt
```

### 步驟 5: 驗證安裝

```bash
# 測試 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.is_available()}')"

# 測試其他套件
python -c "import numpy, scipy, matplotlib; print('All packages OK')"
```

---

## 6️.快速開始指南

### 方式 1: 使用 Shell 腳本

```bash
# 確保腳本有執行權限
chmod +x train_blue.sh

# 執行訓練
bash train_blue.sh

# 腳本會自動執行：
# 1. 環境檢查
# 2. 資料準備
# 3. 模型初始化
# 4. 訓練迴圈
# 5. 結果保存
```

### 方式 2: 使用 Python 腳本

```bash
# 基本訓練
python trainnew_blue.py

# 自定義參數訓練
python trainnew_blue.py \
    --batch-size 64 \
    --epochs 200 \
    --learning-rate 0.001 \
    --gpus 0,1 \
    --mixed-precision \
    --save-dir ./checkpoints

# 從檢查點恢復訓練
python trainnew_blue.py \
    --resume-from ./ckpt/last_checkpoint.pth \
    --epochs 300
```

### 方式 3: 互動式訓練（開發調試）

```python
# 在 Python 互動環境或 Jupyter Notebook 中
from trainer import Trainer
from conf import trainer_conf, nnet_conf

trainer = Trainer(nnet_conf, trainer_conf)
trainer.run(num_epochs=100)
trainer.save_checkpoint('my_model.pth')
```

---

## 7️.詳細配置說明

編輯 `conf.py` 檔案以自定義訓練參數：

### 訓練器配置

```python
trainer_conf = {
    # ========== 最佳化參數 ==========
    'learning_rate': 1e-3,           # 初始學習率 (0.0001 ~ 0.01)
    'weight_decay': 1e-5,            # L2 正規化係數
    'momentum': 0.9,                 # SGD 動量（若使用 SGD）
    'optimizer': 'adam',             # 最佳化器選擇
  
    # ========== 訓練超參數 ==========
    'epochs': 100,                   # 訓練輪數
    'batch_size': 32,                # 批次大小（根據 GPU 記憶體調整）
    'max_grad_norm': 10.0,           # 梯度裁剪閾值
    'gradient_accumulation': 2,      # 梯度累積步數
  
    # ========== 學習率排程 ==========
    'lr_scheduler': 'cosine',        # warmup, cosine, exponential
    'warmup_steps': 1000,            # 預熱步數
    'lr_decay_factor': 0.5,          # 衰減係數
    'patience': 10,                  # 早期停止耐心值
  
    # ========== 損失函數 ==========
    'loss_type': 'mse',              # mse, l1, huber, mix
    'loss_weights': [1.0, 0.5],     # 多任務加權
  
    # ========== 檢查點設定 ==========
    'save_interval': 5,              # 保存間隔（epoch 數）
    'keep_last_k': 3,               # 保留最後 K 個檢查點
    'save_best_only': True,          # 僅保存最佳模型
}

nnet_conf = {
    # ========== 模型架構 ==========
    'model_type': 'sc_chm_fusion',   # 模型類型
    'num_layers': 10,                # 網絡層數
    'hidden_size': 512,              # 隱層維度
    'num_heads': 8,                  # 多頭注意力頭數（若使用）
  
    # ========== 編碼器配置 ==========
    'enc_dim': 128,                  # 編碼器輸出維度
    'kernel_size': 21,               # 卷積核大小
    'conv_channels': 64,             # 卷積通道數
  
    # ========== 正規化 ==========
    'norm_type': 'layer',            # batch, layer, group
    'dropout_rate': 0.1,             # Dropout 比率
  
    # ========== 輸入/輸出 ==========
    'sample_rate': 16000,            # 採樣率
    'frame_length': 512,             # 幀長度
    'frame_shift': 128,              # 幀位移
}

# ========== 資料路徑 ==========
train_data = '/path/to/train_data'
dev_data = '/path/to/dev_data'
test_data = '/path/to/test_data'

# ========== 資料參數 ==========
chunk_size = 16000                   # 音訊段長度（樣本數）
num_workers = 4                      # 資料加載工作進程數
prefetch_factor = 2                  # 預取係數
pin_memory = True                    # GPU 記憶體鎖定
```

---

## 8️.模型說明

### Model 架構詳解

#### `model.py` - 基礎元件

```python
# ChannelWiseLayerNorm: 逐通道正規化
# 公式: y = γ * (x - μ) / √(σ² + ε) + β
# 好處：穩定訓練、加速收斂、減少內部協變量位移

# 重要參數計算函數
param(nnet)  # 返回模型參數數量（百萬）
```

#### `model_SC_CHM_Fusion.py` - 融合模型架構 

```
輸入音訊 [B, 1, T]
    ↓
┌─────────────────────────────────┐
│   聲道編碼器 (Speech Channel)    │
│  ◆ 時域卷積特徵提取             │
│  ◆ 長短時依賴建模               │
└─────────────────────────────────┘
         ↓
  中間表徵 [B, 256, T']
    ↙           ↖
┌───────────────┐  ┌──────────────┐
│ 聲道解碼器  │  │ 諧波解碼器  │
│(Speech D.)│  │(Harmonics) │
└───────────────┘  └──────────────┘
    ↓                  ↓
 去噪路徑1          頻譜增強路徑
    ↓                  ↓
┌─────────────────────────────────┐
│       融合層 (Fusion Layer)      │
│  ◆ 特徵加權融合                 │
│  ◆ 上下文感知調適               │
└─────────────────────────────────┘
        ↓
  輸出音訊 [B, 1, T]
```

**核心創新**：

- **SC 分支**：捕捉語音本質特徵（音素、聲調等）
- **CHM 分支**：增強諧波結構（改善音色清晰度）
- **融合機制**：自適應加權與跨層連接

**技術優勢**：

```
✓ 語音特定設計（非通用去噪）
✓ 多層級表徵學習
✓ 計算效率高（參數少）
✓ 對新噪音類型魯棒性強
```

---

## 9️.深入訓練指南

### 準備階段

#### 1. 資料集組織

```
data/
├── train/                          
│   ├── clean/
│   │   ├── speaker_001.wav
│   │   ├── speaker_002.wav
│   │   └── ...
│   └── noisy/
│       ├── speaker_001_noisy.wav
│       └── ...
├── dev/                            
│   ├── clean/
│   └── noisy/
└── test/                          
    ├── clean/
    └── noisy/
```

#### 2. 生成資料列表

```bash
# 掃描資料目錄
python gen_scp.py \
    --train-dir data/train \
    --dev-dir data/dev \
    --test-dir data/test \
    --output-dir scp_files

# 生成檔案
# train.scp: 訓練列表
# dev.scp: 驗證列表
# test.scp: 測試列表
```

### 訓練階段

#### 3. 開始訓練

```bash
# 基本執行
python trainnew_blue.py

# 詳細參數執行
python trainnew_blue.py \
    --config conf.py \
    --batch-size 64 \
    --epochs 200 \
    --learning-rate 0.001 \
    --device 0 \
    --seed 42 \
    --log-dir logs \
    --save-dir checkpoints

# 監視訓練過程
tensorboard --logdir logs
```

#### 4. 訓練監控指標

```
Epoch [1/200]
├─ Train Loss: 0.0352 ✓
├─ Dev Loss: 0.0389 ✓
├─ Learning Rate: 1.00e-03
├─ Batch Time: 125ms
├─ Best Dev Loss: 0.0389 (saved!)
└─ Time Remaining: ~8h 30m
```

### 最佳化技巧

#### 5. 進階訓練策略

```python
# 分段訓練（預訓練 → 微調）
# 第 1 階段: 預訓練音訊編碼器
python trainnew_blue.py --stage 1 --epochs 50

# 第 2 階段: 訓練完整模型
python trainnew_blue.py --stage 2 --epochs 150 --lr 0.0005

# 知識蒸餾（教師-學生模型）
python trainnew_blue.py --teacher-model best_model.pth --distill

# 混合精度訓練（加快速度）
python trainnew_blue.py --amp --batch-size 128
```

---


```

#### 檔案命名規範

```
train/clean/speaker_001.wav       # 清晰音訊
train/noisy/speaker_001_noisy.wav # 對應帶噪音訊

配對規則: 同一說話人的清晰/帶噪版本必須配對
```

### 資料增強技術

```python
# audio.py 內建增強
augmentations = [
    "time_shift",          # 時移 (-10% ~ +10%)
    "pitch_shift",         # 音高位移 (±2 半音)
    "speed_perturbation",  # 速度擾動 (0.9x ~ 1.1x)
    "mix_speech",          # 混合多說話人
    "add_reverb",          # 加入混響
    "dynamic_range_comp",  # 動態範圍壓縮
]
```

## 10. loss vs epoch

<img width="1082" height="642" alt="image" src="https://github.com/user-attachments/assets/d59cc047-0064-4913-92f7-2c328d9cc12e" />


---

