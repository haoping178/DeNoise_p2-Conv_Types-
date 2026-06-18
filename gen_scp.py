import random
from pathlib import Path

def process_split(base_path, split_name, fraction):
    
    # 1. 建立目標資料夾的路徑物件
    split_dir = base_path / split_name
    mix_dir = split_dir / "mix"
    spk1_dir = split_dir / "s1"  # 修改點
    spk2_dir = split_dir / "s2"  # 修改點

    # 檢查最重要的 mix 目錄是否存在
    if not mix_dir.exists():
        print(f"警告：找不到路徑 {mix_dir}，跳過 {split_name} 資料集。")
        return

    # 2. 抓取所有 wav 檔案
    mix_files = list(mix_dir.glob("*.wav"))
    total_files = len(mix_files)
    
    if total_files == 0:
        print(f"⚠️ 警告：{mix_dir} 中沒有 .wav 檔案。")
        return

    # 3. 計算要抽取的數量，並進行隨機抽樣
    sample_size = int(total_files * fraction)
    sampled_mix_files = random.sample(mix_files, sample_size)

    # 4. 定義準備要輸出的三個 .scp 文字檔路徑
    mix_scp = split_dir / "mix.scp"
    spk1_scp = split_dir / "spk1.scp" # 修改點
    spk2_scp = split_dir / "spk2.scp" # 修改點

    # 5. 開啟檔案並寫入資料
    with open(mix_scp, "w", encoding="utf-8") as f_mix, \
         open(spk1_scp, "w", encoding="utf-8") as f_spk1, \
         open(spk2_scp, "w", encoding="utf-8") as f_spk2:
        
        for mix_file in sampled_mix_files:
            filename = mix_file.stem 
            
            # 利用相同檔名，拼湊出 spk1 與 spk2 的正確檔案路徑
            spk1_file = spk1_dir / f"{filename}.wav" # 修改點
            spk2_file = spk2_dir / f"{filename}.wav" # 修改點

            # 寫入格式
            f_mix.write(f"{filename} {mix_file.resolve()}\n")
            f_spk1.write(f"{filename} {spk1_file.resolve()}\n")
            f_spk2.write(f"{filename} {spk2_file.resolve()}\n")

    print(f"✅ [{split_name.upper()}] 處理完成！總數: {total_files} -> 抽樣後: {sample_size}")

def main():    
    base_dir_str = "D:/finals/final_dataset/final_dataset"
    base_path = Path(base_dir_str)
    fraction = 0.5 
    
    print(f"目標資料集根目錄: {base_path.resolve()}")
    print(f"隨機抽樣比例: {fraction} (約 {fraction*100}%)\n")

    splits = ["tr", "cv", "tt"]
    for split in splits:
        process_split(base_path, split, fraction)

if __name__ == "__main__":
    main()