"""
为 D-iHarmony4 子数据集创建 composite_images_train/ 和 composite_images_test/ 目录。

D-HCOCO 和 D-HFlickr 只有 composite_degraded_images/ 目录，
而 data/dataset.py 的路径推导需要 composite_images_train/test 目录名。
本脚本用硬链接从 composite_degraded_images/ 分出 train/test 子目录。

用法: python tools/setup_diharmony4_datasets.py
"""
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = [
    {
        "name": "D-HCOCO",
        "train_txt": "HCOCO_train.txt",
        "test_txt": "HCOCO_test.txt",
    },
    {
        "name": "D-HFlickr",
        "train_txt": "HFlickr_train.txt",
        "test_txt": "HFlickr_test.txt",
    },
    {
        "name": "D-Hday2night",
        "train_txt": "Hday2night_train.txt",
        "test_txt": "Hday2night_test.txt",
    },
]


def read_txt(filepath):
    """读取 txt 文件，返回去重后的文件名列表"""
    names = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def hardlink_files(src_dir, dst_dir, filenames):
    """从 src_dir 创建硬链接到 dst_dir，返回成功数量"""
    os.makedirs(dst_dir, exist_ok=True)
    linked = 0
    skipped = 0
    for fname in filenames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(src):
            skipped += 1
            continue
        if os.path.exists(dst):
            linked += 1
            continue
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            # NTFS hardlink failed, fallback to copy
            shutil.copy2(src, dst)
            linked += 1
    return linked, skipped


def main():
    for ds in DATASETS:
        ds_dir = os.path.join(PROJECT_ROOT, ds["name"])
        if not os.path.isdir(ds_dir):
            print(f"[SKIP] {ds['name']}: directory not found")
            continue

        src_dir = os.path.join(ds_dir, "composite_degraded_images")
        train_txt = os.path.join(ds_dir, ds["train_txt"])
        test_txt = os.path.join(ds_dir, ds["test_txt"])

        if not os.path.isdir(src_dir):
            print(f"[SKIP] {ds['name']}: no composite_degraded_images/")
            continue

        # Train
        dst_train = os.path.join(ds_dir, "composite_images_train")
        if os.path.isdir(dst_train) and len(os.listdir(dst_train)) > 0:
            n = len(os.listdir(dst_train))
            print(f"[OK]   {ds['name']}: composite_images_train/ already has {n} files")
        elif os.path.isfile(train_txt):
            names = read_txt(train_txt)
            linked, skipped = hardlink_files(src_dir, dst_train, names)
            print(f"[LINK] {ds['name']}: train {linked}/{len(names)} files (skipped {skipped})")
        else:
            print(f"[SKIP] {ds['name']}: no train txt file")

        # Test
        dst_test = os.path.join(ds_dir, "composite_images_test")
        if os.path.isdir(dst_test) and len(os.listdir(dst_test)) > 0:
            n = len(os.listdir(dst_test))
            print(f"[OK]   {ds['name']}: composite_images_test/ already has {n} files")
        elif os.path.isfile(test_txt):
            names = read_txt(test_txt)
            linked, skipped = hardlink_files(src_dir, dst_test, names)
            print(f"[LINK] {ds['name']}: test {linked}/{len(names)} files (skipped {skipped})")
        else:
            print(f"[SKIP] {ds['name']}: no test txt file")

    print("\nDone.")


if __name__ == "__main__":
    main()
