# 修改后的 download_codebert.py
from huggingface_hub import snapshot_download
import os

# 强制使用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 避免符号链接警告

snapshot_download(
    repo_id="microsoft/codebert-base",
    local_dir="./codebert",
    local_dir_use_symlinks=False,
    force_download=False,         # 如果本地已有部分文件，不强制重下
    resume_download=True,         # 支持断点续传
    ignore_patterns=["*.msgpack", "*.h5"],
)

print("CodeBERT 下载完成！")
print("模型路径：", os.path.abspath("./codebert"))