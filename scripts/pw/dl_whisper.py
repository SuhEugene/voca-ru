import os

from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
path = snapshot_download("deepdml/faster-whisper-large-v3-turbo-ct2",
                         local_dir="/workspace/voca/.cache/models/whisper-large-v3-turbo")
print("whisper at", path)
