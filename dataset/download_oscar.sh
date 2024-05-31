wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz -O data/oscar-1GB.jsonl.xz
xz -d data/oscar-1GB.jsonl.xz
head -10000 data/oscar-1GB.jsonl > data/oscar-en-10k.jsonl