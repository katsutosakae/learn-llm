# learn-llm
for learning LLM architecture

# Reference against each directory

1. https://www.youtube.com/watch?v=kCc8FmEb1nY
   1. https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=YJb0OXPwzvqg


# python memo

## PyTorch のインストール

```powershell
# CPU 版（一番かんたん）, numpyは無くても動くけど
py -m pip install numpy
py -m pip install torch

# GPU (CUDA 12.x) 版を使いたい場合
py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

インストール確認:

```powershell
py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## よく使うコマンド

```powershell
py -m pip list                 # インストール済みパッケージ一覧
py -m pip install <package>    # パッケージ追加
py -m pip freeze > requirements.txt   # 依存関係を書き出す
py -m pip install -r requirements.txt # 依存関係をまとめて入れる
```
