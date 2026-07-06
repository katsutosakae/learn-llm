####################################################################################
# 入力データに基づく文字・IDマップを作成（文字単位のEmbedded Modelに対応）
####################################################################################
# 入力ファイルの取得
# f = open('input.txt', 'r', encoding='utf-8')
# text = f.read()
# f.close()
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# 入力ファイルから、使用する文字を取得
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)


# ASCII単位の辞書作成（IDと文字直接対応）
# stoi = {}
# for i, ch in enumerate(chars):
#     stoi[ch] = i
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("hello world"))
# print(decode(encode("hello world")))

####################################################################################
# torchを使用した実装
####################################################################################
import torch

####################################################################################
# データセットの定義
####################################################################################
# torch.float64等のデータ型を指定可能。今回は整数型
# テンソルは、スカラー、ベクトル、行列といった概念を一般化した「多次元の数値の配列」
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# 最初の90%を学習に、残りを評価に
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# 学習時の乱数設定
torch.manual_seed(1337)
# バッチサイズ：同時学習個数
batch_size = 4
# ブロックサイズ：学習の入力に使用する文字列長さ、予測用のコンテキストにあたる
block_size = 8

# そもそもLLMは次の文字を予測する学習をさせたい
# なので文章自体を全部読ませずに1個目から順に見せていき、その次の文字をそのまま正解データにすればいい
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

def get_batch(split):
    # 学習用か評価用のデータセット切り替え
    data = train_data if split == 'train' else val_data
    # データセットの中から開始位置をランダムに設定（ブロックサイズ分はケツを開ける）
    # バッチサイズの個数だけ始点を決める
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")