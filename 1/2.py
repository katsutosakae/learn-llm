####################################################################################
# 入力データに基づく文字・IDマップを作成（文字単位のEmbedded Modelに対応）
####################################################################################
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 入力ファイルから、使用する文字を取得
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)


# ASCII単位の辞書作成（IDと文字直接対応）
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

####################################################################################
# torchを使用した実装
####################################################################################
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

####################################################################################
# バッチサイズで分割した、文字列データ（問題）の設定
####################################################################################
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')


####################################################################################
# 簡易モデル（Embeddで直接全文字に対するスコアを取得しちゃう）
####################################################################################
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # 今回はEmbedded Vectorではなく、IDに対応する全文字に対するスコアを返す（しいて言うなら最後の生成確率Ot直前の値）
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # バッチサイズ分まとめてEmbded計算、Lossの設定を行う
    def forward(self, idx, targets=None):

        # idxを入力の(B, T)テンソル（文章サイズ×バッチサイズのまとまり群）として
        # 文章内の各文字（[b, t]で一意に決まる文字）に対応するEmbedしたスコアベクトル（全IDに対応するもの）を取得し配置
        # なので(B, T, C)となる、[b, t]で文字列内の各文字に対して付与された各IDに対するスコアを持つベクトルが取得できる
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Loss計算する際には、バッチサイズや文字列サイズを気にせずに、各文字に対する次の正解の文字を学習させているのかな
            # 2次元になる（バッチサイズ個の文字列ではなく、バッチサイズ×文字列長さ 個の文字に対応する、文字データ長さのスコア）
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # 各文字に対するベクトルと、その答えが対応する
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # バッチサイズ分まとめて新規文字を生成する
    def generate(self, idx, max_new_tokens):
        # idxを入力の(B, T)テンソル（文章サイズ×バッチサイズのまとまり群）として
        for _ in range(max_new_tokens):
            # そもそもlossは空
            logits, loss = self(idx)
            # バッチサイズ個の各問題（文字列）に対して、文字列最後の文字に対応するEmbdeddのベクトルだけ返す (B, C)
            logits = logits[:, -1, :] 
            # バッチサイズ個の各問題（文字列）の最後の文字に対応するベクトルに対して、softmaxで確立に変換する
            probs = F.softmax(logits, dim=-1) # (B, C)
            # バッチサイズ個の各問題（文字列）の最後の文字に対応する確率テンソルより、確率の高いものを抜粋して答えのテンソルにする
            # このテンソルはバッチサイズに一致　⇒　バッチ単位での生成
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 元のバッチサイズ全体の各文字列の最後に、決定した生成文字を付与
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)

logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

# 学習前の出力（入力はID 0、トークナイザを経由した体）
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

####################################################################################
# 誤差逆伝搬による調整
####################################################################################
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
step_size = 5000
step_eval_size = 1000
for steps in range(step_size): # increase number of steps for good results...

    # 関数呼び出しで、ランダムにバッチサイズごとの開始位置取得＋文字列サイズに基づく文字列群を取得
    # なので、ループごとにランダムな、バッチサイズ個ある文字列を取得
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # 重みの更新
    # 今回は self.token_embedding_table が対応（文字に対するスコアの値）
    optimizer.step()
    if steps % step_eval_size == 0:
        print(loss.item())

# 学習後の出力（入力はID 0、トークナイザを経由した体）
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))