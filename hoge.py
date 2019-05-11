# coding: utf-8
import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []#多分つかわん。

    def forward(self, x):#行列サイズに関係なく全ての成分にかける。
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]#paramの１成分に一本の行列orベクトルを保管

    def forward(self, x):
        W, b = self.params#paramsからW,b(成分ごと)を取り出す
        out = np.dot(x, W) + b#行(出力行列(ベクトル本数),列(出力データ列(出力データ成分数))
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = np.random.randn(I, H)#行列
        b1 = np.random.randn(H)#ベクトル
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)#バイアス

        # レイヤの生成(線形,非線形,線形)
        #今回で言えば、シグモイドにかけるときにベクトルを増やす(レイヤーやその引数ベクトルの性質ごとに適切な次元を設定)
        #関数オブジェクト、配列の要素一つ一つがオブジェクトをがコンストラクタを代入したクラスを保持
        self.layers = [Affine(W1, b1),Sigmoid(),Affine(W2, b2)]#
        '''
        arg1=Affine(W1, b1)
        arg2=Sigmoid()
        arg3=Affine(W2, b2)
        self.layers =[arg1,arg2,arg3]
        '''
        # すべての重みをリストにまとめる
        #インスタンスのパラメータはここにまとまっているので、これらをtxtに吐けば再現可能
        self.params = []#配列内のクラスを一個一個実行
        for layer in self.layers:#配列に入れられたレイヤーを順番にlayerに代入
            self.params += layer.params#enqueue(レイヤごとのパラメータ)

    def predict(self, x):
        for layer in self.layers:#layerをdequeueして次の引数にする
            x = layer.forward(x)#層3(層3(層3(ベクトル))
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)#中間レイヤーサイズは中間に広げる次元数
s = model.predict(x)
print(s)
