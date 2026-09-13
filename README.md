# logicNN

論理ゲートニューラルネットワークの学習、評価および回路出力を行うプロジェクトです。

詳細ドキュメントは`website`にあります。

## 実行

利用するPython環境を有効化し、プロジェクトルートで実行します。

```shell
python tools/download_dataset.py --dataset mnist --root data
python main.py --config config/mnist_lgn.json
```
