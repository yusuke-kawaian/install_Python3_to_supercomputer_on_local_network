# ローカルネットワーク内のスパコンへのPython3のインストールとtensorflowの導入  
本試行では表題の通りローカルネットワーク内のスパコンへPython3をインストールし, tensorflowを導入して機械学習の訓練環境を構築した.  

# Environment
```
# cat /etc/redhat-release
CentOS release 6.3 (Final)
```

# Install Python3.8.3
参考にしたページ[[1]](https://qiita.com/hasegit/items/6228da974c5837a1c393)に従い, 別のPCにDLしたPython-3.8.3.tar.xzをスパコンに転送し, /usr/local/src内でtarコマンドで展開した.  
```
# tar Python-3.8.3.tar.xz  
# cd Python-3.8.3  
# ./configure --with-ensurepip
# make 
# make test
# make install
```
`make test` で7つのエラーが出たが解決法が分からなかったので今回は無視した. (インストール, tesnorflowの展開共に成功したため, とりあえず保留. ~~詳細はmake_test_log.txt参照~~)  
```
7 tests failed:
    test_dtrace test_gdb test_normalization test_pdb test_robotparser
    test_urllib2net test_urllibnet
```
`make install`でインストールは正常に完了した. 特に指定した覚えはないが, `python3`, `pip3`コマンドがそれぞれ自動で使用可能になった.  


# Install Libraries
今回はtensorflowを用いて書いたコードを実行することがゴールなので, numpy, pandas, tensorflow等をインストールした.  
ローカルネットワークなのでもちろんpip3 installは使えない. そこで, webページ[[2]](https://tech-diary.net/python-library-offline-install/)を参考に.whlファイルもしくはtar.xzファイルを https://pypi.org/ からDLしてインストールした.  

.whlファイルをDLするページを見ると様々なバージョンが展開されている. そこで, Python3を会話形式で実行し, スパコンが対応しているバージョンを以下のコマンドで調べた[[3]](https://qiita.com/r-ngtm/items/f7dc53bf713d12f6be9d).  
```
# python3
>>> from pip._internal.pep425tags import get_supported
>>> get_supported()

[('cp38', 'cp38', 'manylinux2010_x86_64'), ('cp38', 'cp38', 'manylinux1_x86_64'), ('cp38', 'cp38', 'linux_x86_64'), ('cp38', 'abi3', 'manylinux2010_x86_64'), ('cp38', 'abi3', 'manylinux1_x86_64'), ('cp38', 'abi3', 'linux_x86_64'), ('cp38', 'none', 'manylinux2010_x86_64'), ('cp38', 'none', 'manylinux1_x86_64'), ('cp38', 'none', 'linux_x86_64'), ('cp37', 'abi3', 'manylinux2010_x86_64'), ('cp37', 'abi3', 'manylinux1_x86_64'), ('cp37', 'abi3', 'linux_x86_64'), ('cp36', 'abi3', 'manylinux2010_x86_64'), ('cp36', 'abi3', 'manylinux1_x86_64'), ('cp36', 'abi3', 'linux_x86_64'), ('cp35', 'abi3', 'manylinux2010_x86_64'), ('cp35', 'abi3', 'manylinux1_x86_64'), ('cp35', 'abi3', 'linux_x86_64'), ('cp34', 'abi3', 'manylinux2010_x86_64'), ('cp34', 'abi3', 'manylinux1_x86_64'), ('cp34', 'abi3', 'linux_x86_64'), ('cp33', 'abi3', 'manylinux2010_x86_64'), ('cp33', 'abi3', 'manylinux1_x86_64'), ('cp33', 'abi3', 'linux_x86_64'), ('cp32', 'abi3', 'manylinux2010_x86_64'), ('cp32', 'abi3', 'manylinux1_x86_64'), ('cp32', 'abi3', 'linux_x86_64'), ('py3', 'none', 'manylinux2010_x86_64'), ('py3', 'none', 'manylinux1_x86_64'), ('py3', 'none', 'linux_x86_64'), ('cp38', 'none', 'any'), ('cp3', 'none', 'any'), ('py38', 'none', 'any'), ('py3', 'none', 'any'), ('py37', 'none', 'any'), ('py36', 'none', 'any'), ('py35', 'none', 'any'), ('py34', 'none', 'any'), ('py33', 'none', 'any'), ('py32', 'none', 'any'), ('py31', 'none', 'any'), ('py30', 'none', 'any')]
```
上記の実行結果からcp38, manylinux1_x86_64のファイルをDLすればいいことが分かった.  

とりあえずnumpy1.18.5, pandas1.0.4, tensorflow2.2.0の.whlファイルをDLしインストール.  
```
# pip3 install --no-deps whlファイル名
```
後述のインストールで使用することになったが, .whlファイルが無くtar.xzファイルを用いてインストールする場合は, 以下のコマンドを用いる.  
```
# pip3 install tar.xzファイル名
```

インストール後, 早速NNの訓練プログラムを投入するもimport pandas/tensorflowでエラー. **上記の方法によるローカルでのインストールはpip3とは違い関連ライブラリのインストールをしてくれないのである.**  
調べてもなかなか出てこなかったので, 以下に地道に調べて分かったpandasとtensorflowの関連ライブラリを記載する.  
```
required libraries for introduction of pandas
  pandas1.0.4
  python_dateutil-2.8.1
  pytz-2020.1
```
```
required libraries for introduction of tensorflow
  tensorflow2.2.0
  absl-py-0.9.0
  astunparse-1.6.3
  gast-0.3.3
  google_api_python_client-1.9.1
  google-2.0.3
  opt_einsum-3.2.1
  protobuf-3.12.2
  six-1.15.0
  termcolor-1.1.0
  wrapt-1.12.1
```
ちなみに調べたら出てきたkerasの関連ライブラリは以下の通り.  
```
required libraries for introduction of keras
  absl
  astor
  BeautifulSoup4
  gast
  google
  Keras-Application
  Keras-Preprocessing
  Protobuf
  PyYAML
  scipy
  tensorflow
  termcolor
  wrapt
```

matplotlibを導入した際は以下の関連ライブラリを要求された. (追記 2020/06/29)
```
required libraries for introduction of matplotlib
  matplotlib-3.2.2
  pyparsing-2.4.7
  kiwisolver-1.2.0
  cycler-0.10.0.
```

# Carrying Out Tensorflow  
環境構築が出来たところでNNの訓練プログラムを再び走らせると, 以下のエラーが出現した.  
```
module 'tensorflow' has no attribute 'placeholder'
```
調べてみると[[4]](https://ja.stackoverflow.com/questions/59745/module-tensorflow-has-no-attribute-placeholder), tensorflowの文法にはv1とv2の2種類があるらしく, 自分の書いたプログラムはv1記法であるのにv2記法として読み込んでいたためであった. よく考えたら`import tensorflow as tf`時に以下の警告が出ていたなと...
```
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:root:Limited tf.compat.v2.summary API due to missing TensorBoard installation.
WARNING:tensorflow:From /usr/local/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
```
そこで, ここでもwebページを参考に以下のコマンドを`import tensorflow as tf`と置き換えることで解決した.  
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

# Reference
[1] https://qiita.com/hasegit/items/6228da974c5837a1c393  
[2] https://tech-diary.net/python-library-offline-install/  
[3] https://qiita.com/r-ngtm/items/f7dc53bf713d12f6be9d  
[4] https://ja.stackoverflow.com/questions/59745/module-tensorflow-has-no-attribute-placeholder  
  
  


