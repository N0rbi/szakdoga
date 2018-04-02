# Szakdoga

My goal is to build an RNN that creates music lyrics based on the dataset.

## Runnable installation
```bash
pip install -r requirements.txt
```

### Train

To train use for example
```bash
python lyrics.py --artist=30y --epochs=250 --patience_limit=25 --lstm_layers=3 \
--lstm_units=64 --embedding=32 --size_x=100 --model_name=foobar
```
You can change any params, or you can just delete them (except for artist)
or:
```bash
python lyrics.py --help
```
for info.
For tensorboard use
```bash
tensorboard --logdir=target/train_log
```
from base folder (only works if you have at least 1 running/finished training).

### Predict

```bash
python demo.py --model_name=${name_of_the_model}
```
or if you keep it blank you get the list of models you can predict from.

## Latex

### Installation

```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### Compiling

```bash
./doc/gen_pdf.sh
```
use -s flag if you want it to be silent.

## Sources: 
- [RNN effectiveness (karpathy)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Difficulty of training RNNs](http://proceedings.mlr.press/v28/pascanu13.pdf)
- [Visualizing and understanding RNNs](https://arxiv.org/pdf/1506.02078.pdf)
- [AI written movie](https://arstechnica.com/gaming/2016/06/an-ai-wrote-this-movie-and-its-strangely-moving/)
- **LSTM**
  -  [paper about LSTM](http://www.bioinf.jku.at/publications/older/2604.pdf)
  -  [colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  -  [LSTM & diagrams](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714)
  -  [LSTM hard math](https://arxiv.org/pdf/1503.04069.pdf)
  -  [Deep sparse rectifier](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
  -  [Thesis on LSTM hyphenation (TDK, BME)](http://tdk.bme.hu/VIK/DownloadPaper/Szotagolas-mely-neuralis-halozatokkal1)
- **Multitask learning**
  - [Multi task learning](http://ruder.io/multi-task/)
  - [Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/abs/1604.05529)
  - [Semi-supervised Multitask Learning for Sequence Labeling](https://arxiv.org/abs/1704.07156)

## Frameworks
- [Dynet](https://github.com/clab/dynet)
- [Pytorch](https://github.com/pytorch/pytorch)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)

## Word based encoding
- [fastText](https://fasttext.cc/)
- [fastText genism wrapper](https://radimrehurek.com/gensim/models/wrappers/fasttext.html)
- [tokenization](http://www.nltk.org/api/nltk.tokenize.html)
- [word embeddings (and motivation behind it)](https://www.youtube.com/watch?v=5PL0TmQhItY)

## Heuristics
- [Bias-variance tradeoff (?)](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [perplexity](https://en.wikipedia.org/wiki/Perplexity)

## Environment
- [Anaconda](https://anaconda.org/)

#### [Deep learning A-Z](https://www.superdatascience.com/deep-learning/)
#### [Deep learning Goodfellow](http://www.deeplearningbook.org/)
#### [Software 2.0 - (karpathy)](https://medium.com/@karpathy/software-2-0-a64152b37c35)
#### [Intro to ANN in tensorflow](https://www.youtube.com/watch?v=BhpvH5DuVu8&index=3&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP)
