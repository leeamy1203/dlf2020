# dlf2020
The goal of the project is to use deep neural network to translate spoken language to American SignLanguage (ASL) in the form of continuous 3D skeletal poses. For a given English word, we will be outputting a stream of skeletal poses (x, y, and z coordinates) representing ASL signs for the upper body which includes the torso, arms, hands and facial key points

## Setup
Use virtual env (used python 3.8.6). Install dependencies with 
```
pip3 install -r requirements.txt
```
Add virtual env to jupyter notebook
```
python -m ipykernel install --user --name=your_virtual_env
jupyter notebook --port=your_port_number
```
Remember to activate environment to install libraries
```
pynev shell your_virtual_env
```

### Notebook setup
Leverage the following code snippet to use the src directory in notebooks
```python
import sys
# This should navigate to the repository root
sys.path.append('../')
%reload_ext autoreload
%autoreload 2
```

## Project Structure
Most idea pulled from [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure)

### app.py
Main command line interface to run each step of the ML pipeline
 - Create and save word labels and embeddings by running
   ```python 
   python app.py data
   ```
   This will create embeddings.pkl and words.pkl in the data/interim folder which contains 2000 entries of the WLASL words.
   See data/preprocessor.py for more details
  


### data
Where data lives
### notebook
Any jupyter notebook should be placed here
### metrics
Any output of metrics (MSE..etc) should be saved under here as a csv
### src
All code. Here's the breakdown per folder within src.
#### conf
Config file. Currently has logging set up
#### data
All code related to data preprocessing to create a trainable data 
#### train
All code related to modeling and traning
#### validate
All code related to validating the model 

## Pipeline

### Data Preprocessing
Utilize the [code](https://github.com/gopeith/SignLanguageProcessing) by "Words are Our Glosses" [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Zelinka_Neural_Sign_Language_Synthesis_Words_Are_Our_Glosses_WACV_2020_paper.pdf). 

Their code includes (3DposeEstimator/demo.py and wacv2020/pipeline_demo*.py):
 - openpose implementation of converting video to 2d pose estimations 
 - correction process for misplaced joints correction 
 - z-estimation to convert from 2d to 3d skeletons
 - normalization  

Some of these codes already live under the google drive due to the need for Colab. 

### Generator
Following "Progressive Transformer" [paper](https://arxiv.org/pdf/2004.14874.pdf) as it describes the Generator step in detail.

The Progressive Transformer is an extension of classic Transformer made popular by the "Attention is All you need" [paper](https://arxiv.org/abs/1706.03762)
This [article and linked youtube](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0) was very good explaining how Transformer works. 

Utilize one of these implemented transformer in pytorch and tweak it to meet our needs:
 - https://github.com/jadore801120/attention-is-all-you-need-pytorch 
 - http://nlp.seas.harvard.edu/2018/04/03/attention.html
 - https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

#### Transformer Tweaks - Progressive Transformer
Here's a breakdown of what needs to be tweaked from traditional transformer. 

##### Simplify Encoder Step
Because we aren't using sentences, the encoder step can be much simplified. There's no need for MHA and positional encoder since these layers 
try to understand the relationship of words within the sentences. 
 - Instead of words ->  MHA -> Linear Normalization -> Feed Forward (Linear + ReLU + Linear) -> Linear Normalization  
 - Implement word -> Feed Forward -> Linear Normalization 

##### Add Counter Embedding
Similar to a "period" to mark the end. The Progressive Transformer has a Counter that is also learned as part of the output. 
No mention of how this loss is computed but perhaps should be a straightforward distance calculation as well. 

#### Input tweaks - Embeddings
To deal with out-of-vocabulary word, we can use word or character embedding.
There are pretrained modesl we can find

##### Word embedding
 - https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
 - https://towardsdatascience.com/deep-learning-for-nlp-with-pytorch-and-torchtext-4f92d69052f
 - https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html
 - fasttext by facebook ai 
   - https://github.com/facebookresearch/fastText?utm_source=catalyzex.com
   - https://fasttext.cc/docs/en/english-vectors.html
##### Character embedding
While one of the pretrained word embedding encompass a lot of words, there can still be OOV problems. Character embedding can be used to get rid of
oov problems commpletely. Perhaps we should do a combination of word and character embedding
 - https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10
 - https://github.com/makcedward/nlp/blob/master/sample/nlp-character_embedding.ipynb

#### Computing Loss for Generator
Compute **MSE** after applying **DTW**. Utilize the code written by "Words are Our Glosses". See [wacv2020/modeling.py](https://github.com/gopeith/SignLanguageProcessing/blob/master/wacv2020/modeling.py)


### Discriminator
Conditioanl GAN from the "Adversarial Training for Multi-Channel SLP" [paper](https://arxiv.org/pdf/2008.12405.pdf)

### Back Translation
Sign Language Transformers: Joint End-to-end Sign Language. Recognition and Translation [paper](https://arxiv.org/abs/2003.13830) [code](https://www.catalyzex.com/redirect?url=https://github.com/neccam/slt)

### Data
Trainable data available: https://drive.google.com/file/d/1-6b7_Rsum_fHTN4kKEUeR5Y3bJ0z76jd/view





