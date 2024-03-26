from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

# 假设你的文本文件名为'text.txt'，确保文本已经预处理并分词
input_file = 'novel.segment'

# 加载文本文件
sentences = LineSentence(input_file)

# 训练word2vec模型
model = Word2Vec(sentences, vector_size=200, window=5, min_count=4, workers=multiprocessing.cpu_count())

# 保存模型为bin文件
model.save("novel.bin")