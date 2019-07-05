import tensorflow as tf
import numpy as np
from rnn_models import EvalModel
import os
from nltk import word_tokenize

# 指定验证时不使用cuda，这样可以在用gpu训练的同时，使用cpu进行验证
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

x_data = tf.placeholder(tf.int32, [1, None])

emb_keep = tf.placeholder(tf.float32)

rnn_keep = tf.placeholder(tf.float32)

# 验证用模型
model = EvalModel(x_data, emb_keep, rnn_keep)

saver = tf.train.Saver()
# 读取文本
text=open('vocab/poetry.vocab', "r", encoding='GBK').read()
#转列表
tokens = word_tokenize(text)
#生成单词到id的映射
word2id_dict=dict(zip(tokens,range(len(tokens))))
#生成id到单词的映射
id2word_dict=dict(zip(range(len(tokens)),tokens))



def generate_word(prob):
    """
    选择概率最高的前100个词，并用轮盘赌法选取最终结果
    :param prob: 概率向量
    :return: 生成的词
    """
    prob = sorted(prob, reverse=True)[:100]
    index = np.searchsorted(np.cumsum(prob), np.random.rand(1) * np.sum(prob))
    return id2word_dict[int(index)]

def generate_poem():
    """
    随机生成一首诗歌
    :return:
    """
    with tf.Session() as sess:
        # 加载最新的模型
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 预测第一个词
        rnn_state = sess.run(model.cell.zero_state(1, tf.float32))
        x = np.array([[word2id_dict['s']]], np.int32)
        prob, rnn_state = sess.run([model.prob, model.last_state],
                                   {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                    model.rnn_keep: 1.0})
        word = generate_word(prob)
        poem = ''
        # 循环操作，直到预测出结束符号‘e’
        while word != 'e':
            poem += word
            x = np.array([[word2id_dict[word]]])
            prob, rnn_state = sess.run([model.prob, model.last_state],
                                       {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                        model.rnn_keep: 1.0})
            word = generate_word(prob)
        # 打印生成的诗歌
        print (poem)


if __name__ == '__main__':
    generate_poem()
