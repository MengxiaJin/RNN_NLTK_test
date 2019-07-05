import tensorflow as tf
from rnn_models import TrainModel
import dataset
import setting
import multiprocessing
TRAIN_TIMES = 30000  # 迭代总次数（没有计算epoch）
SHOW_STEP = 1  # 显示loss频率
SAVE_STEP = 100  # 保存模型参数频率

x_data = tf.placeholder(tf.int32, [setting.BATCH_SIZE, None])  # 输入数据
y_data = tf.placeholder(tf.int32, [setting.BATCH_SIZE, None])  # 标签
emb_keep = tf.placeholder(tf.float32)  # embedding层dropout保留率
rnn_keep = tf.placeholder(tf.float32)  # lstm层dropout保留率

data = dataset.Dataset(setting.BATCH_SIZE)  # 创建数据集

model = TrainModel(x_data, y_data, emb_keep, rnn_keep)  # 创建训练模型

saver = tf.train.Saver()
def xunlian(q,sess,step):
    # 获取训练batch
    x, y = data.next_batch()
    # 计算loss
    loss, _ = sess.run([model.loss, model.optimize],
                       {model.data: x, model.labels: y, model.emb_keep: setting.EMB_KEEP,
                        model.rnn_keep: setting.RNN_KEEP})
    #if step % SHOW_STEP == 0:
     #   print ('step {}, loss is {}'.format(step, loss))
        # 保存模型
    if step % SAVE_STEP == 0:
        saver.save(sess, setting.CKPT1_PATH, global_step=model.global_step)
    q.put(step)
def main():
    # 4，创建进程池
    po = multiprocessing.Pool(5)
    # 5,创建一个队列
    q = multiprocessing.Manager().Queue()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化
        for step in range(TRAIN_TIMES):
            po.apply_async(xunlian,args=(q,sess,step))
    po.close()
    copy_ok_num = 0
    while True:
        file_name = q.get()
        # print("已经完成copy:%s"%file_name)
        copy_ok_num += 1
        print("\r训练的进度为：%.3f %%" % (copy_ok_num * 100 / TRAIN_TIMES), end="")
        if copy_ok_num >= TRAIN_TIMES:
            break
    # po.join()
    print()
if __name__ == '__main__':
    main()
