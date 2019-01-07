# tf
1111
# 我生气



import tensorflow as tf
import  os


隐藏提示警告




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
定义函数
学习率和步数的设置

def myregression():
    '''
    
    自实现一个线性回归预测
    :return: None
    '''
    
    #建立作用域
    with tf.variable_scope("data"):
        #1.准备数据.x 特征值 [100,10] y 目标值[100]
        x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name='x_data')
        #矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8
    with tf.variable_scope("model"):
        #2.建立线性回归模型 1个特征,1个权重.1个偏置 y=x*w +b
        #随机给一个权重和偏置的值,让他去计算损失,然后优化,然后在当前状态下优化
        #用变量定义才能优化
        #只有一个数据而且是二维的 写法为[1,1] 一行一列
        #trainable参数:指定这个变量能跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1,1],mean=0.0,stddev=1.0),name='w',trainable=True)
        bias = tf.Variable(0.0,name='b')

        #权重与偏置相乘然后再添加
        y_redict = tf.matmul(x,weight)+bias

    with tf.variable_scope("loss"):
        #3.建立损失函数,均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_redict))

    with tf.variable_scope("model"):
        #4.梯度下降优化损失 learning_rate:0 ~ 1 , 2 , 3
        #学习率如果低 步数就要设置的高
        '''
        在极端情况下,权重的值变得非常大,以至于溢出,导致NaN值
        在深度神经网络(如RNN)当中更容易出现
        解决:
            1.重新设计网络
            2.调整学习率
            3.使用梯度截断(在训练过程中检查和限制梯度的大小)
            4.使用激活函数
        '''

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        #定义一个初始化变量的op
        init_op = tf.global_variables_initializer()

    #通过会话运行程序
    with tf.Session() as sess:
        #初始化变量
        sess.run(init_op)
        #打印随机最先初始化的权重和偏置
        print("随机初始化的参数权重为: %f,偏置为: %f" %(weight.eval(),bias.eval()))
        #在后台显示,建立事件未见
        tf.summary.FileWriter("./tmp/summary/test/",graph=sess.graph)
        #循环训练 运行优化
        for i in  range(200):
         sess.run(train_op)
         print("第%d次参数权重为: %f,偏置为: %f" % (i,weight.eval(), bias.eval()))

    return None

if __name__ == "__main__":
    myregression()
