# coding:utf8
import sys, os
import torch as t
from data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
#import ipdb
import warnings
warnings.filterwarnings('ignore')

class Config(object):
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 2
    batch_size = 128
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20  # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env = 'poetry'  # visdom env
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = None  # 预训练模型路径
    prefix_words = '细雨鱼儿出，微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/tang'  # 模型保存路径


opt = Config()


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：

    """
    
    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu: input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

#生成藏头诗
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    if opt.use_gpu: input = input.cuda()
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '<START>'

    if prefix_words: # prefix_words是控制生成的诗的风格
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}): #如果前一个单词属于这几种类型 则下面要生成藏头的word
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1) #直接将w作为输入的单词 这一步不能随意生成了
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    opt.device=t.device('cuda:0') if opt.use_gpu else t.device('cpu')
    device = opt.device
    #vis = Visualizer(env=opt.env)

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data) #[57580,125]
    dataloader = t.utils.data.DataLoader(data,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    # AverageMeter类用来管理一些变量的更新
    loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset() # 在每一个epoch 都要进行reset一遍

        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            #contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
            #一种可能的解释是：
            #有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
            # 也就是说使用contiguous()是为了能够使用view（）
            data_ = data_.long().transpose(1, 0).contiguous() #data_ shape:[seq_len,batch_size]
            data_ = data_.to(device)
            optimizer.zero_grad()
            #input_ shape:[124,128] target shape:[124,128]
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_) # ouput shape:[seq_len * batch_size,vocab_size] 此处seq_len为124
            loss = criterion(output, target.view(-1)) # target需要规整成[seq_len * batch_size]

            loss.backward()
            optimizer.step()

            # 更新loss_meter
            loss_meter.add(loss.item())

            '''
            
            # 可视化
            if (1 + ii) % opt.plot_every == 0:

                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 诗歌原文
                poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                           for _iii in range(data_.shape[1])][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')
            '''
        # 每一个epoch都打印下loss的值
        print('epoch:%d, loss:%.3f'%(epoch, loss_meter.value()[0]))

        #需要改进的地方是 得到模型在验证集的结果 根据结果只保存最好的模型
        t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))

    return ix2word,word2ix

def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """

    for k, v in kwargs.items():
        setattr(opt, k, v)

    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256);
    map_location = lambda s, l: s
    #state_dict = t.load(opt.model_prefix, map_location=map_location)
    state_dict=t.load('%s_%s.pth' % (opt.model_prefix, 19))
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))


if __name__ == '__main__':

    # step1: 训练模型
    ix2word,word2ix=train()

    # step2: 加载训练好的模型
    # data, word2ix, ix2word = get_data(opt)
    # data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()

    model = PoetryModel(len(word2ix), 128, 256)
    #opt.device = t.device('cuda:0') if opt.use_gpu else t.device('cpu')
    model.to(opt.device)
    model.load_state_dict(t.load('%s_%s.pth' % (opt.model_prefix, 19)))
    #利用训练好的模型生成藏头诗

    results=gen_acrostic(model,start_words='深度学习', ix2word=ix2word, word2ix=word2ix, prefix_words=None)
    print(' '.join(results))

    # 生成普通的诗
    gen()