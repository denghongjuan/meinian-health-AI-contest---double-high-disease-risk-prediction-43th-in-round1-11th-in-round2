# meinian-health-AI-contest---double-high-disease-risk-prediction-43th-in-round1-11th-in-round2
阿里云天池算法大赛 美年健康AI大赛—双高疾病风险预测解决方案，初赛43，复赛11

由于是在最后一两个星期才开始处理特征，初赛的特征清洗做的比较粗糙，本次开源的代码是在的初赛的代码上加入了部分复赛特征挖掘代码；
初赛的最终得分是0.0287，这次整理的代码交叉验证的分数比之前的好很多，都是单模型预测未做模型融合，所以还是有不少提升空间的。

代码运行说明：

1、在天池官网下载下面四个文件放在data目录下，网址https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.42d6342dbjczKD&raceId=231654
    meinian_round1_data_part1_20180408.zip
    meinian_round1_data_part2_20180408.zip
    meinian_round1_test_b_20180505.csv
    meinian_round1_train_20180408.csv
    
2、解压两个zip压缩文件到当前目录；

3、运行./code/main.py
