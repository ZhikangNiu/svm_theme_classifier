# 设计方案 SVM 主题分类
## 抽帧
1. 均匀抽帧（当前策略）
2. 抽关键帧（待测试，会导致数据十分少）
## 特征提取
1. ResNet50 抽取layer4的特征
2. PCA降维（对于PCA降维的维数做了验证，一般来说10维就够了，但是考虑了20维）
## SVM的设计
1. 采用了sklearn的SVM
2. 使用了四种kernel：linear, poly, rbf, sigmoid选取最优策略
3. KFold交叉验证，选取最优参数
4. 每一类训练数：KFold * 4
## 评估
1. 采用了sklearn的classification.score()函数
2. 对每一类进行了测评
## TODO:
1. 代码使用了大量读写
2. 可以尝试使用GridSearchCV进行参数搜索
3. 加入log写入
4. 测试一下均匀抽帧的结果（抽帧的这个策略需要改写成多线程的，不然太慢了）