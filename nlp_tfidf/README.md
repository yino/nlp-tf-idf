# nlp-qa-tfidf

#### 介绍
tfidf
余弦短文本求相似度

#### package
gensim
jieba
pandas


#### 安装教程

1.  pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt 

#### 使用说明


###### 实例化参数

```python
Tfidf = Tfidf(stopwords_file="", work_dir="", work_file_prefix="",MAX_INDEX_NUM=10,DIFFERENCE_SIMS_NUM=0.1)
#stopwords_file 停用词文件 可不填 包里自带有
#work_dir 工作目录 模型存储的路径，不填则在当前项目根目录下 建议填
#work_file_prefix 文件名字前缀， 不填则训练的模型名字为 _.model 可以填时间戳，或者用户id，建议填写 不然模型会相互覆盖
#MAX_INDEX_NUM 最大的数量，匹配到结果后返回的数量
#DIFFERENCE_SIMS_NUM 最小的相似值，以这个参数为标准，大于这个值的，全部判定为相似结果，然后根据MAX_INDEX_NUM 返回最大的 MAX_INDEX_NUM个答案
```
###### 模型训练

```python
# QA 场景
# 问题列表
question_list = [
    "test1-question","test2-question","test3-question"
]

# 答案列表
answer_list = [
    "test1-answer","test2-answer","test3-answer"
]


Tfidf = Tfidf(stopwords_file="", work_dir="", work_file_prefix="")
# 训练模型并保存
Tfidf.save_model(question_list=question_list,answer_list=answer_list)

# 输入问题，并去计算相似度
Tfidf.run(question="test question")
```
