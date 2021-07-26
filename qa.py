import heapq
import json
import operator
import os
import warnings

warnings.filterwarnings('ignore')
import jieba
import pandas as pd
from gensim import corpora, models, similarities


class Qa:
    MAX_INDEX_NUM = 10  # 获取最多多少条相似度数据
    DIFFERENCE_SIMS_NUM = 0.1  # 最小的 相似值 >0.1

    stopWordsList = []
    stopwords_file = ""
    userdict_file = ""
    work_dir="", 
    work_file_prefix=""

    def __init__(self, stopwords_file="", work_dir="", work_file_prefix=""):
        if (stopwords_file != '') or len(stopwords_file) == 0:
            self.stopwords_file = './stopwordList/stopword.txt'
            self.load_stop_words()
    
        if (work_dir != '') or len(work_dir) == 0:
            work_dir='.'

        if (work_file_prefix != '') or len(work_file_prefix) == 0:
            work_file_prefix=''
        self.work_dir = work_dir
        self.work_file_prefix = work_file_prefix
            
    # load stop_words
    def load_stop_words(self):
        with open(self.stopwords_file, mode="r", encoding="utf-8") as f:
            content = f.read()

        content_list = content.split('\n')
        self.stopWordsList = content_list

    # delete stop words
    def delete_stop_words(self, words_list):
        new_words = []
        for word in words_list:
            if word not in self.stopWordsList:
                new_words.append(word)
        return new_words

    # cut words
    def cut_words(self, text):
        return jieba.lcut(text)

    # save dictionary
    def save_model(self, question_list=[], answer_list=[], ):
        """
        :param questionList:
        :param file: 默认是公司id
        :return:
         base_data = [
            "教职工离职办理",
            "教师如何办理离职手续",
            "教师离职手续怎么办？",
            "教职工的离职手续办理？",
            "教职工办理离职手续？",
            "教职工怎么办理离职？",
        ]
        """
        dirname, file = str(self.work_dir), str(self.work_file_prefix)
        base_data = question_list
        # create dir
        # self.create_dir(dirpath='./dictionary_file/%s' % (dirname,))
        # save question id
        self.save_answer(answer_list=answer_list,
                              filepath='%s/%s_answer.json' % (dirname, file))
        # save question
        self.save_question(question_list=question_list,
                           filepath='%s/%s_question.json' % (dirname, file))
        # 1.将base_data中的数据进行遍历后分词

        base_items = []
        for item in base_data:

            # base_items.append(self.cut_words(item))
            # cut words and delete stop words
            base_items.append(self.delete_stop_words(self.cut_words(item)))
        # 2.生成词典
        dictionary = corpora.Dictionary(base_items)
        dictionary.save('%s/%s.dict' % (dirname, file,))
        # 获取 key 对应 key
        # for key in dictionary.iterkeys():
        #     print(dictionary[key], key)
        # exit()
        # 通过doc2bow稀疏向量生成语料库
        corpus = [dictionary.doc2bow(item) for item in base_items]

        # for item in base_items:
        #     print(dictionary.doc2bow(item)) [(0, 1), (1, 1), (2, 1)]
        #     print(type(dictionary.doc2bow(item))) list
        #     exit()
        # corpus = [dictionary.doc2bow(item) for item in base_items]
        # for item in base_items:
        #     print(dictionary.doc2bow(item))

        # 4.通过TF模型算法，计算出tf值
        # 生成model文件以便下次直接加载model
        tf = models.TfidfModel(corpus)
        tf.save('%s/%s.model' % (dirname, file,))
        # 5.通过token2id得到特征数（字典里面的键的个数）
        num_features = len(dictionary.token2id.keys())
        # 6.计算稀疏矩阵相似度，建立一个索引
        index = similarities.MatrixSimilarity(tf[corpus], num_features=num_features)
        index.save('%s/%s.index' % (dirname, file,))

    def save_answer(self, answer_list, filepath):
        answer_list = json.dumps(answer_list)
        with open(filepath, mode="w+", encoding="utf-8") as f:
            f.write(answer_list)

    def save_question(self, question_list, filepath):
        question_list = json.dumps(question_list)
        with open(filepath, mode="w+", encoding="utf-8") as f:
            f.write(question_list)

    # 加载语料库
    def load_dictionary(self, file):
        return corpora.Dictionary.load('%s.dict' % (file,))

    # 加载 tf-idf model
    def load_tf_model(self, file):
        return models.TfidfModel.load('%s.model' % (file,))

    # load index 加载索引
    def load_index(self, file):
        return similarities.MatrixSimilarity.load('%s.index' % (file,))

    # load answer json
    def load_answer(self, file):
        with open('%s_answer.json' % (file,), mode="r", encoding="utf-8")as f:
            data = f.read()
        data = json.loads(data)
        return data

    # load question json
    def load_question(self, file):
        with open('%s_question.json' % (file,), mode="r", encoding="utf-8")as f:
            data = f.read()
        data = json.loads(data)
        return data

    # create save dir
    def create_dir(self, dirpath):
        if os.path.isdir(dirpath) == False:
            os.mkdir(dirpath, 777)
    # run
    def run(self, question):
        """
        最终
        :param stopwords_file:
        :param userdict_file:
        :return:
        """
        dir = self.work_dir
        file = self.work_file_prefix
        # load stop words
        self.load_stop_words()
        # cut words
        words_list = self.cut_words(question)
        # delete stop words
        question = self.delete_stop_words(words_list)
        file = '%s/%s' % ( dir, file)
        print(file)
        # 获取词典
        dictionary = self.load_dictionary(file=file)
        
        # 获取模型
        tf = self.load_tf_model(file=file)
        
        # 获取索引文件
        index = self.load_index(file=file)
       
        # 新的稀疏向量
        new_vec = dictionary.doc2bow(question)

        # 9.算出相似度
        sims = index[tf[new_vec]]
        answer_list = self.load_answer(file=file)
        
        # 获取最大的10个元素
        question_index_list = heapq.nlargest(10, range(len(sims)), sims.take)
        # 获取相似度
        answer_ret_list = []
        question_sims_list = []
        # question_sims_list = [sims[index] for index in question_index_list]
        for index in question_index_list:
            question_sims_list.append(sims[index])
            answer_ret_list.append(answer_list[index])

        del question_index_list
        data = {
            'answer_list': answer_ret_list,
            'sims': question_sims_list
        }
        df = pd.DataFrame(data)

        # 剔除0
        df = df.query("sims>0")
        res_data = []
        df['sims_1'] = df['sims'].shift(1)
        df['sims_min'] = df['sims_1'] - df['sims']
        new_df = df.query("sims_1-sims>0.1")
        return_data = []
        
        if len(new_df) != 0:
           # 获取最后一行的行标
            last_row_index = new_df[-1:].index.tolist()[0]
            for index, val in df[0:last_row_index + 1].iterrows():
                res_data.append({
                    'answer': val['answer_list'],
                    'sims': float(val['sims'])
                })

        return res_data

if __name__ == '__main__':
    # Qa = Qa()
    # Qa.save_model(questionList=[], file="test")
    # exit()
    # print(Qa.check(question="交通费用", dir="8", file="28"))
    # exit()
    base_data = [
        "教职工离职办理",
        "教师如何办理离职手续",
        "教师离职手续怎么办？",
        "教职工的离职手续办理？",
        "教职工办理离职手续？",
        "教职工怎么办理离职？",
        "学生怎么处理教职工关系。。。"
    ]

    Qa = Qa()
    # Qa.save_model(question_list=base_data, answer_list=base_data)
    res = Qa.run(question="教职工离职")
    print(res)
