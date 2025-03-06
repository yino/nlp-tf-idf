import heapq
import json
import operator
import os
import warnings

warnings.filterwarnings('ignore')
import jieba
import pandas as pd
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess

from . import stop_word


class Tfidf:
    MAX_INDEX_NUM = 10  # 获取最多多少条相似度数据
    DIFFERENCE_SIMS_NUM = 0.1  # 最小的 相似值 >0.1

    stopWordsList = []
    stopwords_file = ""
    userdict_file = ""
    work_dir="", 
    work_file_prefix=""

    def __init__(self, stopwords_file="", work_dir="", work_file_prefix="",MAX_INDEX_NUM=10,DIFFERENCE_SIMS_NUM=0.1):
        work_file_prefix = str(work_file_prefix)
        work_dir = str(work_dir)
        self.stopwords_file = stopwords_file
        # load stop words
        self.load_stop_words()

        if (work_dir == '') or len(work_dir) == 0:
            work_dir='.'

        if (work_file_prefix == '') or len(work_file_prefix) == 0:
            work_file_prefix=''
        self.work_dir = work_dir
        self.work_file_prefix = work_file_prefix
        self.MAX_INDEX_NUM = MAX_INDEX_NUM
        self.DIFFERENCE_SIMS_NUM = DIFFERENCE_SIMS_NUM
        self.create_dir(dirpath=work_dir)
    # load stop_words
    def load_stop_words(self):
        if (self.stopwords_file == '') or len(self.stopwords_file) == 0:
            self.stopWordsList = stop_word.stop_word_arr
            return

        with open(self.stopwords_file, mode="r", encoding="utf-8") as f:
            content = f.read()

        content_list = content.split('\n')
        print(content_list)
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
        try:
            dirname, file = str(self.work_dir), str(self.work_file_prefix)
            print('%s/%s_answer.json' % (dirname, file))
            base_data = question_list
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
            # 通过doc2bow稀疏向量生成语料库
            corpus = [dictionary.doc2bow(item) for item in base_items]
            # 4.通过TF模型算法，计算出tf值
            # 生成model文件以便下次直接加载model
            tf = models.TfidfModel(corpus)
            tf.save('%s/%s.model' % (dirname, file,))
            # 5.通过token2id得到特征数（字典里面的键的个数）
            num_features = len(dictionary.token2id.keys())
            print("num_features",num_features)
            # 6.计算稀疏矩阵相似度，建立一个索引
            index = similarities.MatrixSimilarity(tf[corpus], num_features=num_features)
            index.save('%s/%s.index' % (dirname, file,))
            return True
        except BaseException as e:
            print("save model fail", e)
            return False
        

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
            os.makedirs(dirpath)
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
        # self.load_stop_words()
        # cut words
        words_list = self.cut_words(question)
        # delete stop words
        question = self.delete_stop_words(words_list)
        file = '%s/%s' % ( dir, file)
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
        # 加载问题库和 答案
        answer_list = self.load_answer(file=file)
        question_list = self.load_question(file=file)
        # 获取最大的10个元素
        question_index_list = heapq.nlargest(10, range(len(sims)), sims.take)
        # 获取相似度
        answer_ret_list = []
        question_sims_list = []
        question_ret_list= []
        # question_sims_list = [sims[index] for index in question_index_list]
        for index in question_index_list:
            question_sims_list.append(sims[index])
            answer_ret_list.append(answer_list[index])
            question_ret_list.append(question_list[index])
        del question_index_list
        data = {
            'answer_list': answer_ret_list,
            'sims': question_sims_list,
            'question_list': question_ret_list 
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
                    'sims': float(val['sims']),
                    'question': val['question_list']
                })

        return res_data

    # 快速比对文本
    def quickRun(self, originQuestions = [], matchQuestion = ""):
        if len(originQuestions) == 0 or len(matchQuestion) == 0:
            return []

        # 预处理文本
        texts = [self.preprocess(text) for text in originQuestions]
        matchQueCorpu = self.preprocess(matchQuestion)

        # 构建词典和语料库
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # 使用TF-IDF模型
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        vec1 = tfidf[dictionary.doc2bow(matchQueCorpu)]

        # 计算余弦相似度
        index = similarities.MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))
        sims = index[vec1]

        # 对结果进行排序和过滤
        results = sorted([(i, sim) for i, sim in enumerate(sims)], key=lambda x: x[1], reverse=True)
        filtered_results = [result for result in results if result[1] > self.DIFFERENCE_SIMS_NUM]

        return [{"index": index, "sims": sim, "text": originQuestions[index]} for index, sim in filtered_results]

    # 预处理文本
    def preprocess(self, question):
        return self.delete_stop_words(self.cut_words(question))


if __name__ == '__main__':
    # 示例
    base_data = [
        "若X线片显示其远端骨折线与两髂嵴连线的夹角（Pauwells角）为60°，说明此骨折属于; 内收型骨折",
    ]

    Tfidf = Tfidf()
    # Tfidf.save_model(question_list=base_data, answer_list=base_data)
    # res = Tfidf.run(question=="教职工离职")

    res = Tfidf.quickRun(originQuestions = base_data, matchQuestion="该病人受伤的原因是")
    print(res)
