#-*- encoding:utf-8 -*-
import jieba
from gensim import corpora, models, similarities
import heapq
import os
import json
import pandas as pd
import operator

class Qa:
    stopwords_file = './stopwordList/stopword.txt'
    stopWordsList = ''
    questionList = [
            "教职工离职办理",
            "教师如何办理离职手续",
            "教师离职手续怎么办？",
            "教职工的离职手续办理？",
            "教职工办理离职手续？",
            "教职工怎么办理离职？",
            "智能客服的来源",
            "什么是智能客服？",
            "智能客服是啥？",
        ]

    # load stop_words
    def load_stop_words(self)->list:
        with open(self.stopwords_file, mode="r", encoding="utf-8") as f:
            content = f.read()
        content_list = content.split('\n')
        self.stopWordsList = content_list
        return self.stopWordsList

    #delete question list stopwords 删除词组列表中的停用词
    def delete_stop_words(self, wordsList:list) -> list:
        """
        :param wordsList: 问题列表 [question1, question2, question3, question4,...]
        :return: list
        """
        newWords = []
        for word in wordsList:
            if word not in self.stopWordsList:
                newWords.append(word)
        return newWords

    #question list
    def get_question_list(self):
        """
        可从数据库文本等 获取
        :return: list
        """
        questionList = self.questionList

        #将question list 分词并去除停用词
        result = [self.delete_stop_words(jieba.lcut(val)) for val in questionList]
        return result

    #run
    def run(self, question: str) -> list:

        #1. 加载语料
        #load stop words
        self.load_stop_words()
        #get question list
        questionList = self.get_question_list()

        #delete stop words for input question
        question = self.delete_stop_words(wordsList=jieba.lcut(question))

        #2. 生成词典
        # 生成gensim 词典
        dictionary = corpora.Dictionary(questionList)

        #3. 通过doc2bow 稀疏向量生成语料库
        corpus = [dictionary.doc2bow(item) for item in questionList]
        #4. 计算tf值
        tf = models.TfidfModel(corpus)
        # 5.通过token2id得到特征数（特征数：字典里面的键的个数）
        #dictionary.token2id: {'title': id}
        numFeatures = len(dictionary.token2id.keys())
        #计算稀疏矩阵相似度 建立索引
        index = similarities.MatrixSimilarity(tf[corpus], num_features=numFeatures)

        #生成新的稀疏向量  根据原有的dictionary 生成新的 稀疏向量
        newDec = dictionary.doc2bow(question)
        # result
        simsQuestion = index[tf[newDec]]

        #
        result = []
        for val in list(enumerate(simsQuestion)):
            if val[1] > 0:
                result.append({
                    'question': str(self.questionList[val[0]]),
                    'sims': val[1],
                    'index': val[0],
                })
        print(result)
        exit()

if __name__ == '__main__':
    Qa = Qa()
    Qa.run('智能客服是什么')