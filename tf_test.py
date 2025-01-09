# encoding=utf-8

from nlp_tfidf import Tfidf

if __name__ == '__main__':
    # 示例
    base_data = [
         "这是一个测试文本",
        "这是另一个测试文本",
        "这是一个完全不同的文本",
        "测试文本示例"
    ]

    Tfidf = Tfidf()
    res = Tfidf.quickRun(originQuestions=base_data, matchQuestion="这是另一个测试文本")
    print(res)