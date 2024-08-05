# encoding=utf-8

from nlp_tfidf import Tfidf

if __name__ == '__main__':
    # 示例
    base_data = [
        "教职工离职办理",
        "教师如何办理离职手续",
        "教师离职手续怎么办？",
        "教职工的离职手续办理？",
        "教职工办理离职手续？",
        "教职工怎么办理离职？",
        "学生怎么处理教职工关系。。。"
    ]

    Tfidf = Tfidf()
    Tfidf.save_model(question_list=base_data, answer_list=base_data)
    res = Tfidf.run(question="教职工离职")
    print(res)