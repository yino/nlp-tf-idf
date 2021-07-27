# -*- coding: UTF-8 -*-
from __future__ import print_function

import os

import setuptools

setuptools.setup(
    name='nlp_tfidf',
    version='1.0',
    keywords='nlp-qa',
    description='',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            './nlp_tfidf/README.md'
        ), encoding="utf8"
    ).read(),
    # long_description_content_type="text/markdown",
    author='yino',      # 替换为你的Pypi官网账户名
    author_email='m15829090357@163.com',  # 替换为你Pypi账户名绑定的邮箱
    url='https://github.com/yino/nlp-tf-idf',   # 这个地方为github项目地址，貌似非必须
    packages=setuptools.find_packages(),
    license='MIT',
)
