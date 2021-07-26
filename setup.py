# -*- coding: UTF-8 -*-
import os
 
import setuptools
 
setuptools.setup(
    name='nlp-tf-idf',
    version='v1.0',
    keywords='nlp-qa',
    description='',
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.rst'
        )
    ).read(),
    author='yino',      # 替换为你的Pypi官网账户名
    author_email='m15829090357@163.com',  # 替换为你Pypi账户名绑定的邮箱
 
    url='https://github.com/yino/nlp-tf-idf',   # 这个地方为github项目地址，貌似非必须
    packages=setuptools.find_packages(),
    license='MIT'
)
