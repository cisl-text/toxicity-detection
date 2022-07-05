#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Perspective.py
@Contact :   1720613556@qq.com
@License :   (C)Copyright 2021-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/27 11:31   dst      1.0         None
'''
from perspective import PerspectiveAPI
API_KEY = "AIzaSyA2dVIbryh2AOqGtNWvne-za_9pUV-Wy7o"
p = PerspectiveAPI(API_KEY)
result = p.score("This is a comment")
print("Toxicity score: ", result["TOXICITY"])