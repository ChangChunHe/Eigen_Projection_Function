#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from distutils.core import setup

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.


setup(name = "eigprofuc",
    version = "0.0.1",
    description = "Get difference between any two structures",
    author = "XiaoTianLi and ChangChunHe",
    author_email = "731176792@qq.com",
    #Name the folder where your packages live:
    #(If you have other packages (dirs) or modules (py files) then
    #put them into the package directory - they will be found
    #recursively.)
    packages = ['eigprofuc'],
    install_requires=['numpy>=1.15.4','scipy>=1.1.0','matplotlib>=2.2.2','nose2']
)
