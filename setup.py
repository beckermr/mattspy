from distutils.core import setup
import os

setup (name = 'mattspy',
       version = '0.1.0',
       description = "Matt's python utils",
       author = 'Matthew R. Becker',
       author_email = 'becker.mr@gmail.com',
       license = "GPL",
       url = 'https://github.com/beckermr/mattspy',
       packages = ['mattspy','mattspy.plotting','mattspy.stats'])
