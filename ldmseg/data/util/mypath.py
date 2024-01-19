"""
Author: Wouter Van Gansbeke

File with the root path to different datasets
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database='', prefix='/efs/datasets/'):

        db_names = {'coco', 'cityscapes'}
        assert (database in db_names), 'Database {} not available.'.format(database)

        return os.path.join(prefix, database)
