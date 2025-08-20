import logging
from common.singleton import Singleton
from logging.handlers import TimedRotatingFileHandler
import os
import sys
import time

@Singleton
class AgentLogger(object):

    def __init__(self):
        # 文件的命名
        # cur_path = os.path.dirname(os.path.realpath(log_file))
        log_path = os.path.join(os.getcwd(), 'logs')
        # 如果不存在这个logs文件夹，就自动创建一个
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.logname = os.path.join(log_path, 'Agent.log')
        # self.logname = os.path.join(log_path, 'lm_excutor.log')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # 日志输出格式
        self.formatter = logging.Formatter(
            '[%(asctime)s] - [%(filename)s->%(funcName)s line:%(lineno)d] - %(levelname)s: %(message)s')
        self.config_handler()

    def config_handler(self):
        fh = self.get_file_handler(self.logname)
        self.logger.addHandler(fh)

        ch = self.get_console_handler()
        self.logger.addHandler(ch)

        fh.close()
        ch.close()

    def get_file_handler(self, filename):  
        fh = TimedRotatingFileHandler(filename, when="H", interval=4, backupCount=42, encoding="utf-8")
        fh.setFormatter(self.formatter)
        return fh

    def get_console_handler(self):  
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(self.formatter)
        return ch
    
    def get_logger(self):
        return self.logger
