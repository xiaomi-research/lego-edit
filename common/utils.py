from io import BytesIO
from PIL import Image
import base64
import binascii
import hashlib
import redis
from time import time
import json
import aiohttp
import asyncio
import async_timeout
import threading
from fds import GalaxyFDSClient, GalaxyFDSClientException, FDSClientConfiguration
import common.aiohttp_util as aiohttp_util
from enum import Enum  

from common.agent_logger import AgentLogger
logger = AgentLogger().get_logger()

class ResultCode(Enum):
    SUCCESS = 200,
    FDS_TIME_OUT = 1001,
    

class ImageInfo:
    def __init__(self, url):
        self.imageUrl = url

def convert_image2binary(images):
    image_binary_list = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_binary = buffered.getvalue()
        image_binary_list.append(image_binary)

    return image_binary_list

def convert_image_to_base64(images):
    base64_list = []
    for image in images:
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        base64_list.append(image_base64)
    
    return base64_list

def convert_base64_to_image(base64_list):
    images = []

    for base64_string in base64_list:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        images.append(image)
    
    return images


class RedisUtils():
    def __init__(self,config :dict):
        self.sid = config["redis"]["sid"]
        self.password = config["redis"]["password"]
        self.host = config["redis"]["host"]
        self.port = config["redis"]["port"]
        self.redis_client = redis.StrictRedis(
            host = self.host,
            port = self.port,
            password = self.password)
    def set(self,key, value, ex=1):
        self.redis_client.set(key, value)
    def get(self,key):
        value = self.redis_client.get(key)
        return value



class FdsUtils():
    def __init__(self,config):
        access_key = config["fds"]["access_key"]
        access_secret = config["fds"]["access_secret"]
        endpoint =  config["fds"]["endpoint"]
        client_config = FDSClientConfiguration(
                endpoint = endpoint,
                enable_cdn_for_upload = False,
                enable_cdn_for_download = False,
        )
        self.fds_client  = GalaxyFDSClient(access_key,access_secret,config=client_config)
        self.bucket_name = config["fds"]["bucket_name"]
        self.base_uri = config["fds"]["base_uri"]
        self.looper = aiohttp_util.get_loop()
        self.timeout = config["fds"]["timeout"]
        self.image_name_tag = config["fds"]["image_name_tag"]
        self.image_path = config["fds"]["image_path"]
    
    async def post_to_fds(self,task_id,tag,image_list,timeout):
        try:
            async with async_timeout.timeout(timeout):
                url_list = self.put_photos(task_id, tag,image_list)
                return True,'success',url_list
        except Exception as e:
            logger.error(f'Post to fds error: {str(e)}')
            return False,str(e),None


    def put_photo(self, data, object_name):
        self.fds_client.put_object(self.bucket_name, object_name, data)

    def put_photos(self, task_id,tag, data_list):
        object_names = []
        binary_data = convert_image2binary(data_list)
        for i, data in enumerate(binary_data):
            object_name = self.image_path + "/" + self.image_name_tag + "_" + tag + "_" + task_id + str(i) + ".jpg"
            object_names.append(object_name)
            self.put_photo(data, object_name)

        url_list = self.get_urls(object_names)
        return url_list

    def get_urls(self, object_names):
        url_list = []

        for object_name in object_names:
            url = self.fds_client.generate_presigned_uri(self.base_uri, self.bucket_name, object_name, time() * 1000 + 604800*1000)
            url_list.append(url)

        return url_list

    def post_data(self, task_id, tag,image_list):
        fds_task = self.looper.create_task(self.post_to_fds(task_id, tag,image_list, self.timeout))
        self.looper.run_until_complete(fds_task)

        fds_task_status, fds_task_error_info, fds_task_result = fds_task.result()
        return fds_task_status, fds_task_error_info, fds_task_result