# -*- coding: utf-8 -*-

#!/usr/bin/env python

import json
import aiohttp
import asyncio
import async_timeout
import threading
import os

'''async def fetch1(session, url):
    async with session.get(url) as response:
        return await response.text()


async def main(name_list):
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.ensure_future(
            fetch1(session, f'http://127.0.0.1:5000/{name}')) for name in name_list]
        res = await asyncio.gather(*tasks)
        print(res)'''

def get_tid_str():  
    tid = threading.current_thread().ident
    tid_name=threading.current_thread().name
    pid = os.getpid()
    #pid_name = psutil.Process(pid).name()
    res = 'tid:{},tid_name={},pid:{}'.format(tid,tid_name,pid)
    return res
        
def get_loop():
    try:
        loop = asyncio.get_event_loop()
        #print("get_event_loop")
    except RuntimeError as er:
        #print("new_event_loop")
        print(er.args[0], 'create a new EventLoop')
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
        
async def get_from(url,timeout):
    try:
        async with aiohttp.ClientSession() as session, async_timeout.timeout(timeout):
            #async with aiohttp.request('GET', url) as resp:
            async with session.get(url) as resp:
                assert resp.status == 200
                p = await resp.text()
                print(p)
                return p
    except asyncio.TimeoutError:
        return None

async def post_to(url,headers,body,timeout):
    try:
        async with aiohttp.ClientSession(headers=headers) as session, async_timeout.timeout(timeout):
            async with session.post(url ,data=json.dumps(body)) as resp:
                #assert resp.status == 200
                #p = await resp.text()
                #p = await resp.json()
                p = await resp.read()
                p=json.loads(p)
                #print("post status={} res_json={}".format(resp.status,p))
                return resp.status,p
    except asyncio.TimeoutError:
        return 1001,None

async def post_to_v2(url,headers,body,timeout):
    try:
        async with aiohttp.ClientSession() as session, async_timeout.timeout(timeout):
            async with session.post(url ,data=body) as resp:
                p = await resp.read()
                p=json.loads(p)
                return resp.status,p
    except asyncio.TimeoutError:
        return 1001,None

async def post_to_fds(task_id,fds_client,image_binary_list,timeout):
    try:
        async with async_timeout.timeout(timeout):
            url_list = fds_client.put_photos(task_id, image_binary_list)
            return 200,url_list
    except asyncio.TimeoutError:
        return 1001,None
