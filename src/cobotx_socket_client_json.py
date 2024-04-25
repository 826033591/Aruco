# -*- coding: utf-8 -*-
import socket
import json                                     # 导入json包
 
client = socket.socket()  
client.connect(('192.168.2.101', 9000))             # 设置连接的服务器的IP和端口

while True:
    input_data = input("Please enter data (in JSON format, for example: {'key': 'value'}): ")
    if input_data.lower() == 'exit':
        client.send(b'byebye')
        break   
    try:
        dict_data = json.loads(input_data)                    # json格式化，将 JSON 字符串解析为 Python 对象
        if isinstance(dict_data, dict):
            json_data = json.dumps(dict_data)           # json对dict进行格式化
            client.send(json_data.encode('utf-8'))      # 设置编码为utf-8并转换为字节流
            info = client.recv(1024).decode('utf-8')
            if info == "byebye":
                break
            else:
                print("接收到的内容：", info)
        else:
            print("输入的数据不是字典类型，无法发送。")
            
    except Exception as e:
        print("输入的数据不是字典类型，无法发送。")
        #print(f"发生异常：{e}")

