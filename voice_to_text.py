# -*- coding: utf-8 -*-
 
import wave
from pyaudio import PyAudio,paInt16
from openai import OpenAI

 
framerate=8000   #采样率
NUM_SAMPLES=2000 #采样点
channels=1  #一个声道
sampwidth=2 #两个字节十六位
TIME=2      #条件变量，可以设置定义录音的时间
 
def save_wave_file(filename, data):   #save the date to the wav file
    wf = wave.open(filename, 'wb')  #二进制写入模式
    wf.setnchannels(channels)  
    wf.setsampwidth(sampwidth)  #两个字节16位
    wf.setframerate(framerate)  #帧速率
    wf.writeframes(b"".join(data))  #把数据加进去，就会存到硬盘上去wf.writeframes(b"".join(data)) 
    wf.close()
 
def my_record():
    pa=PyAudio()
    stream=pa.open(format=paInt16,channels=1,rate=framerate,input=True,frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0  #
    while count < TIME*8: #循环2*20次
        string_audio_data=stream.read(NUM_SAMPLES) #每读完2000个采样加1
        my_buf.append(string_audio_data)
        count+=1
        print(u'Currently recording (simultaneously recording internal system and microphone sound)')
    save_wave_file('03.wav',my_buf) #文件保存
    stream.close()
    
    
def voice_to_text():
    client = OpenAI(api_key="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")

    audio_file= open("03.wav", "rb")
    transcription = client.audio.transcriptions.create(model="whisper-1",file=audio_file)
    return transcription.text
 
if __name__ == "__main__": 
    while True:
        a = input(u"Please press enter to start recording. The recording time is 4 seconds (press q to exit)：")
        if a == 'q':
            break
        my_record()
        print(u'Recording ended！')
        res = voice_to_text()
        print(u"Output:", res)