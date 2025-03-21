## 背景
lighteval仅仅支持openai的model调用，但是不支持普通restful api调用
而目前Astro One + RAG模式, 只能提供restful api接口，所以对lighteval进行
是培训代码改造
## 方法
1、继承LightevalModel，实现CustomAPIMLModel，重点重写**greedy_until**方法
Astro One + RAG模式仅仅只能支持流式输出，所以Post调用方法有点特殊，可以改写
**generate**方法支持不同Post请求
## 运行方式
``python model_api.py``
可以修改evaluate_api.py里面的配置，支持不同的天文任务




