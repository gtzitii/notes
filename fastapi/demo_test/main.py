# 从 fastapi 包中导入 FastAPI 类
# FastAPI 是整个 Web 应用的核心对象
from fastapi import FastAPI

# 创建 FastAPI 应用实例
# app 就是 ASGI 应用对象，uvicorn 启动时会加载它
app = FastAPI()

# 定义一个 GET 接口
# 当客户端以 GET 方式访问 /test 路径时，会调用下面的函数
@app.get("/test")
def test():
    """
    测试接口
    访问路径: /test
    请求方式: GET
    用于验证 FastAPI 服务是否正常运行
    """
    # FastAPI 会自动把 Python 对象转换成 JSON 返回
    return {
        "message": "This is a test demo!"
    }