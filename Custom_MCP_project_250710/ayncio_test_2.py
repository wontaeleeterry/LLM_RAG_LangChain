import asyncio
from time import time

async def add_a(a, b):
    await asyncio.sleep(1)  # simulate I/O
    return a + b

async def add_b(a, b):
    await asyncio.sleep(2)  # longer I/O
    return a + b

async def main():
    start = time()
    await asyncio.gather(add_a(1, 2), add_b(1, 2))  # 비동기 병렬 실행
    print("Async 실행 시간:", time() - start)

asyncio.run(main())
