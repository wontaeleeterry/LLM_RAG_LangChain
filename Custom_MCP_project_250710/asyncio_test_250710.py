import asyncio
from time import time

async def add_a(a, b):
    for i in range(300000):  # 반목문에 의한 임의의 시간 지연 생성
        print(i)


    print('add_a: {0} + {1}'.format(a, b))
    
    return a+b

async def add_b(a, b):
    for i in range(600000):  # 반복문에 의한 시간 지연 생성 - add_a 보다 약간 길게
        print(i)

    print('add_b: {0} + {1}'.format(a, b))

    return a+b

async def print_add(a, b):
    await add_a(a, b)            # await로 다른 네이티브 코루틴 실행
    result = await add_b(a, b)   # await로 다른 네이티브 코루틴 실행하고 반환 값을 변수에 저장
    print('print_add_b: {0} + {1} = {2}'.format(a, b, result))


start = time()

# 이벤트 루프를 얻는 과정까지 소모되는 시간을 초과하도록
# 임의의 반복문을 설정하여 시간 지연을 생성함

loop = asyncio.get_event_loop()           # 이벤트 루프를 획득
loop.run_until_complete(print_add(1, 2))  # print_add(1, 2)가 끝날 때까지 이벤트 루프를 실행
loop.close()                              # 이벤트 루프를 닫음

end = time()

print('실행 시간 : {0:.5f}초'.format(end-start))