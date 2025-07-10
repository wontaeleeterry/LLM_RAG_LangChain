from time import time

def add_a(a, b):
    for i in range(300000):  # 반목문에 의한 임의의 시간 지연 생성
        print(i)


    print('add_a: {0} + {1}'.format(a, b))
    
    return a+b

def add_b(a, b):
    for i in range(600000):  # 반복문에 의한 시간 지연 생성 - add_a 보다 약간 길게
        print(i)

    print('add_b: {0} + {1}'.format(a, b))

    return a+b

def print_add(a, b):
    add_a(a, b)            
    result = add_b(a, b)   
    print('print_add_b: {0} + {1} = {2}'.format(a, b, result))


start = time()

# 임의의 반복문을 설정하여 시간 지연을 생성함
print_add(1, 2)

end = time()

print('실행 시간 : {0:.5f}초'.format(end-start))