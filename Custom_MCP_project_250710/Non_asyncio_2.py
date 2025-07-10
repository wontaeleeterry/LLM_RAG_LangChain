from time import sleep  # time 모듈에서 sleep 함수만 임포트
import time

def add_a(a, b):
    sleep(1)  # I/O 작업 시뮬레이션
    return a + b

def add_b(a, b):
    sleep(2)  # 더 긴 I/O 작업 시뮬레이션
    return a + b

def main():
    start = time.time()  # time 모듈의 time() 함수 호출
    result_a = add_a(1, 2)  # 순차적 실행
    result_b = add_b(1, 2)  # 순차적 실행
    print("동기 실행 결과:", result_a, result_b)  # 결과값 출력
    print("동기 실행 시간:", time.time() - start)

main()
