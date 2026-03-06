"""
테스트용 TCP 에코 서버

클라이언트가 보낸 메시지를 그대로 돌려줍니다 (Echo).
socket_mcp_server.py 와 함께 동작을 테스트할 때 사용하세요.

실행:
  python tcp_echo_server.py              # 기본 127.0.0.1:9000
  python tcp_echo_server.py 0.0.0.0 9000 # 호스트·포트 지정

테스트 흐름:
  1. 터미널 A: python echo_server.py
  2. Claude Desktop(또는 MCP 클라이언트)에서 socket_MCP.py 등록 후
     - connect("127.0.0.1", 9000)
     - send_message("Hello!")
     - receive_message()   → "Hello!" 가 돌아옴
"""

import socket
import threading
import sys
import signal
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

_server_sock: socket.socket | None = None
_shutdown_event = threading.Event()


# ──────────────────────────────────────────────
# 클라이언트 핸들러 (스레드)
# ──────────────────────────────────────────────
def handle_client(conn: socket.socket, addr: tuple) -> None:
    print(f"[+] 연결: {addr}")
    conn.settimeout(1.0)
    try:
        while not _shutdown_event.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue

            if not data:
                break  # 클라이언트가 연결을 끊음

            msg = data.decode("utf-8", errors="replace").rstrip("\r\n")
            timestamp = time.strftime("%H:%M:%S")
            print(f"  [{timestamp}] {addr} → 수신: {msg!r}")

            # 에코 응답
            response = f"[ECHO] {msg}\n"
            conn.sendall(response.encode("utf-8"))
            print(f"  [{timestamp}] {addr} ← 송신: {response.rstrip()!r}")

    except ConnectionResetError:
        print(f"[-] 강제 종료: {addr}")
    finally:
        conn.close()
        print(f"[-] 연결 해제: {addr}")


# ──────────────────────────────────────────────
# 메인 서버 루프
# ──────────────────────────────────────────────
def run_server() -> None:
    global _server_sock

    _server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _server_sock.bind((HOST, PORT))
    _server_sock.listen(5)
    _server_sock.settimeout(1.0)

    print(f"TCP 에코 서버 시작: {HOST}:{PORT}")
    print("Ctrl+C 로 종료\n")

    while not _shutdown_event.is_set():
        try:
            conn, addr = _server_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        t.start()

    _server_sock.close()
    print("\n서버가 종료되었습니다.")


# ──────────────────────────────────────────────
# 시그널 핸들러
# ──────────────────────────────────────────────
def _shutdown(signum, frame):
    print("\n종료 신호 수신 — 서버를 닫는 중...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

if __name__ == "__main__":
    run_server()
