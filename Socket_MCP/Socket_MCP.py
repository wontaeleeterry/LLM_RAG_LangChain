"""
FastMCP Socket 통신 MCP 서버

기능:
  - send_message   : TCP 소켓으로 메시지 전송
  - receive_message: 소켓에서 메시지 수신
  - connect        : TCP 소켓 연결
  - disconnect     : TCP 소켓 연결 해제
  - connection_status: 현재 연결 상태 확인

실행:
  pip install fastmcp
  python socket_mcp_server.py
"""

import socket
import threading
from typing import Optional
from fastmcp import FastMCP

# ──────────────────────────────────────────────
# 전역 소켓 상태 (단일 연결 관리)
# ──────────────────────────────────────────────
_sock: Optional[socket.socket] = None
_lock = threading.Lock()
_received_buffer: list[str] = []   # 수신 메시지 버퍼
_receiver_thread: Optional[threading.Thread] = None
_running = False

# ──────────────────────────────────────────────
# MCP 앱 초기화
# ──────────────────────────────────────────────
mcp = FastMCP(
    name="socket-mcp",
    instructions=(
        "TCP 소켓 통신 도구입니다. "
        "connect → send_message / receive_message → disconnect 순서로 사용하세요."
    ),
)


# ──────────────────────────────────────────────
# 내부 헬퍼: 백그라운드 수신 루프
# ──────────────────────────────────────────────
def _recv_loop(sock: socket.socket) -> None:
    """소켓에서 데이터를 지속적으로 읽어 버퍼에 저장합니다."""
    global _running
    sock.settimeout(1.0)
    while _running:
        try:
            data = sock.recv(4096)
            if not data:
                break
            msg = data.decode("utf-8", errors="replace")
            with _lock:
                _received_buffer.append(msg)
        except socket.timeout:
            continue
        except OSError:
            break
    _running = False


# ──────────────────────────────────────────────
# Tool: connect
# ──────────────────────────────────────────────
@mcp.tool()
def connect(host: str, port: int) -> str:
    """
    지정한 호스트/포트에 TCP 연결을 맺습니다.

    Args:
        host: 접속할 호스트 (예: "127.0.0.1")
        port: 접속할 포트 번호 (예: 9000)

    Returns:
        연결 성공/실패 메시지
    """
    global _sock, _running, _receiver_thread, _received_buffer

    with _lock:
        if _sock is not None:
            return "이미 연결되어 있습니다. 먼저 disconnect를 호출하세요."

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        with _lock:
            _sock = sock
            _running = True
            _received_buffer = []

        _receiver_thread = threading.Thread(target=_recv_loop, args=(sock,), daemon=True)
        _receiver_thread.start()

        return f"{host}:{port} 연결 성공"
    except Exception as e:
        return f"연결 실패: {e}"


# ──────────────────────────────────────────────
# Tool: disconnect
# ──────────────────────────────────────────────
@mcp.tool()
def disconnect() -> str:
    """
    현재 TCP 연결을 해제합니다.

    Returns:
        연결 해제 결과 메시지
    """
    global _sock, _running

    with _lock:
        if _sock is None:
            return "현재 연결된 소켓이 없습니다."
        _running = False
        try:
            _sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        _sock.close()
        _sock = None

    if _receiver_thread and _receiver_thread.is_alive():
        _receiver_thread.join(timeout=2)

    return "🔌 연결이 해제되었습니다."


# ──────────────────────────────────────────────
# Tool: connection_status
# ──────────────────────────────────────────────
@mcp.tool()
def connection_status() -> str:
    """
    현재 소켓 연결 상태를 반환합니다.

    Returns:
        연결 상태 문자열
    """
    with _lock:
        if _sock is None:
            return "연결 없음"
        try:
            peer = _sock.getpeername()
            return f"연결 중 → {peer[0]}:{peer[1]}"
        except OSError:
            return "소켓 존재하나 피어 정보 없음"


# ──────────────────────────────────────────────
# Tool: send_message
# ──────────────────────────────────────────────
@mcp.tool()
def send_message(message: str, add_newline: bool = True) -> str:
    """
    현재 연결된 소켓으로 메시지를 전송합니다.

    Args:
        message   : 전송할 문자열
        add_newline: True이면 메시지 끝에 '\\n' 추가 (기본값 True)

    Returns:
        전송 성공/실패 메시지
    """
    with _lock:
        if _sock is None:
            return "연결된 소켓이 없습니다. 먼저 connect를 호출하세요."
        sock = _sock

    try:
        payload = message + ("\n" if add_newline else "")
        sock.sendall(payload.encode("utf-8"))
        return f"전송 완료 ({len(payload.encode())} bytes): {message!r}"
    except Exception as e:
        return f"전송 실패: {e}"


# ──────────────────────────────────────────────
# Tool: receive_message
# ──────────────────────────────────────────────
@mcp.tool()
def receive_message(timeout_sec: float = 3.0, clear_buffer: bool = True) -> str:
    """
    수신 버퍼에 쌓인 메시지를 반환합니다.
    버퍼가 비어 있으면 최대 timeout_sec 초 동안 대기합니다.

    Args:
        timeout_sec : 대기 최대 시간 (초, 기본 3.0)
        clear_buffer: True이면 반환 후 버퍼를 비움 (기본 True)

    Returns:
        수신된 메시지 목록 또는 타임아웃 메시지
    """
    import time

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with _lock:
            if _received_buffer:
                msgs = list(_received_buffer)
                if clear_buffer:
                    _received_buffer.clear()
                result = "\n".join(msgs)
                return f"수신된 메시지 ({len(msgs)}건):\n{result}"
        time.sleep(0.1)

    return f"{timeout_sec}초 동안 수신된 메시지 없음"

# ──────────────────────────────────────────────
# Tool: send_and_receive_message
# ──────────────────────────────────────────────

@mcp.tool()
def send_and_receive(message: str, timeout_sec: float = 3.0) -> str:
    """
    메시지를 전송하고 응답을 기다려 전송·수신 결과를 함께 반환합니다.
    전송 직후 수신 버퍼를 최대 timeout_sec 초 동안 대기합니다.

    Args:
        message     : 전송할 문자열
        timeout_sec : 수신 대기 최대 시간 (초, 기본 3.0)
        add_newline : True이면 메시지 끝에 '\n' 추가 (기본값 True)

    Returns:
        전송 결과와 수신된 메시지를 합친 문자열,
        또는 수신 없을 경우 타임아웃 메시지
    """
    # 전송
    send_result = send_message(message)
    # 수신 대기
    recv_result = receive_message(timeout_sec=timeout_sec)
    return f"{send_result}\n{recv_result}"


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Socket MCP 서버 시작 (stdio 모드)")
    mcp.run(transport="stdio")