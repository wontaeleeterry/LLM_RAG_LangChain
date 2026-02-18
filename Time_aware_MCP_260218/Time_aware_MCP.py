import sys
import json
import asyncio
import datetime
import time
from datetime import timezone, timedelta

# ===============================
# CONFIG
# ===============================

KST = timezone(timedelta(hours=9))
SERVER_BOOT_TIME = datetime.datetime.now(KST)
MONOTONIC_START = time.monotonic()

# ===============================
# TOOLS
# ===============================

TOOLS = [
    {
        "name": "get_current_time",
        "description": "Return current time in Asia/Seoul and elapsed monotonic seconds.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "calculate_datetime",
        "description": "Calculate datetime relative to base date.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "today or specific ISO datetime"},
                "days_offset": {"type": "integer"},
                "hours_offset": {"type": "integer"}
            }
        }
    },
    {
        "name": "get_relative_range",
        "description": "Return date range for last N days based on today.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer"}
            },
            "required": ["days"]
        }
    },
    {
        "name": "schedule_task",
        "description": "Calculate future execution time after given seconds.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "delay_seconds": {"type": "integer"}
            },
            "required": ["delay_seconds"]
        }
    }
]

# ===============================
# TOOL IMPLEMENTATION
# ===============================

def now_kst():
    return datetime.datetime.now(KST)

def monotonic_elapsed():
    return time.monotonic() - MONOTONIC_START


async def handle_request(request):
    method = request.get("method")
    id_ = request.get("id")

    # 1️. Initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": "time-aware-mcp",
                    "version": "2.0.0"
                }
            }
        }

    # 2️. tools/list
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "tools": TOOLS
            }
        }

    # 3️. tools/call
    elif method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})

        # -----------------------
        if name == "get_current_time":
            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "current_time_kst": now_kst().isoformat(),
                                "elapsed_seconds": monotonic_elapsed()
                            }, indent=2)
                        }
                    ]
                }
            }

        # -----------------------
        elif name == "calculate_datetime":
            base = arguments.get("base", "today")
            days = arguments.get("days_offset", 0)
            hours = arguments.get("hours_offset", 0)

            if base == "today":
                base_dt = now_kst()
            else:
                base_dt = datetime.datetime.fromisoformat(base).astimezone(KST)

            result_dt = base_dt + datetime.timedelta(days=days, hours=hours)

            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_dt.isoformat()
                        }
                    ]
                }
            }

        # -----------------------
        elif name == "get_relative_range":
            days = arguments.get("days")
            end = now_kst()
            start = end - datetime.timedelta(days=days)

            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "start": start.isoformat(),
                                "end": end.isoformat()
                            }, indent=2)
                        }
                    ]
                }
            }

        # -----------------------
        elif name == "schedule_task":
            delay = arguments.get("delay_seconds")
            execution_time = now_kst() + datetime.timedelta(seconds=delay)

            return {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "execute_at": execution_time.isoformat(),
                                "delay_seconds": delay
                            }, indent=2)
                        }
                    ]
                }
            }

    return None


async def main():
    loop = asyncio.get_event_loop()

    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break

        try:
            request = json.loads(line)
            response = await handle_request(request)

            if response:
                print(json.dumps(response), flush=True)

        except Exception as e:
            print(f"ERROR: {str(e)}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
