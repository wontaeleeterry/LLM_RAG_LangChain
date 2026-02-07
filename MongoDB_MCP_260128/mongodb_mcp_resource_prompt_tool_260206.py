#!/usr/bin/env python3
# 이모지 삭제 후 정상 작동 (260207)

"""
MongoDB Aggregation MCP Server (Structured & Fixed Version)

이 서버는 MCP의 세 가지 핵심 Capability를 명확하게 구현합니다:
- Prompts   : AI에게 서버 목적과 사용 흐름을 설명
- Resources : AI가 읽을 수 있는 데이터 구조 및 메타정보 제공
- Tools     : MongoDB Aggregation 실행 기능 제공
"""

import json
import os
from datetime import datetime
from typing import Any, Sequence
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptMessage,
    GetPromptResult
)
import mcp.server.stdio

# ═════════════════════════════════════════════════════════════════════
# SECTION 1: Environment & MongoDB Configuration
# ═════════════════════════════════════════════════════════════════════

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "TerryDB"
COLLECTION_NAME = "DailyLog"

mongo_client = None
db = None
collection = None


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: Domain Schema (Resource Data)
# ═════════════════════════════════════════════════════════════════════

DAILYLOG_SCHEMA = {
    "collection": "DailyLog",
    "database": "TerryDB",
    "description": "일일 건강 및 활동 로그",
    "fields": {
        "date": {"type": "date", "description": "기록 날짜"},
        "sleep": {
            "type": "object",
            "fields": {
                "hours": {"type": "float"},
                "quality": {"type": "string"}
            }
        },
        "inbody": {
            "type": "object",
            "fields": {
                "weight": {"type": "float"},
                "muscle": {"type": "float"},
                "fatper": {"type": "float"}
            }
        },
        "study": {
            "type": "array",
            "items": {
                "subject": {"type": "string"},
                "minutes": {"type": "int"}
            }
        },
        "bloodpressure": {
            "type": "object",
            "fields": {
                "systolic": {"type": "int"},
                "diastolic": {"type": "int"},
                "bpm": {"type": "int"}
            }
        },
        "expenses": {
            "type": "array",
            "items": {
                "category": {"type": "string"},
                "amount": {"type": "float"}
            }
        },
        "exercise": {
            "type": "array",
            "items": {
                "subject": {"type": "string"},
                "steps": {"type": "int", "optional": True},
                "count": {"type": "int", "optional": True}
            }
        },
        "food": {
            "type": "array",
            "items": {
                "type": {"type": "string"},
                "menu": {"type": "string"}
            }
        },
        "mood": {"type": "string"}
    },
    "example_queries": [
        {
            "description": "지난 7일 평균 수면 시간",
            "pipeline": [
                {"$match": {"date": {"$gte": "2026-01-19"}}},
                {"$group": {"_id": None, "avg_sleep": {"$avg": "$sleep.hours"}}}
            ]
        },
        {
            "description": "카테고리별 총 지출",
            "pipeline": [
                {"$unwind": "$expenses"},
                {"$group": {"_id": "$expenses.category", "total": {"$sum": "$expenses.amount"}}},
                {"$sort": {"total": -1}}
            ]
        },
        {
            "description": "과목별 총 공부 시간",
            "pipeline": [
                {"$unwind": "$study"},
                {"$group": {"_id": "$study.subject", "total_minutes": {"$sum": "$study.minutes"}}},
                {"$sort": {"total_minutes": -1}}
            ]
        }
    ]
}


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: Infrastructure Utilities
# ═════════════════════════════════════════════════════════════════════

def initialize_mongodb():
    """MongoDB 연결 초기화"""
    global mongo_client, db, collection
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print(f" MongoDB 연결 성공: {DATABASE_NAME}.{COLLECTION_NAME}")
    except ConnectionFailure as e:
        print(f" MongoDB 연결 실패: {e}")
        raise


def serialize_mongo(obj: Any) -> Any:
    """MongoDB 결과를 JSON 직렬화 가능한 구조로 변환"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_mongo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_mongo(v) for v in obj]
    return obj


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: MCP Server Initialization
# ═════════════════════════════════════════════════════════════════════

app = Server("mongodb-aggregation-server")


# ═════════════════════════════════════════════════════════════════════
# CAPABILITY 1️: PROMPTS
# → AI에게 이 MCP 서버의 목적과 사용 흐름을 설명
# ═════════════════════════════════════════════════════════════════════

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """사용 가능한 프롬프트 목록 반환"""
    return [
        Prompt(
            name="mongodb_analytics_assistant",
            description="DailyLog MongoDB 컬렉션에 대해 분석 질문을 수행하는 전문가 AI 역할",
            arguments=[]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> GetPromptResult:
    """특정 프롬프트의 상세 내용 반환"""
    if name == "mongodb_analytics_assistant":
        return GetPromptResult(
            description="DailyLog 데이터 분석을 위한 MongoDB 전문가 AI",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            "You are an analytics assistant connected to a MongoDB database.\n"
                            "Your job is to help users analyze their DailyLog data.\n\n"
                            "**Workflow:**\n"
                            "1. First, read the collection schema using the resource: dailylog://schema\n"
                            "2. Understand the data structure and available fields\n"
                            "3. Design an appropriate MongoDB aggregation pipeline\n"
                            "4. Execute it using the execute_aggregation tool\n"
                            "5. Interpret and explain the results clearly\n\n"
                            "**Important:**\n"
                            "- Always examine the schema before writing pipelines\n"
                            "- Use example queries in the schema as reference\n"
                            "- Explain your aggregation logic to the user\n"
                            "- Handle errors gracefully and suggest corrections\n\n"
                            "Now, help the user analyze their DailyLog data!"
                        )
                    )
                )
            ]
        )
    
    raise ValueError(f"Unknown prompt: {name}")


# ═════════════════════════════════════════════════════════════════════
# CAPABILITY 2️: RESOURCES
# → AI가 읽을 수 있는 외부 세계의 데이터 구조와 메타정보
# ═════════════════════════════════════════════════════════════════════

@app.list_resources()
async def list_resources() -> list[Resource]:
    """사용 가능한 리소스 목록 반환"""
    return [
        Resource(
            uri="dailylog://schema",
            name="DailyLog Collection Schema",
            description="DailyLog MongoDB 컬렉션의 필드 구조, 타입, 예시 쿼리 정보",
            mimeType="application/json"
        ),
        Resource(
            uri="dailylog://stats",
            name="DailyLog Collection Statistics",
            description="DailyLog 컬렉션의 문서 수 및 최신 샘플 데이터",
            mimeType="application/json"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """특정 리소스의 내용 반환"""
    if uri == "dailylog://schema":
        return json.dumps(DAILYLOG_SCHEMA, indent=2, ensure_ascii=False)

    if uri == "dailylog://stats":
        stats = {
            "document_count": collection.count_documents({}),
            "sample_document": None,
            "date_range": None
        }
        
        # 최신 문서 샘플
        sample = collection.find_one(sort=[("date", -1)])
        if sample:
            stats["sample_document"] = serialize_mongo(sample)
        
        # 날짜 범위
        oldest = collection.find_one(sort=[("date", 1)])
        newest = collection.find_one(sort=[("date", -1)])
        if oldest and newest:
            stats["date_range"] = {
                "oldest": serialize_mongo(oldest.get("date")),
                "newest": serialize_mongo(newest.get("date"))
            }

        return json.dumps(stats, indent=2, ensure_ascii=False)

    raise ValueError(f"Unknown resource URI: {uri}")


# ═════════════════════════════════════════════════════════════════════
# CAPABILITY 3️: TOOLS
# → AI가 실제로 실행할 수 있는 외부 세계 조작 인터페이스
# ═════════════════════════════════════════════════════════════════════

@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록 반환"""
    return [
        Tool(
            name="execute_aggregation",
            description=(
                "Execute a MongoDB aggregation pipeline on the DailyLog collection.\n\n"
                "**Before using this tool:**\n"
                "- Read dailylog://schema resource to understand the data structure\n"
                "- Review example queries in the schema\n\n"
                "**Supported stages:**\n"
                "$match, $group, $sort, $project, $unwind, $limit, $skip, $lookup, $addFields\n\n"
                "**Example:**\n"
                "```json\n"
                "{\n"
                '  "pipeline": [\n'
                '    {"$match": {"date": {"$gte": "2026-01-01"}}},\n'
                '    {"$group": {"_id": null, "avg_sleep": {"$avg": "$sleep.hours"}}}\n'
                "  ]\n"
                "}\n"
                "```"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline": {
                        "type": "array",
                        "description": "MongoDB aggregation pipeline (array of stage objects)",
                        "items": {"type": "object"}
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results to return (default 100, max 1000)",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000
                    }
                },
                "required": ["pipeline"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """도구 실행"""
    if name != "execute_aggregation":
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Unknown tool: {name}"
            }, ensure_ascii=False)
        )]

    try:
        pipeline = arguments.get("pipeline", [])
        limit = min(arguments.get("limit", 100), 1000)

        # 입력 검증
        if not isinstance(pipeline, list):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "pipeline must be an array of stage objects"
                }, ensure_ascii=False)
            )]

        if len(pipeline) == 0:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": "pipeline cannot be empty"
                }, ensure_ascii=False)
            )]

        # 파이프라인 실행
        pipeline_with_limit = pipeline + [{"$limit": limit}]
        results = list(collection.aggregate(pipeline_with_limit))
        serialized = serialize_mongo(results)

        response = {
            "success": True,
            "pipeline": pipeline,
            "result_count": len(serialized),
            "results": serialized
        }

        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2, ensure_ascii=False)
        )]

    except OperationFailure as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"MongoDB operation failed: {str(e)}",
                "error_type": "OperationFailure"
            }, indent=2, ensure_ascii=False)
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": type(e).__name__
            }, indent=2, ensure_ascii=False)
        )]


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: Server Entrypoint
# ═════════════════════════════════════════════════════════════════════

async def main():
    """서버 시작"""
    initialize_mongodb()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())