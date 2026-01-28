# MCP -> FastMCP로 코드 변환 (260128, 작성 중)

#!/usr/bin/env python3
"""
MongoDB Aggregation FastMCP Server (Async + Pydantic)
DailyLog 컬렉션에 대한 Aggregation Query를 실행하는 MCP 서버
"""

import json
import os
from datetime import datetime
from typing import Any, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pydantic import BaseModel, Field, conint
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from mcp.server.fastmcp import FastMCP

# --------------------------------------------------
# 환경변수 로드
# --------------------------------------------------
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "TerryDB"
COLLECTION_NAME = "DailyLog"

# --------------------------------------------------
# DailyLog 스키마
# --------------------------------------------------
DAILYLOG_SCHEMA = {
    "collection": "DailyLog",
    "database": "TerryDB",
    "description": "일일 건강 및 활동 로그",
    "fields": {
        "date": {
            "type": "string",
            "format": "YYYY-MM-DD",
            "description": "날짜"
        },
        "sleep": {
            "type": "object",
            "description": "수면 정보",
            "fields": {
                "hours": {"type": "float", "description": "수면 시간"},
                "quality": {"type": "string", "description": "수면 품질"}
            }
        },
        "inbody": {
            "type": "object",
            "description": "체성분 정보",
            "fields": {
                "weight": {"type": "float", "description": "체중 (kg)"},
                "muscle": {"type": "float", "description": "근육량 (kg)"},
                "fatper": {"type": "float", "description": "체지방률 (%)"}
            }
        },
        "study": {
            "type": "array",
            "description": "공부 기록",
            "items": {
                "subject": {"type": "string", "description": "과목"},
                "minutes": {"type": "int", "description": "공부 시간 (분)"}
            }
        },
        "bloodpressure": {
            "type": "object",
            "description": "혈압 정보",
            "fields": {
                "High": {"type": "int", "description": "수축기 혈압"},
                "low": {"type": "int", "description": "이완기 혈압"},
                "bpm": {"type": "int", "description": "심박수"}
            }
        },
        "expenses": {
            "type": "array",
            "description": "지출 내역",
            "items": {
                "category": {"type": "string", "description": "카테고리"},
                "amount": {"type": "int", "description": "금액"}
            }
        },
        "exercise": {
            "type": "array",
            "description": "운동 기록",
            "items": {
                "subject": {"type": "string", "description": "운동 종류"},
                "steps/count": {"type": "int", "description": "걸음 수 또는 횟수"}
            }
        },
        "food": {
            "type": "array",
            "description": "식사 기록",
            "items": {
                "breakfast": {"type": "string", "description": "아침"},
                "lunch": {"type": "string", "description": "점심"},
                "dinner": {"type": "string", "description": "저녁"},
                "refreshments": {"type": "string", "description": "간식"}
            }
        },
        "mood": {
            "type": "string",
            "description": "기분/컨디션"
        }
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

# --------------------------------------------------
# MongoDB 연결
# --------------------------------------------------
mongo_client: MongoClient | None = None
db = None
collection = None


def initialize_mongodb():
    global mongo_client, db, collection
    try:
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command("ping")
        db = mongo_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print(f"MongoDB 연결 성공: {DATABASE_NAME}.{COLLECTION_NAME}")
    except ConnectionFailure as e:
        print(f"MongoDB 연결 실패: {e}")
        raise


# --------------------------------------------------
# MongoDB 결과 직렬화
# --------------------------------------------------
def serialize_mongo_result(obj: Any) -> Any:
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_mongo_result(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_mongo_result(i) for i in obj]
    return obj


# --------------------------------------------------
# Pydantic 입력 스키마
# --------------------------------------------------
class AggregationRequest(BaseModel):
    pipeline: List[dict] = Field(
        ...,
        description="MongoDB aggregation pipeline stages"
    )
    limit: conint(ge=1, le=1000) = Field(
        100,
        description="Maximum number of results (1–1000)"
    )


class EmptyRequest(BaseModel):
    pass


# --------------------------------------------------
# FastMCP 앱
# --------------------------------------------------
mcp = FastMCP("mongodb-aggregation-fastmcp-async")


# --------------------------------------------------
# Tools
# --------------------------------------------------
@mcp.tool()
async def get_collection_schema(_: EmptyRequest) -> str:
    """
    Get the schema and structure information for the DailyLog collection.
    """
    try:
        stats = {
            "document_count": collection.count_documents({}),
            "sample_document": None
        }

        sample = collection.find_one(sort=[("date", -1)])
        if sample:
            stats["sample_document"] = serialize_mongo_result(sample)

        result = {
            "schema": DAILYLOG_SCHEMA,
            "statistics": stats
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


@mcp.tool()
async def execute_aggregation(req: AggregationRequest) -> str:
    """
    Execute a MongoDB aggregation pipeline on the DailyLog collection.
    """
    try:
        pipeline = req.pipeline
        limit = req.limit

        pipeline_with_limit = pipeline + [{"$limit": limit}]
        results = list(collection.aggregate(pipeline_with_limit))
        serialized_results = serialize_mongo_result(results)

        response = {
            "success": True,
            "pipeline": pipeline,
            "result_count": len(results),
            "results": serialized_results
        }

        return json.dumps(response, indent=2, ensure_ascii=False)

    except OperationFailure as e:
        return f"MongoDB operation failed: {str(e)}\nPlease check your aggregation pipeline syntax."
    except Exception as e:
        return f"Error executing aggregation: {str(e)}"


# --------------------------------------------------
# 엔트리포인트
# --------------------------------------------------
if __name__ == "__main__":
    initialize_mongodb()
    mcp.run()

