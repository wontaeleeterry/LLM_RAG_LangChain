#!/usr/bin/env python3
"""
샘플 데이터를 MongoDB에 임포트하는 스크립트
"""

import json
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "TerryDB"
COLLECTION_NAME = "DailyLog"

def import_sample_data():
    """샘플 데이터 임포트"""
    try:
        # MongoDB 연결
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # sample_data.json 읽기
        with open('sample_data.json', 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # 기존 데이터 확인
        existing_count = collection.count_documents({})
        print(f"기존 문서 수: {existing_count}")
        
        # 데이터 삽입
        if sample_data:
            result = collection.insert_many(sample_data)
            print(f"✅ {len(result.inserted_ids)}개의 문서가 성공적으로 삽입되었습니다.")
            print(f"총 문서 수: {collection.count_documents({})}")
        else:
            print("삽입할 데이터가 없습니다.")
        
        # 샘플 문서 조회
        print("\n최신 문서 샘플:")
        latest = collection.find_one(sort=[("date", -1)])
        print(json.dumps(latest, indent=2, ensure_ascii=False, default=str))
        
        client.close()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    import_sample_data()
