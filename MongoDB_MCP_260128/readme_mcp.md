# MongoDB Aggregation MCP Server

DailyLog 컬렉션에 대한 자연어 쿼리를 MongoDB Aggregation Pipeline으로 변환하여 실행하는 MCP 서버입니다.

## 기능

- **스키마 조회**: DailyLog 컬렉션의 구조와 필드 정보 제공
- **Aggregation 실행**: MongoDB aggregation pipeline을 실행하고 결과 반환
- **자동 쿼리 생성**: Claude가 자연어 요청을 aggregation query로 자동 변환

## 설치

### 1. 필수 요구사항

- Python 3.10 이상
- MongoDB 서버 (로컬 또는 MongoDB Atlas)

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env` 파일을 생성하고 MongoDB 연결 정보를 입력합니다:

```bash
cp .env.example .env
```

`.env` 파일 편집:
```
MONGODB_URI=mongodb://localhost:27017/
```

## Claude Desktop 설정

Claude Desktop에서 MCP 서버를 사용하려면 설정 파일을 수정합니다.

### Windows

파일 위치: `%APPDATA%\Claude\claude_desktop_config.json`

### macOS

파일 위치: `~/Library/Application Support/Claude/claude_desktop_config.json`

### Linux

파일 위치: `~/.config/Claude/claude_desktop_config.json`

### 설정 내용

```json
{
  "mcpServers": {
    "mongodb-dailylog": {
      "command": "python",
      "args": [
        "/절대/경로/mongodb_mcp_server.py"
      ],
      "env": {
        "MONGODB_URI": "mongodb://localhost:27017/"
      }
    }
  }
}
```

**주의**: `/절대/경로/`는 실제 파일이 위치한 절대 경로로 변경해야 합니다.

## 사용 예시

Claude Desktop에서 다음과 같이 자연어로 요청할 수 있습니다:

### 예시 1: 평균 수면 시간 조회
```
지난 일주일간 평균 수면 시간을 알려줘
```

Claude가 자동으로 다음과 같은 aggregation pipeline을 생성합니다:
```json
[
  {"$match": {"date": {"$gte": "2026-01-19"}}},
  {"$group": {"_id": null, "avg_sleep": {"$avg": "$sleep.hours"}}}
]
```

### 예시 2: 카테고리별 지출 분석
```
이번 달 카테고리별 총 지출을 계산해줘
```

생성되는 pipeline:
```json
[
  {"$match": {"date": {"$gte": "2026-01-01", "$lte": "2026-01-31"}}},
  {"$unwind": "$expenses"},
  {"$group": {"_id": "$expenses.category", "total": {"$sum": "$expenses.amount"}}},
  {"$sort": {"total": -1}}
]
```

### 예시 3: 체중 변화 추이
```
최근 30일간 체중 변화를 날짜별로 보여줘
```

생성되는 pipeline:
```json
[
  {"$match": {"date": {"$gte": "2025-12-27"}}},
  {"$project": {"date": 1, "weight": "$inbody.weight"}},
  {"$sort": {"date": 1}}
]
```

### 예시 4: 과목별 공부 시간
```
과목별 총 공부 시간을 순위로 보여줘
```

생성되는 pipeline:
```json
[
  {"$unwind": "$study"},
  {"$group": {"_id": "$study.subject", "total_minutes": {"$sum": "$study.minutes"}}},
  {"$sort": {"total_minutes": -1}}
]
```

## DailyLog 스키마 구조

```javascript
{
  "date": "YYYY-MM-DD",
  "sleep": {
    "hours": 7.5,
    "quality": "good"
  },
  "inbody": {
    "weight": 70.5,
    "muscle": 35.2,
    "fatper": 18.5
  },
  "study": [
    {"subject": "Python", "minutes": 120},
    {"subject": "Math", "minutes": 60}
  ],
  "bloodpressure": {
    "High": 120,
    "low": 80,
    "bpm": 72
  },
  "expenses": [
    {"category": "식비", "amount": 15000},
    {"category": "교통", "amount": 5000}
  ],
  "exercise": [
    {"subject": "걷기", "steps/count": 10000},
    {"subject": "런닝", "steps/count": 5000}
  ],
  "food": [
    {"breakfast": "오트밀, 바나나"},
    {"lunch": "샐러드, 치킨"},
    {"dinner": "현미밥, 생선구이"},
    {"refreshments": "사과"}
  ],
  "mood": "상쾌함"
}
```

## 주요 Aggregation 연산자

- `$match`: 문서 필터링
- `$group`: 그룹화 및 집계
- `$sort`: 정렬
- `$project`: 필드 선택/변환
- `$unwind`: 배열 필드 펼치기
- `$limit`: 결과 개수 제한
- `$sum`: 합계
- `$avg`: 평균
- `$min`/`$max`: 최소/최대값

## 문제 해결

### MongoDB 연결 실패
- MongoDB 서버가 실행 중인지 확인
- `.env` 파일의 MONGODB_URI가 올바른지 확인
- 방화벽 설정 확인

### Claude Desktop에서 서버가 보이지 않음
- Claude Desktop 재시작
- `claude_desktop_config.json` 파일 경로 확인
- Python 경로가 올바른지 확인 (`which python` 또는 `where python`)

### 쿼리 실행 오류
- 스키마 정보를 먼저 확인 (`get_collection_schema` 도구 사용)
- aggregation pipeline 문법 확인
- 필드명이 정확한지 확인 (대소문자 구분)

## 라이선스

MIT License
