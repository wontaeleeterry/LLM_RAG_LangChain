import sqlite3, json
from typing import Any, Dict, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver  # langgragh → langgraph

class SQLiteCheckpointSaver(BaseCheckpointSaver):  # Chekpoint → Checkpoint
    def __init__(self, db_path: str = "conversation.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                              CREATE TABLE IF NOT EXISTS checkpoints(
                              thread_id TEXT,
                              step_id TEXT,
                              state TEXT,
                              PRIMARY KEY(thread_id, step_id)
                              )
                              """)
            
    def save(self, thread_id: str, step_id: str, state: Dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute("""
                              INSERT OR REPLACE INTO checkpoints (thread_id, step_id, state)
                              VALUES (?, ?, ?)
                              """, (thread_id, step_id, json.dumps(state)))
            
    def load(self, thread_id: str, step_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        if step_id:
            cursor.execute("SELECT state FROM checkpoints WHERE thread_id=? AND step_id=?", (thread_id, step_id))
        else:
            cursor.execute("SELECT state FROM checkpoints WHERE thread_id=? ORDER BY step_id DESC LIMIT 1", (thread_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None