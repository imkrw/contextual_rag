import uuid
from typing import Optional
from agents import SQLiteSession


class ChatSession:
    def __init__(self, session_db: str) -> None:
        self.session_db = session_db
        self.session: Optional[SQLiteSession] = None

    def ensure(self) -> SQLiteSession:
        if self.session is None:
            session_id = uuid.uuid4().hex
            self.session = SQLiteSession(session_id, db_path=self.session_db)
        return self.session

    def reset(self) -> None:
        self.session = None


def reset_session(chat_session: ChatSession):
    chat_session.reset()
    return [], ""
