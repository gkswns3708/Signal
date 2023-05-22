from typing import Optional

from pydantic import BaseModel, UUID


# TODO : 현재 생성할 Entity들은 Backend와 통신하거나 DataStorage에 저장, 참조할 예정이기에 필요할 것으로 생각됨.
class member_schema(BaseModel):
    Idx : int = ...
    UUID : Optional[UUID] = None
    Nickname : Optional[str] = None
    Field : Optional[str] = None # TODO : Type(?)