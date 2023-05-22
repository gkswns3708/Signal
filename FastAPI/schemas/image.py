from typing import Optional

from sqlalchemy import create_engine

from pydantic import BaseModel
from caption import Caption_schema
from gps import Gps_schema


# TODO : 현재 생성할 Entity들은 Backend와 통신하거나 DataStorage에 저장, 참조할 예정이기에 필요할 것으로 생각됨.
class Image_schema(BaseModel):
   Idx : int = ...
   Cap : Caption_schema = ...
   SavePath : str = ...
   MemberIdx : int = ...
   Gps : Gps_schema