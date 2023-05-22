from typing import Optional

from pydantic import BaseModel

class Gps_schema(BaseModel):
    Key : int = ... # TODO : 정확한 key값 정의 안됨 수정 예정.
    Longitude : int 
    Latitude : int
    ImgIdx : int = ...