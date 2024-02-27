from typing import Dict, List, Union

HEADER_SEPERATOR = ":::"
REQUEST_SEPERATOR = "@@@"

class MsgPack:

    def __init__(self, header: str, msg: str) -> None:
        if not header or not msg:
            raise ValueError("Header and message must be non-empty strings.")
        self.header: str = header
        self.msg: str = msg

    def __str__(self) -> str:
        """String representation of the MsgPack object."""
        return f"{self.header}{HEADER_SEPERATOR}{self.msg}"

class RequestMsg(MsgPack):
    pass

class ResponseMsg(MsgPack):
    pass

class MsgPackParser:

    def parse(self, pack: str) -> MsgPack:
        parts = pack.split(HEADER_SEPERATOR, maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Packed string does not contain the header separator '{HEADER_SEPERATOR}'.")
        header, msg = parts
        return MsgPack(header, msg)

class MsgFactory:

    def create_msg(self, header: str, msg: str) -> MsgPack:
        return MsgPack(header, msg)

    def create_request(req: str) -> MsgPack:
        pass


class ObjData(Dict[str, Union[str, List[float], bool]]):
    pass

class SimData(Dict[str, ObjData]):
    pass
