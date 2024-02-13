from .unity_entity import UMetaEntity

class URDFOrigin(UMetaEntity):

	def __init__(self, element: XMLNode) -> None:
		super().__init__(element)
		self.xyz : List[float] = element.get("xyz", [0.0, 0.0, 0.0])
		self.rpy : List[float] = element.get("rpy", [0.0, 0.0, 0.0])
		
class URDFLoader(XMLFileLoader):
    pass