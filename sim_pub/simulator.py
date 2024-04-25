from abc import ABC, abstractmethod
from pathlib import Path

class PublisherSimulator(ABC):

  @abstractmethod
  def initialize_data(file_path : Path, **kwargs):
    pass

  
  @abstractmethod
  def update():
    pass

