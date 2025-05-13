from dataclasses import dataclass

@dataclass
class CistercianSymbol:
    number: int
    quadrant: str
    position: tuple  
    
    @property
    def is_valid(self):
        return 0 <= self.number <= 9999
