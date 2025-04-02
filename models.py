# This file is included to maintain consistency with the Flask blueprint structure
# but is not needed for this specific application as we are not using a database.
# If we expand to include user accounts or store conversion history, models would go here.

from dataclasses import dataclass

@dataclass
class CistercianSymbol:
    """
    Represents a Cistercian numeral symbol with its properties.
    """
    number: int
    quadrant: str  # 'units', 'tens', 'hundreds', 'thousands'
    position: tuple  # (x, y) coordinates
    
    @property
    def is_valid(self):
        return 0 <= self.number <= 9999
