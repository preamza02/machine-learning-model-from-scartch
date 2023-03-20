class nice:
    def __init__(self) -> None:
        self.name = self._setname()
        print(self.name)
        
    def _setname(self):
        return "nice"
    
class innice(nice):
    def _setname(self):
        return "nice2"
    

a = nice()
b = innice()