import sys

class ProgressTracker:
    """A progress tracker"""    
    _process: str
    """Current process being worked on"""
    _formula: str
    """Current formula being worked on"""
    _progress: int
    """How far through processing"""
    _bar_size: int
    """How big of a bar to make"""
    _max: int
    """Total amount of processes to complete"""

    def __init__(self, form_count: int):
        self._process = "Initialized Process Tracker"
        self._formula = "N/A"
        self._progress = 0
        self._bar_size = 20
        self._max = max(form_count, 1)
        self._disabled = not sys.stdout.isatty()
        self._update(True)
    
    @property
    def process(self) -> str:
        return self._process
    
    @process.setter
    def process(self, new_process: str) -> None:
        self._process = new_process
        self._update()
    
    @property
    def formula(self) -> str:
        return self._formula
    
    @formula.setter
    def formula(self, new_formula: str) -> None:
        self._formula = new_formula
        self._update()
    
    @property
    def progress(self) -> int:
        return self._progress // self._bar_size
    
    @progress.setter
    def progress(self, new_progress: int) -> None:
        self._progress = new_progress * self._bar_size
        self._update()

    def _update(self, first: bool = False) -> None:
        if self._disabled:
            return
        if first:
            sys.stdout.write("=================================\n")
        else:
            for _ in range(3):
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
        
        sys.stdout.write(self._process+"\n")
        sys.stdout.write(self._formula+"\n")
        completed = min(self._progress // self._max, self._bar_size)
        incomplete = self._bar_size - completed
        progress = self.progress
        sys.stdout.write("["+('#' * completed)+('-' * incomplete)+"] "+str(round(progress * 100.0 / self._max, 2))+"% : "+str(progress)+"/"+str(self._max)+"\n")

        sys.stdout.flush()



    









