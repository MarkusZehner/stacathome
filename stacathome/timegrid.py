import pandas as pd


class TimeGridCell:

    def __init__(self, index: int, start: pd.Timestamp, end: pd.Timestamp, grid: 'TimeGrid'):
        self.index = index
        self.start = start
        self.end = end
        self.grid = grid
        
    def __contains__(self, date: pd.Timestamp):
        """
        Tests if a date is within the interval.
        """
        return self.start <= date < self.end
    
    def before(self, date: pd.Timestamp):
        """
        Returns True if the interval is before the given date.
        """
        return date >= self.end
    
    def after(self, date: pd.Timestamp):
        """
        Returns True if the interval is after the given date.
        """
        return date < self.start

    def abbrev(self) -> str:
        return f'{self.index}'

    def __repr__(self):
        return f'TimeGridCell(index={self.index}, start={repr(self.start)}, end={repr(self.end)})'

    def __eq__(self, other):
        return self.index == other.index and self.grid == other.grid
    
    def __hash__(self):
        return hash((self.index, id(self.grid)))


class TimeGrid:

    def __init__(self, name: str, start: pd.Timestamp | str, end: pd.Timestamp | str, freq: str):
        self.name = name
        self.start = start
        self.end = end
        self.freq = freq
        self._dt_index = pd.date_range(start=self.start, end=self.end, freq=self.freq)

    def __len__(self):
        return len(self._dt_index) - 1
    
    def __getitem__(self, index) -> TimeGridCell | list[TimeGridCell]:
        if isinstance(index, int):
            start = self._dt_index[index]
            end = self._dt_index[index + 1]
            return TimeGridCell(index, start, end, grid=self)
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        else:
            raise TypeError('Invalid argument type')
    
    def left_bound(self, date: pd.Timestamp | str) -> pd.Timestamp:
        idx = self.date_to_index(date)
        return self._dt_index[idx]
    
    def right_bound(self, date: pd.Timestamp | str) -> pd.Timestamp:
        idx = self.date_to_index(date)
        return self._dt_index[idx + 1]

    def date_to_index(self, date: pd.Timestamp | str) -> int:
        date = pd.Timestamp(date)

        try:
            loc = self._dt_index.get_loc(date)
        except KeyError:
            # KeyError -> No exact match, try for padded
            indexer = self._dt_index.get_indexer([date], method="pad")
            loc = indexer.item()
        
        if loc == -1 or loc >= len(self):
            raise KeyError(f'{date} is not in the DateGrid')

        return loc

    def index_range(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> range:
        start_idx = self.date_to_index(start)
        end_idx = self.date_to_index(end)
        return range(start_idx, end_idx)

    def find_date(self, date: pd.Timestamp | str) -> TimeGridCell:
        """
        Find the coordinate that contains the given date.
        """
        date = pd.Timestamp(date)
        idx = self.date_to_index(date)
        return self[idx]
    
    def find_dates(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> list[TimeGridCell]:
        """
        Find all coordinates that intersect with the given interval.

        The interval is inclusive on the left and exclusive on the right.
        """
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        start_idx = self.date_to_index(start)
        end_idx = self.date_to_index(end - pd.Timedelta(1, unit='ns'))
        return self[start_idx:end_idx+1]

    def __repr__(self):
        return f'TimeGrid(name={self.name}, start={self.start}, end={self.end}, freq={self.freq})'


WT5 = TimeGrid(name='WT5', start='1900-01-01', end='2100-01-01', freq='5D')
WT15 = TimeGrid(name='WT10', start='1900-01-01', end='2100-01-01', freq='10D')
WT15 = TimeGrid(name='WT15', start='1900-01-01', end='2100-01-01', freq='15D')


if __name__ == '__main__':
    print(WT15)
    print()

    print(WT15[-1:])
    print()
    print(WT15.find_coords('2000-01-01', '2000-02-02'))