from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict


class Index(BaseModel):
    index: int
    day: int

    @classmethod
    def from_day(cls, day: int, episode_length: int) -> "Index":
        return cls(index=day * episode_length, day=day)

    @classmethod
    def from_index(cls, index: int, episode_length: int) -> "Index":
        return cls(index=index, day=int(index / episode_length))

    @classmethod
    def null_index(cls) -> "Index":
        return Index(index=0, day=0)

    @classmethod
    def list_to_days(cls, indexes: List["Index"]) -> List[int]:
        return [i.day for i in indexes]

    @classmethod
    def list_to_index(cls, indexes: List["Index"], fill_days: int = 1) -> List[int]:
        """ Returns a list of indices from the Index list. If fill_days is 1, then just the start indexes are returned.
         If fill_days is bigger than 1, fill_days amount of indices after each start index is returned additionally. """

        if fill_days == 1:
            return [i.index for i in indexes]
        elif fill_days < 1:
            raise ValueError("fill_days must be a positive integer")

        return [
            t
            for index in indexes
            for t in range(index.day * fill_days, (index.day + 1) * fill_days)
        ]

    def __str__(self):
        return f"(idx={self.index},day={self.day})"

    def __hash__(self):
        return hash((self.index, self.day))

    def __eq__(self, other):
        if isinstance(other, Index):
            return self.index == other.index and self.day == other.day
        return NotImplemented

class RangeConfig(BaseModel):
    start: int = 0
    end: Optional[int] = None
    step: int = 1
    values: Optional[List[int]] = None

    def get_indexes(self, episode_length: int) -> List[Index]:
        if self.end:
            return [Index.from_day(i, episode_length) for i in range(self.start, self.end + 1, self.step)]
        elif self.values:
            return [Index.from_day(i, episode_length) for i in sorted(self.values)]
        else:
            raise ValueError("Either 'end' or 'values' must be provided.")

    def get_timesteps(self, episode_length: int) -> List[int]:
        """ Converts indexes (in days format) into timesteps """
        return Index.list_to_index(self.get_indexes(episode_length), fill_days=episode_length)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "start": self.start,
            "step": self.step,
        }

        if self.end:
            result["end"] = self.end
        elif self.values:
            result["values"] = self.values

        return result