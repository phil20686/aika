from abc import abstractmethod, ABC


class Dataset:

    def __init__(self, X, y=None):
        self._X=X
        self._y=y

class Transformer(ABC):


    @abstractmethod
    def fit(self, dataset : Dataset) -> None:
        """
        May require only X, or both X and y to fit depending on the transformer.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset : Dataset) -> Dataset:
        """
        Note that some transformations may overwrite y. Eg, transform for a regression
        will use X to generate new y, this will over-write any y that is passed in.
        """
        raise NotImplementedError


    def fit_transform(self, dataset) -> Dataset:
        """
        Convience method that just calls fit and then transform
        """
        self.fit(dataset)
        return self.transform(dataset)


class Pipeline:
    """
    A pipeline is just a set of steps.
    """

    def __init__(self, steps : List[Transformer]):
        self._steps = steps

    @property
    def steps(self):
        return self._steps

    def fit_transform(self, dataset : Dataset):
        for step in self.steps:
            dataset = step.fit_transform(dataset)
        return dataset

    def transform(self, dataset : Dataset):
        for step in self.steps:
            dataset = step.transform(dataset)
        return dataset

    def fit(self, dataset : Dataset):
        self.fit_transform(dataset)

