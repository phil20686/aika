from typing import Dict


class PersistenceEngine:

    pass

class iDataSetDescription:

    @property
    def params(self):
        raise NotImplementedError

    @property
    def predecessors(self) -> Dict[str, "iDataSetDescription"]:
        raise NotImplementedError



class iDataSet():

class iDataGraph:


    pass