from typing import List


class _ModelConst:
    NUM_CLASSES = 'num_class'
    NETWORK = 'network'
    STATE_DICT = 'state_dict'
    OPTIMIZER = 'optimizer'
    MODEL_NAME = 'model_name'

    class ConstError(TypeError):
        pass

    def get_all_consts(self) -> List[str]:
        return [self.NUM_CLASSES, self.NETWORK, self.STATE_DICT, self.OPTIMIZER, self.MODEL_NAME]

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value
