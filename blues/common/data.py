class Data:

    def __init__(self, inputs, teachers, file_name: list, scale: float = 1):
        """
        :param inputs: tensor
        :param teachers: tensor
        :param file_name: list
        :param transformers:
        """
        self._inputs = inputs
        self._teachers = teachers
        self._file_name = file_name
        self._scale = scale

    def get_inputs(self):
        return self._inputs

    def get_teachers(self):
        return self._teachers

    def get_file_names(self):
        return self._file_name

    def get_scale(self):
        return self._scale
