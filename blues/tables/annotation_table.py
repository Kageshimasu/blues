class AnnotationTable:

    def __init__(self, annotation_dir: dir):
        """
        :param predicting_table:
        {
        '0': 'class1',
        '1': 'class2',
        }
        """
        if len(annotation_dir) == 0:
            raise ValueError('the table has one value or over')
        for key in annotation_dir.keys():
            if type(key) is not str or type(annotation_dir[key]) is not str:
                raise ValueError('both keys and values must be string type')
        self._annotation_dir = annotation_dir
        self._keys = list(annotation_dir.keys())
        self._i = 0

    def __len__(self):
        return len(self._annotation_dir)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration()
        id = self._keys[self._i]
        class_name = self.get_class_name_from_id(id)
        self._i += 1
        return id, class_name

    def get_class_name_from_id(self, id):
        return self._annotation_dir[str(id)]

    def get_id_from_class_name(self, class_name):
        id = [k for k, v in self._annotation_dir.items() if v == class_name][0]
        return id
