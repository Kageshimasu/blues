import pandas as pd
import os
from tqdm import tqdm

from blues.base.base_predictor import BasePredictor


class ClassificationPredictor(BasePredictor):

    def __init__(self, predicting_table, annotation_table, pred_dataset, result_path):
        super().__init__(predicting_table, annotation_table, pred_dataset, result_path)

    def run(self):
        preds_dir = {}
        os.makedirs(self._result_path, exist_ok=True)

        preds_dir['image_path'] = []
        for _, class_name in self._annotation_table:
            preds_dir[class_name] = []

        for data in tqdm(self._pred_dataset):
            inputs = data.get_inputs_on_torch()
            file_names = data.get_file_names()
            total_acc = 0
            total_preds = None

            for acc, model in self._predicting_table:
                preds = model.predict(inputs) * acc
                total_acc += acc
                if total_preds is None:
                    total_preds = preds
                else:
                    total_preds += preds

            final_preds = total_preds / total_acc
            batch_size = final_preds.shape[0]
            for b in range(batch_size):
                preds_dir['image_path'].append(file_names[b])
                for id in range(final_preds.shape[1]):
                    class_name = self._annotation_table.get_class_name_from_id(id)
                    preds_dir[class_name].append(final_preds[b][id])
        pd.DataFrame(preds_dir).to_csv(os.path.join(self._result_path, 'submission.csv'), index=False)
