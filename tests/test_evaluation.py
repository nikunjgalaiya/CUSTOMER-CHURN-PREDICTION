import unittest
import numpy as np
from evaluate import FEATURES,ROOT,load_data,scores

class EvaluationTests(unittest.TestCase):
    def test_source_contract(self):
        frame=load_data(ROOT/"customer_churn_large_dataset.xlsx")
        self.assertTrue(all(c in frame for c in FEATURES));self.assertGreater(len(frame),100)
    def test_auc_uses_probability_not_thresholded_labels(self):
        s=scores([0,0,1,1],[.1,.2,.3,.4])
        self.assertEqual(s["roc_auc"],1);self.assertEqual(s["recall"],0)
    def test_confusion_matrix_labels(self):
        s=scores([0,0,1,1],[.1,.8,.9,.2])
        self.assertEqual(s["confusion_matrix"],{"tn":1,"fp":1,"fn":1,"tp":1})
    def test_no_positive_predictions_have_zero_precision(self):
        self.assertEqual(scores([0,1],[.1,.2])["precision"],0)
if __name__=="__main__":unittest.main()
