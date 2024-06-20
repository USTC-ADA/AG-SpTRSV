import model_predict
import mlp_model_def
import sys

if (len(sys.argv) != 2):
    print("Usage: python model_pythonc_test.py {model_name}")
    exit(1)

model_name = sys.argv[1]

model_predict.module_load(model_name)

input_feature = [300.0, 592.0, 1.9733333587646484, 38.0, 2.9802238941192627, 150.0, 292.0, 0.9466666579246521, 0.02666666731238365, 0.019779743626713753, 2.0]
input_feature.pop(-1)

scheme = model_predict.model_predict(input_feature)

print("Test done! Predicted scheme:")
print(scheme)
