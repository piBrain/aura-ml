from model import Seq2Seq
from tensorflow import estimator
from sys import argv
from pprint import pprint
INPUT_VOCAB_SIZE = 0
OUTPUT_VOCAB_SIZE = 0
EMBED_DIM = 0
NUM_UNITS = 0
INP_SEQ_LEN = 0
OUT_SEQ_LEN = 0
BATCH_SIZE = 20


def model_fn(features, labels, mode):
    model = Seq2Seq(
            BATCH_SIZE, features, labels,
            INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, EMBED_DIM,
            mode
    )

    enc_out, enc_state = model.encode(NUM_UNITS, INP_SEQ_LEN)

    if mode == estimator.ModeKeys.TRAIN:
        t_out, _ = model.decode(NUM_UNITS, OUT_SEQ_LEN)
        spec = model.prepare_train(t_out, OUT_SEQ_LEN, labels, 0.03)
    if mode == estimator.ModeKeys.PREDICT:
        _, sample_id = model.decode(NUM_UNITS, OUT_SEQ_LEN)
        spec = model.prepare_predict(sample_id)
    return spec


def print_predictions(predictions):
    for pred in predictions:
        pprint(pred)


def main():
    features = None
    labels = None
    input_fn = estimator.inputs.numpy_input_fn(
        x=features,
        y=labels,
        num_epochs=None,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    classifier = estimator.Estimator(
        model_fn=model_fn,
        model_dir='./saves',
        params=None
    )
    if len(argv) < 2 or argv[1] == 'train':
        classifier.train(input_fn)
    elif argv[1] == 'predict':
        predictions = classifier.predict()
        print_predictions(predictions)
    else:
        print('Unknown Operation.')

if __name__ == '__main__':
    main()
