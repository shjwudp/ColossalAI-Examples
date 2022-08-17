from colossalai.amp import AMP_TYPE
from titans.loss.lm_loss import GPTLMLoss
from titans.model.gpt import _create_gpt_model
#from model_zoo.gpt.gpt import gpt2_small_pipeline
from torch.optim import Adam


BATCH_SIZE = 256
SEQ_LEN = 2048
NUM_EPOCHS = 1
HIDDEN_SIZE = 1024
NUM_MICRO_BATCHES = 256
PIPELINE = 2
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)


def gpt3_350M(**kwargs):
    model_kwargs = dict(hidden_size=HIDDEN_SIZE, depth=24, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)

optimizer = dict(
    type=Adam,
    lr=3.0e-4,
    weight_decay=1e-1,
)

fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

loss = dict(
    type=GPTLMLoss,
)

model = dict(
    type=gpt3_350M,
    checkpoint=False,
    max_position_embeddings=SEQ_LEN,
    fuse_scale_mask_softmax=True,
)

parallel = dict(
    pipeline=PIPELINE,
    tensor=dict(size=1, mode=None),
)