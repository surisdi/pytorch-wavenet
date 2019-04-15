import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from scipy.io import wavfile

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu', flush=True)
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

current_time = datetime.now().strftime('%b%d_%H-%M-%S')

model = WaveNetModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     embedding_dim=32,
                     dtype=dtype,
                     bias=True)

# dataset_name = 'LJSpeech'
dataset_name = 'SoundPixels'
snapshot_path = f'/scratch/gobi1/didacsuris/checkpoints/multimodal_seung/snapshots_{dataset_name}'
snapshot_name = f'{dataset_name}_lr001_model'
# dataset_path = '/scratch/gobi1/didacsuris/data/multimodal_seung'
dataset_path = '/scratch/gobi2/didacsuris/data'

# model = load_latest_model_from(snapshot_path, use_cuda=True)
# if os.path.exists(snapshot_path + '/' + snapshot_name + '.pth.tar'):
#     model = torch.load(snapshot_path + '/' + snapshot_name + '.pth.tar')

if use_cuda:
    print("move model to gpu", flush=True)
    model.cuda()

print('model: ', model, flush=True)
print('receptive field: ', model.receptive_field, flush=True)
print('parameter count: ', model.parameter_count(), flush=True)

divisions = ['train', 'valid', 'test']
data = {k: WavenetDataset(dataset_file=f'{dataset_path}/{dataset_name}/dataset_{k}.npz',
                          item_length=model.receptive_field + model.output_length - 1, division=k,
                          target_length=model.output_length, train=(k=='train'),
                          file_location=f'{dataset_path}/{dataset_name}/{k}',
                          test_stride=500) for k in divisions}
for k in divisions:
    print(f'the {k} dataset has ' + str(len(data[k])) + ' items', flush=True)


def generate_and_log_samples(step):
    sample_length=32000
    gen_model = load_latest_model_from(snapshot_path, use_cuda=False)
    print("start generating...", flush=True)
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated", flush=True)


logger = TensorboardLogger(log_interval=200,  # 200
                           validation_interval=400,  # 400
                           generate_interval=800,  # 800
                           generate_function=generate_and_log_samples,
                           log_dir=f"logs/{dataset_name}_{current_time}_lr001_model'")

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.001,
                         weight_decay=0.0,
                         snapshot_path=snapshot_path,
                         snapshot_name=snapshot_name,
                         snapshot_interval=500,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...', flush=True)
trainer.train(batch_size=16,
              epochs=10,
              continue_training_at_step=0)
