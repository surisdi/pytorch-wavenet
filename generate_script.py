import librosa
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *

dataset_name = 'LJSpeech'
snapshot_path = f'/scratch/gobi1/didacsuris/checkpoints/multimodal_seung/snapshots_{dataset_name}'
snapshot_name = f'chaconne_{dataset_name}_model'

model = load_latest_model_from(snapshot_path, use_cuda=True)

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

divisions = ['train', 'valid', 'test']
data = {k: WavenetDataset(dataset_file= f'/scratch/gobi1/didacsuris/data/multimodal_seung/{dataset_name}/dataset_{k}.npz',
                          item_length=model.receptive_field + model.output_length - 1, division=k,
                          target_length=model.output_length, train=(k=='train'),
                          file_location=f'/scratch/gobi1/didacsuris/data/multimodal_seung/{dataset_name}/{k}',
                          test_stride=500) for k in divisions}

print('the dataset has ' + str(len(data)) + ' items')

start_data = data['test'][250000][0]
start_data = torch.max(start_data, 0)[1]


def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")


generated = model.generate_fast(num_samples=16000,
                                first_samples=start_data,
                                progress_callback=prog_callback,
                                progress_interval=1000,
                                temperature=1.0,
                                regularize=0.)

print(generated)
librosa.output.write_wav('latest_generated_clip.wav', generated, sr=16000)