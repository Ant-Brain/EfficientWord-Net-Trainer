from simple_trainer import AudioClassifierVectorDatasetPL, LightningWordClassifier
data_loader_pl = AudioClassifierVectorDatasetPL(batch_size=2**10)

pl_model = LightningWordClassifier()
pl_model = pl_model.load_from_checkpoint(
    "/home/captain-america/external_disk/.eff/AttentiveMobileWord-Trainer/resnet_50_noise/epoch=49-val_top1=0.7217.ckpt"
)
pl_model = pl_model.eval()
from decimal import Decimal
from aimet_torch.defs import GreedySelectionParameters, ChannelPruningParameters
from aimet_common.defs import CompressionScheme, CostMetric
from contextlib import redirect_stdout

from IPython.utils import io
import pytorch_lightning as pl

greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.95),
                                          num_comp_ratio_candidates=10,)

data_loader_pl.setup(23)

#modules_to_ignore = [pl_model.pytorch_model.feature_network.fc]
modules_to_ignore = [pl_model.pytorch_model.feature_network.conv1] #never prune the first conv

auto_params = ChannelPruningParameters.AutoModeParams(greedy_select_params=greedy_params,modules_to_ignore=modules_to_ignore)
#params = auto_params
data_loader  =  data_loader_pl.train_dataloader()
print("number of samples", len(data_loader))
params = ChannelPruningParameters(data_loader=data_loader,
                                  num_reconstruction_samples=0,
                                  allow_custom_downsample_ops=False,
                                  mode=ChannelPruningParameters.Mode.auto,
                                  params=auto_params,)

pl_trainer = pl.Trainer(
    precision=16,
    accelerator="gpu",
    devices=1,
    deterministic=True,
)
def eval_callback(model, iterations, use_cuda:bool):
    global pl_trainer
    
    #print("args :", iterations, use_cuda)
    pl_model = LightningWordClassifier()
    pl_model.pytorch_model.feature_network = model

    #text_trap = io.StringIO()
    #with redirect_stdout(text_trap) :

    #with io.capture_output() as captured:
    if True  :
        if iterations ==  None :

            pl_trainer = pl.Trainer(
                precision=16,
                accelerator="gpu",
                devices=1,
                deterministic=True,
            )
        else:
            pl_trainer.limit_test_batches = iterations
        results = pl_trainer.test(model = pl_model, dataloaders = data_loader_pl,verbose = True)
    return results[0]['test_top1']

eval_iterations = 10
compress_scheme = CompressionScheme.channel_pruning
cost_metric = CostMetric.mac

from aimet_torch.compress import ModelCompressor
from aimet_torch.compression_factory import CompressionFactory
from aimet_common.bokeh_plots import BokehServerSession
from aimet_pruner_utils import CustomCompressionFactory
#bokeh_session = BokehServerSession(url = "", session_id="compression")

algo = CustomCompressionFactory.create_channel_pruning_algo(
    model=pl_model.pytorch_model.feature_network,
    eval_callback=eval_callback,
    eval_iterations=eval_iterations,
    input_shape=(1,1,149, 64),
    cost_metric=cost_metric,
    params = params,
    bokeh_session=None,
    min_comp_ratio = 0.85
)

print(algo._comp_ratio_select_algo._comp_ratio_candidates)
print(type(algo._comp_ratio_select_algo))
compressed_model, comp_stats = algo.compress_model(
    cost_metric=cost_metric,trainer = None
)
print(comp_stats)