from simple_trainer import AudioClassifierVectorDatasetPL, LightningWordClassifier
from pytorch_lightning import seed_everything
import torch
import tqdm
from nni.compression.pytorch.utils.counter import count_flops_params

from nni.compression.pytorch.pruning import (
    LinearPruner,
    AGPPruner,
    LotteryTicketPruner
)

seed_everything(4)

pl_model = LightningWordClassifier()
pl_model = pl_model.load_from_checkpoint("/home/captain-america/external_disk/.eff/AttentiveMobileWord-Trainer/resnet_50_noise/epoch=49-val_top1=0.7217.ckpt")
pl_model = pl_model.to("cuda:0")

pl_loader = AudioClassifierVectorDatasetPL()
pl_loader.setup(23)

train_loader = pl_loader.train_dataloader()
test_loader = pl_loader.test_dataloader()
iterations = 0

original_flops, _, _  = count_flops_params(pl_model.pytorch_model.feature_network, torch.rand(2,1,149,64).to("cuda:0"), verbose=False) 

def evaluator(feature_network):
    global iterations
    pl_model.pytorch_model.feature_network = feature_network
    pl_model.eval()

    output = 0
    count = 0
    with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False):
        for batch in tqdm.tqdm(test_loader) :
            batch = [batch[0].to("cuda:0"), batch[1].to("cuda:0")]
            x_norm = pl_model.min_max_normalize(batch[0])
            logits = pl_model.forward(x_norm)
            accuracies = pl_model.topk_accuracy(logits, torch.squeeze(batch[1]), 101, mode = "test")
            output += accuracies["test_top1"]
            count+=1
    print(output, count)
    fraction = output/count
    iterations+=1

    current_flops, _, _  = count_flops_params(pl_model.pytorch_model.feature_network, torch.rand(2,1,149,64).to("cuda:0"), verbose=False) 
    torch.save(pl_model, f"workspace/model_{iterations}_{100*current_flops/original_flops:.2f}%_{100*fraction:.4f}%.pt")
    return fraction

current_accuracy = evaluator(pl_model.pytorch_model.feature_network)
print("Current_accuracy:",100*current_accuracy)

def finetuner(feature_network):
    pl_model.pytorch_model.feature_network = feature_network
    pl_model.train()


    with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False,):

        optimizer = torch.optim.Adam(
            pl_model.parameters(), 
            lr=1e-3,
            eps=1e-4
        )

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            batch = [batch[0].to("cuda:0"), batch[1].to("cuda:0")]
            loss = pl_model.training_step(batch, 0)
            loss.backward()
            optimizer.step()
    evaluator(pl_model.pytorch_model.feature_network)

#finetuner(pl_model.pytorch_model.feature_network)

kw_args = {
    "pruning_algorithm":"l2",
    "total_iteration":50,
    "evaluator":evaluator,
    "finetuner":finetuner,
    "speedup":True,
    "dummy_input":torch.rand((2,1,149,64)).to("cuda:0")
}

pruner = LotteryTicketPruner(
    pl_model.pytorch_model.feature_network,
    [
        {
            "op_types":["Conv2d"],
            "total_sparsity":0.5
        },
        {
            "exclude":True,
            "op_names":"layer4.2.conv3"
        }
    ],
    **kw_args
)

pruner.compress()