from nni.compression.pytorch.pruning import L2NormPruner, L1NormPruner
from nni_pruner_filter_lottery_ticket import LightningWordClassifier
import torch
import copy
from tqdm import tqdm
import torch
from nni.compression.pytorch import ModelSpeedup

pl_model = LightningWordClassifier()
pl_model = pl_model.load_from_checkpoint("/home/captain-america/external_disk/.eff/AttentiveMobileWord-Trainer/resnet_50_noise/epoch=49-val_top1=0.7217.ckpt")
pl_model = pl_model.to("cuda:0")

dummy_input = torch.rand(2,1,149,64).to("cuda:0")

def evaluator(feature_network):
    global pl_model, test_loader, pre_flops, pre_params, iterations
    pl_model_copy = copy.deepcopy(pl_model)
    pl_model_copy.pytorch_model.feature_network = feature_network
    pl_model_copy.eval()

    accuracy_sum = 0
    print("Evaluating...")
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        for i, batch in enumerate(tqdm(test_loader)):
            batch = [batch[0].to("cuda:0"), batch[1].to("cuda:0")]
            
            accuracies = pl_model_copy.get_metric_for_batch(test_batch=batch)
            accuracy_sum += accuracies["test_top1"]
    
    output = accuracy_sum/(i+1)
    print("Accuracy top1", output)
    return accuracy_sum/(i+1)

compressor = L2NormPruner(
    pl_model.pytorch_model.feature_network, [
        {
            "op_types":["Conv2d"],
            "total_sparsity":0.4
        }
    ]
)
print(len(compressor._detect_modules_to_compress()))
layers = compressor._detect_modules_to_compress()
for layer in layers :
    for sparsity_ratio in range(10, 100, 10):
        sparsity_ratio /= 100
        print(sparsity_ratio, dummy_input.shape)
        new_pruner_config = [
            {
                "op_types":["Conv2d"],
                #"op_" : [layer[0].name],
                "total_sparsity" : sparsity_ratio
            }
        ]
        feature_network = pl_model.pytorch_model.feature_network
        pruner = L1NormPruner(
            feature_network,
            new_pruner_config,
        )

        _ , masks = pruner.compress()
        pruner.show_pruned_weights()
        pruner._unwrap_model()
        #pruner._wrap_model()
        print(masks)

        ModelSpeedup(feature_network, dummy_input = (dummy_input,), masks_file = masks).speedup_model()

        accuracy = evaluator(feature_network)
        print(layer[0].name, ": sparsity : ", sparsity_ratio, ": accuracy :", accuracy)