import torch
import numpy as np
import configargparse
import resnext
import time
import sys
from torch import optim as optim
import torch.nn as nn
from apex import amp

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        default_config_files=[""], auto_env_var_prefix="veesion_"
    )

    "----------------------------- Modality -----------------------------"

    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-w", "--window", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=20)
    parser.add_argument("-ss", "--sample_size", type=int, default=224)
    parser.add_argument("-sd", "--sample_duration", type=int, default=35)
    parser.add_argument("-ag", "--all_gpus", action="store_true")
    parser.add_argument("-t", "--transfer", action="store_true")
    parser.add_argument("-bt", "--bench_train", action="store_true")
    normalization.initNorm("BN")

    config = parser.parse_args(sys.argv[1:])
    if config.all_gpus:
        config.device = None
    data = np.random.choice(
        256,
        size=(
            config.batch_size,
            3,
            config.sample_duration,
            config.sample_size,
            config.sample_size,
        ),
    )
    if not config.transfer:
        inputs = torch.from_numpy(data).cuda(config.device).float().div(255.0).add(-0.5)
    labels = torch.from_numpy(np.random.choice(10, size=config.batch_size)).cuda(
        config.device
    )
    model = resnext.resnet101(
        num_classes=10,
        sample_size=config.sample_size,
        sample_duration=config.sample_duration,
    ).cuda(config.device)
    if config.bench_train:
        model.train()
    else:
        model.eval()
    if config.all_gpus:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().cuda(config.device)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    t1 = time.time()
    its = 0
    vs = []
    while 1:
        if config.transfer:
            inputs = (
                torch.from_numpy(data).cuda(config.device).float().div(255.0).add(-0.5)
            )
        if config.bench_train:
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(inputs)

        vs.append((config.batch_size, time.time()))
        vts = [x for x in vs[-config.window :]]
        if its % 5 == 1 and its > 5:
            print(
                int(1000 * (vts[-1][1] - vts[0][1]) / np.sum([x[0] for x in vts])),
                "ms",
                np.sum([x[0] for x in vts]) / (vts[-1][1] - vts[0][1]),
                "samples",
            )
        its += 1
