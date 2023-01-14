import os
import torch
from queue import Queue
import logging
import time
from pathlib import Path
# Initializing a queue
q = Queue(maxsize=3)

def save_checkpoint(model, best_acc, val_acc, epoch, path='./checkpoint/ckpt_'):
    # Save checkpoint.
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, path + "latest" + '.pth')

    if best_acc == None:
        return val_acc

    elif best_acc > val_acc:
        l.info(f'Saving {path+ str(round(float(val_acc),6))} ckp ')
        torch.save(state, path + str(round(float(val_acc),6)) + '.pth')
        q.put(path + str(round(float(val_acc),6)) + '.pth')
        if q.full():
            tmp = q.get()
            os.remove(tmp)
            l.info(f'Delete {tmp} ckp')
        return val_acc
    return best_acc

def load_checkpoint(model:torch.nn.Module,path='checkpoint/ckpt_latest.pth'):
    # Load checkpoint.
    l.info(f'==> load checkpoint from {path}')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'],strict=False)
    start_epoch = checkpoint['epoch']
    return model

def setup_logger():
    l = logging.getLogger('l')

    log_dir: Path = Path(__file__).parent / "logs"

    # create log directory if not exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # set log file name
    log_file_name = f"{time.strftime('%Y%m%d_%H%M%S')}.log"

    l.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(
        filename=log_dir / log_file_name,
        mode='w'
    )
    streamHandler = logging.StreamHandler()

    allFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )

    fileHandler.setFormatter(allFormatter)
    fileHandler.setLevel(logging.INFO)

    streamHandler.setFormatter(allFormatter)
    streamHandler.setLevel(logging.INFO)

    l.addHandler(streamHandler)
    l.addHandler(fileHandler)

    return l

l = setup_logger()
