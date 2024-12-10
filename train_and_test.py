import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def evaluator(y_true, logits, multilabel=False):
    y_pred = logits > 0 if multilabel else logits.argmax(axis=1)
    micro_f1 = float(f1_score(y_true, y_pred, average='micro'))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    return 100 * micro_f1, 100 * macro_f1
    

def train_mlp(model, feats, labels, batch_size, optimizer, device, multilabel=False):
    fn = torch.sigmoid if multilabel else lambda x: x
    criterion = F.binary_cross_entropy if multilabel else F.cross_entropy

    feats = feats.to(device)
    labels = labels.to(device)

    num_batches = (feats.shape[0] + batch_size - 1) // batch_size
    idx_random = torch.randperm(feats.shape[0])
    idx_batch = [idx_random[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    model.train()
    total_loss = 0
    for i in range(num_batches):
        output = fn(model(feats[idx_batch[i]]))
        y_true = labels[idx_batch[i]]
        loss = criterion(output, y_true)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / num_batches


def test_mlp(model, feats, labels, batch_size, device, multilabel=False):
    feats = feats.to(device)
    num_batches = (feats.shape[0] + batch_size - 1) // batch_size

    logits = []
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            logits.append(model(feats[i * batch_size:
                                      (i + 1) * batch_size]))
    y_true = labels.cpu().numpy()
    logits = torch.cat(logits).cpu().numpy()
    
    return evaluator(y_true, logits, multilabel)


def train_gnn(model, dataloader, optimizer, device, multilabel=False):
    model.train()
    total_loss = 0
    fn = torch.sigmoid if multilabel else lambda x: x
    criterion = F.binary_cross_entropy if multilabel else F.cross_entropy

    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [block.to(device) for block in blocks]
        x = blocks[0].srcdata['feat']
        y_true = blocks[-1].dstdata['label']
        output = fn(model(blocks, x))
        loss = criterion(output, y_true)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def test_gnn(model, dataloader, device, multilabel=False):
    model.eval()
    y_true = []
    logits = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            x = blocks[0].srcdata['feat']
            y_true.append(blocks[-1].dstdata['label'])
            logits.append(model(blocks, x))
    y_true = torch.cat(y_true).cpu().numpy()
    logits = torch.cat(logits).cpu().numpy()

    return evaluator(y_true, logits, multilabel)


def train_rgnn(model, target, dataloader, optimizer, device, multilabel=False):
    model.train()
    total_loss = 0
    fn = torch.sigmoid if multilabel else lambda x: x
    criterion = F.binary_cross_entropy if multilabel else F.cross_entropy

    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [block.to(device) for block in blocks]
        x = blocks[0].srcdata['feat']
        y_true = blocks[-1].dstdata['label'][target]
        output = fn(model(blocks, x))
        loss = criterion(output, y_true)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def test_rgnn(model, target, dataloader, device, multilabel=False):
    model.eval()
    y_true = []
    logits = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            x = blocks[0].srcdata['feat']
            y_true.append(blocks[-1].dstdata['label'][target])
            logits.append(model(blocks, x))
    y_true = torch.cat(y_true).cpu().numpy()
    logits = torch.cat(logits).cpu().numpy()

    return evaluator(y_true, logits, multilabel)
