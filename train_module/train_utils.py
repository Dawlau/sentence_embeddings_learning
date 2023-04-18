import torch
from tqdm import tqdm


def train_step(encoder, classifier, dataloader, optimizer, loss_fn, device):
    encoder.train()
    classifier.train()

    total_loss = 0

    all_predictions = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)

    for premise, hypothesis, label in tqdm(dataloader):
        premises = premise.to(device)
        hypotheses = hypothesis.to(device)
        labels = label.to(device)

        optimizer.zero_grad()

        u = encoder(premises)
        v = encoder(hypotheses)

        encoder_output = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

        preds = classifier(encoder_output)

        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss * premises.shape[0]

        all_predictions = torch.cat([
            all_predictions,
            preds
        ])

        all_labels = torch.cat([
            all_labels,
            labels
        ])

    accuracy = (all_predictions.argmax(dim=-1) == all_labels).float().mean()

    return total_loss, accuracy


def validation_step(encoder, classifier, dataloader, loss_fn, device):
    encoder.eval()
    classifier.eval()

    total_loss = 0

    all_predictions = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)

    with torch.no_grad():
        for premise, hypothesis, label in tqdm(dataloader):
            premises = premise.to(device)
            hypotheses = hypothesis.to(device)
            labels = label.to(device)

            u = encoder(premises)
            v = encoder(hypotheses)

            encoder_output = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

            preds = classifier(encoder_output)

            loss = loss_fn(preds, labels)

            total_loss += loss * premises.shape[0]

            all_predictions = torch.cat([
                all_predictions,
                preds
            ])

            all_labels = torch.cat([
                all_labels,
                labels
            ])

    accuracy = (all_predictions.argmax(dim=-1) == all_labels).float().mean()

    return total_loss, accuracy
