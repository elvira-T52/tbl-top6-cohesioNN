import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from _datasetPreparer import prepare_datasets, CONTINUOUS_FEATURES
from _rnn import CohesioNN


datasets = prepare_datasets()

vocab_sizes = {family: len(vocab) for family, vocab in datasets["vocabs"].items()}
embed_dims = {"typeDescKey": 8, "zoneCode": 4, "shotType": 6,
              "teamStrength": 4, "opponent": 12, "playerName": 16}

train_loader = DataLoader(datasets["train"], batch_size=32, shuffle=True)
val_loader = DataLoader(datasets["val"], batch_size=32)


def goal_boost_weights(label_tensor, num_classes, goal_idx, boost=None):
    # Every class stays at weight 1.0 EXCEPT goal_idx, which gets boosted --
    # unlike a full "balanced" scheme, this only targets goals specifically
    # rather than also inflating every other rare class (takeaway, giveaway).
    # If boost isn't given, default to how many times more common the most
    # frequent class is than goal -- e.g. if faceoff outnumbers goal 5x,
    # goal's loss contribution gets multiplied by 5x to compensate.
    counts = torch.bincount(label_tensor, minlength=num_classes).float().clamp(min=1)
    if boost is None:
        boost = (counts.max() / counts[goal_idx]).item()
    weights = torch.ones(num_classes)
    weights[goal_idx] = boost
    return weights

## Evaluate typical 
# batch in loader (either val or train):
# inputs, labels = batch
# outputs = model(inputs)
# loss = loss_fn(outputs, labels)
##

def evaluate(model, loader, actor_loss_fn, type_loss_fn, goal_idx):
    # model.eval() must be set by the caller -- this just runs the loader
    # without touching gradients, and reports loss + accuracy per head.
    # loss = loss_fn(outputs, labels)


    total_loss = 0.0
    actor_correct = 0
    type_correct = 0
    n = 0
    goal_true_positives = 0  # predicted goal AND actually was a goal
    goal_predicted = 0       # predicted goal (right or wrong)
    goal_actual = 0          # actually was a goal (predicted or not)
    with torch.no_grad():
        for batch in loader:
            #forward pass for type and player
            actor_logits, type_logits = model(batch)
            # loss for actor that compares the model's actor guess and the batch labels of the actor
            actor_loss = actor_loss_fn(actor_logits, batch["label_actor"])
            #loss for play type that compares the models' predicted play and the actual label.
            type_loss = type_loss_fn(type_logits, batch["label_type"])
            #just need batch size for both player and play, shouuld be the same, but just grab label actor, important for loss
            batch_size = len(batch["label_actor"])

            ##total loss is the combined average of this one batch. CrossEntropyLoss is already the mean over the batch,
            ##So * batch_size just negates the mean.
            ##Turns average loss per example in this batch back into total loss for this batch.
            total_loss += (actor_loss + type_loss).item() * batch_size


            ##For each example in the batch, picks the class index the model scored highest, (batch dize, 67 (67 possible players))
            actor_correct += (actor_logits.argmax(dim=-1) == batch["label_actor"]).sum().item()
            ## == batch["label_actor"]).sum().item(), which compares element wise what the model predicted and the real label.
            type_pred = type_logits.argmax(dim=-1)
            type_true = batch["label_type"]
            type_correct += (type_pred == type_true).sum().item()

            # recall: of all the REAL goals in this batch, how many did we catch?
            # precision: of all the times we PREDICTED goal, how many actually were?
            goal_true_positives += ((type_pred == goal_idx) & (type_true == goal_idx)).sum().item()
            goal_predicted += (type_pred == goal_idx).sum().item()
            goal_actual += (type_true == goal_idx).sum().item()

            ##running count of batches processed so it can get to batch size.
            n += batch_size

    goal_recall = goal_true_positives / goal_actual if goal_actual > 0 else 0.0
    goal_precision = goal_true_positives / goal_predicted if goal_predicted > 0 else 0.0

    return total_loss / n, actor_correct / n, type_correct / n, goal_recall, goal_precision
    ##returns total loss/n which is average loss per example, actor correct/32, and type correct/32, essentially how many we got right of the entire batch.


def train(model, train_loader, val_loader, goal_idx, epochs=20, lr=1e-3, type_class_weights=None):

    #
    actor_loss_fn = nn.CrossEntropyLoss()
    type_loss_fn = nn.CrossEntropyLoss(weight=type_class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()  # enables dropout
        for batch in train_loader:
            optimizer.zero_grad()
            actor_logits, type_logits = model(batch)
            loss = (
                actor_loss_fn(actor_logits, batch["label_actor"])
                + type_loss_fn(type_logits, batch["label_type"])
            )
            loss.backward()
            optimizer.step()

        model.eval()  # disables dropout
        val_loss, actor_acc, type_acc, goal_recall, goal_precision = evaluate(
            model, val_loader, actor_loss_fn, type_loss_fn, goal_idx
        )
        print(f"epoch {epoch + 1}/{epochs}  val_loss={val_loss:.3f}  "
              f"actor_acc={actor_acc:.3f}  type_acc={type_acc:.3f}  "
              f"goal_recall={goal_recall:.3f}  goal_precision={goal_precision:.3f}")


if __name__ == "__main__":
    goal_idx = datasets["vocabs"]["typeDescKey"]["goal"]
    type_class_weights = goal_boost_weights(
        datasets["train"].label_type, vocab_sizes["typeDescKey"], goal_idx
    )
    print(f"goal class weight: {type_class_weights[goal_idx]:.2f} (1.0 = unweighted)")

    model = CohesioNN(embed_dims, vocab_sizes, num_continuous=len(CONTINUOUS_FEATURES))
    train(model, train_loader, val_loader, goal_idx, type_class_weights=type_class_weights)

    torch.save(model.state_dict(), "cohesion_nn.pt")
    print("saved model to cohesion_nn.pt")
